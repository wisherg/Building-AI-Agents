#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s02_simple_file_tools_openai.py - 文件工具（OpenAI 版 / 极简）

s01 的循环一行不改，只把「一个 bash」换成「四个文件工具 + 一张分发表」，
功能与 learn-claude-code s02 的 code.py 对齐（run_read / run_write / run_edit / run_glob）：

    +----------+      +-------+      +-------------------------+
    |   User   | ---> |  LLM  | ---> | Tool Dispatch           |
    |  prompt  |      |       |      | run_bash    -> 跑命令   |
    +----------+      +---+---+      | read_file   -> 读文件   |
                          ^          | write_file  -> 写文件   |
                          |          | edit_file   -> 补丁替换 |
                          +----------+ glob_files  -> 路径匹配 |
                         tool_result +-------------------------+

    + safe_path(): 文件工具一律限制在工作目录内
    + TOOL_MAP 分发，替换 s01 里硬编码的 run_bash 调用

用法：
    pip install openai python-dotenv
    python s02_simple_file_tools_openai.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# tool_description_generator.py 可能在本目录，也可能在上一层目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tool_description_generator import generate_tool_description
except ImportError:
    sys.exit("[Error] 未找到 tool_description_generator.py，请把它放在本文件同级或上层目录。")

from dotenv import load_dotenv
from openai import OpenAI

# Windows 控制台：切到 UTF-8 代码页，避免 print 中文抛 UnicodeEncodeError
if os.name == "nt":
    os.system("chcp 65001 > nul")
    os.system("")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv(override=True)

# ---------- LLM 客户端（OpenAI 兼容协议） ----------
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
)
MODEL = os.getenv("MODEL_ID", "deepseek-chat")

WORKDIR = Path.cwd().resolve()

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain. "
    "Prefer read_file/glob_files over bash, and edit_file over write_file. "
    "The shell is Windows cmd.exe: use dir / type / echo / findstr / copy / del, "
    "never Unix commands like ls / cat / grep / wc."
)

MAX_ROUNDS = 40  # 安全上限，防止模型不停调用工具烧钱


# ---------- s01 保留：bash ----------
def _decode(data: bytes) -> str:
    """按 utf-8 -> gbk 顺序解码命令输出，都失败则替换非法字符"""
    for enc in ("utf-8", "gbk"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _sanitize(text) -> str:
    """去掉 Windows 控制台可能引入的孤立代理字符（如 \\udcxx），否则 openai SDK 序列化会报错"""
    text = "" if text is None else str(text)
    return text.encode("utf-8", errors="replace").decode("utf-8")


def run_bash(command: str) -> str:
    """执行一条 shell 命令并返回它的输出

    :param command: 要执行的 cmd 命令，例如 dir /b *.py、type a.txt、echo hi > a.txt
    :return: 命令的标准输出与错误输出（合并），无输出时返回占位文本
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/", "format "]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=str(WORKDIR),
            capture_output=True,
            timeout=120,
        )
        # 拿到原始字节后自己解码：Windows 中文输出多为 GBK，直接按 utf-8 解会得到替换符
        raw = (r.stdout or b"") + (r.stderr or b"")
        out = _decode(raw).strip()
        if not out:
            return f"(no output, exit code {r.returncode})"
        return out[:50000]
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# ---------- s02 新增：三个文件工具 + 一个路径匹配工具 ----------
def safe_path(p: str) -> Path:
    """解析路径并确保它还在工作目录内

    :param p: 相对或绝对路径
    :return: 落在工作目录内的绝对路径
    """
    path = (WORKDIR / p).resolve()
    if not str(path).lower().startswith(str(WORKDIR).lower()):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def read_file(path: str, offset: int = 1, limit: int = None) -> str:
    """读取文件内容，可用 offset/limit 分页，避免大文件撑爆上下文

    :param path: 文件路径，相对路径基于工作目录
    :param offset: 从第几行开始读，从 1 开始计数
    :param limit: 本次最多读多少行，不传则读到文件末尾
    :return: 文件内容；还有剩余行时给出下一次的 offset 提示
    """
    lines = safe_path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, offset)
    end = len(lines) if not limit else min(len(lines), start + limit - 1)
    out = "\n".join(lines[start - 1:end])
    if end < len(lines):
        out += f"\n... ({len(lines) - end} more lines, continue with offset={end + 1})"
    return out


def write_file(path: str, content: str) -> str:
    """把完整内容写入文件（新建，或整体覆盖已有文件）

    :param path: 目标文件路径，父目录不存在时自动创建
    :param content: 文件的完整内容
    :return: 写入结果
    """
    file_path = safe_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"


def edit_file(path: str, old_text: str, new_text: str) -> str:
    """按上下文做精确补丁替换：给出一段原文，替换成新片段（不用正则，只改这一处）

    :param path: 要修改的文件路径
    :param old_text: 待替换的原文片段，需与文件内容逐字符一致，可跨多行
    :param new_text: 替换后的新内容，传空串表示删除这段原文
    :return: 修改结果
    """
    file_path = safe_path(path)
    text = file_path.read_text(encoding="utf-8")
    if old_text not in text:
        return f"Error: text not found in {path}"
    file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    return f"Edited {path}"


def glob_files(pattern: str) -> str:
    """按 glob 模式匹配文件名或路径，改代码前先定位文件

    :param pattern: glob 模式，例如 **/*.py、*.md、src/**/*.ts；** 递归匹配任意层目录
    :return: 匹配到的相对路径列表，无匹配时返回 (no matches)
    """
    matches = sorted(str(p.relative_to(WORKDIR)) for p in WORKDIR.glob(pattern))
    return "\n".join(matches) if matches else "(no matches)"


# 工具注册表：函数名 -> 真实函数
TOOL_MAP = {
    "run_bash": run_bash,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "glob_files": glob_files,
}

# 用 generate_tool_description 自动生成 OpenAI 规范的工具描述
TOOLS = [generate_tool_description(fn) for fn in TOOL_MAP.values()]


# ---------- 核心 Agent Loop ----------
def agent_loop(messages: list) -> str:
    """反复调用 LLM，直到它不再请求工具为止"""
    for _ in range(MAX_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=8000,
        )
        message = response.choices[0].message

        # 1) 追加 assistant 消息（有工具调用时必须带上 tool_calls）
        assistant_msg = {"role": "assistant", "content": _sanitize(message.content or "")}
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant_msg)

        # 2) 没有工具调用 => 循环结束
        if not message.tool_calls:
            return message.content or ""

        # 3) 执行每个工具调用
        for tc in message.tool_calls:
            if message.content:
                print(message.content)

            func_name = tc.function.name
            try:
                func_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                func_args = {}

            if func_name not in TOOL_MAP:
                output = f"Error: Unknown tool {func_name}"
                print(f"\033[31m! {output}\033[0m")
            else:
                print(f"\033[33m$ {func_name} {str(func_args)[:300]}\033[0m")
                try:
                    output = TOOL_MAP[func_name](**func_args)
                except Exception as e:  # 工具本身炸了也要把错误喂回模型
                    output = f"Error: {e}"
                print(output[:2000])

            # 4) 按 OpenAI 规范把结果回传
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _sanitize(output),
                }
            )

    return "[Stopped] Reached max rounds limit."


# ---------- 入口 ----------
def main():
    print("s02: File Tools (OpenAI, simple) - " + ", ".join(TOOL_MAP))
    print(f"model  = {MODEL}")
    print(f"base   = {client.base_url}")
    print(f"cwd    = {WORKDIR}")
    print("输入问题并回车发送，输入 q 退出。\n")

    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": _sanitize(query)})
        answer = agent_loop(history)
        print(answer)
        print()


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("[Error] 缺少 OPENAI_API_KEY，请在 .env 中配置。")
    main()
