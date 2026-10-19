#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s01_agent_loop_openai.py - Agent Loop（OpenAI 版 / 只保留 run_bash 一个工具）

核心循环与 learn-claude-code s01 完全一致：

    while True:
        response = LLM(messages, tools)
        if 没有 tool_calls: break
        执行工具 -> 把 tool 结果追加进 messages -> 继续

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

与 s01 原版的两点不同：
    1. 大模型改用 openai SDK 调用（chat.completions.create）。
    2. 工具描述用 tool_description_generator.generate_tool_description 自动生成，
       工具结果按 OpenAI 规范用 role="tool" 消息回传（参考《大模型的函数调用方法》）。

用法：
    pip install openai python-dotenv
    python s01_agent_loop_openai.py
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from tool_description_generator import generate_tool_description
except ImportError:
    sys.exit("[Error] 未找到 tool_description_generator.py，请把它放在本文件同级目录。")

from dotenv import load_dotenv
from openai import OpenAI

# Windows 控制台：切到 UTF-8 代码页 + 启用 ANSI 颜色，失败也无妨
if os.name == "nt":
    os.system("chcp 65001 > nul")
    os.system("")
# 统一用 UTF-8 输出并容错，避免 GBK 控制台下 print 抛 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv(override=True)

# ---------- LLM 客户端（OpenAI 兼容协议） ----------
# 默认对接 DeepSeek（课程环境）；换成官方 OpenAI 时设置 OPENAI_BASE_URL=https://api.openai.com/v1
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
)
MODEL = os.getenv("MODEL_ID", "deepseek-chat")

SYSTEM = (
    f"You are a coding agent at {os.getcwd()}. "
    "Use the bash tool to solve tasks. Act, don't explain. "
    "The shell is Windows cmd.exe: use dir / type / echo / findstr / copy / del, "
    "never Unix commands like ls / cat / grep / wc."
)

MAX_ROUNDS = 40  # 安全上限，防止模型不停调用工具烧钱


# ---------- 唯一的工具：run_bash ----------
def _decode(data: bytes) -> str:
    """按 utf-8 -> gbk 顺序解码命令输出，都失败则替换非法字符"""
    for enc in ("utf-8", "gbk"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _sanitize(text) -> str:
    """去掉 Windows 控制台可能引入的孤立代理字符（如 \\udcxx）

    这类字符会让 openai SDK 在 JSON 序列化时报
    UnicodeEncodeError: surrogates not allowed，必须在发往 API 前清洗。
    """
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
            cwd=os.getcwd(),
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


# 工具注册表：函数名 -> 真实函数（按 notebook 的做法，用 map 分发）
TOOL_MAP = {"run_bash": run_bash}

# 用 generate_tool_description 自动生成 OpenAI 规范的工具描述
TOOLS = [generate_tool_description(run_bash)]


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
                print(f"\033[33m$ {func_args.get('command', '')}\033[0m")
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
    print("s01: Agent Loop (OpenAI)")
    print(f"model  = {MODEL}")
    print(f"base   = {client.base_url}")
    print(f"cwd    = {os.getcwd()}")
    print("输入问题并回车发送，输入 q 退出。\n")

    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
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
