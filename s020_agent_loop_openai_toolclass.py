#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s020_agent_loop_openai_toolclass.py - Agent Loop（OpenAI 版 / 一个工具 = 一个类）

核心循环与 s01 完全一致：

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

与 s01 的唯一区别：工具不再是模块级函数，而是一个类对应一个工具
（写法参考 s020/simple_react_agent.py 的 BaseTool + ToolRegistry）
    1. BaseTool：父类，子类只写 name + execute()；to_openai_schema() 内部调用
       tool_description_generator.generate_tool_description，从 execute 的类型注解
       和 docstring（:param xxx:）自动生成 description / parameters，与 s01 一致。
    2. RunBashTool(BaseTool)：唯一的工具，就是 s01 里的 run_bash，
       共享配置（workdir / timeout / 危险命令黑名单）放在 self 上。
    3. ToolRegistry：register(工具实例) / get / execute / get_definitions，
       出错时返回 "Error: ..." 文本，照样喂回模型。
    4. agent_loop 与 s01 同构，只是分发从 TOOL_MAP 换成 REGISTRY。

扩展方式：新写一个 class XxxTool(BaseTool) -> REGISTRY.register(XxxTool())，
循环和下面两行常量都不用动。

用法：
    pip install openai python-dotenv
    python s020_agent_loop_openai_toolclass.py
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# 系统提示词里给模型看的「星期几」，中文比 %A 的英文更省 token 也更直观
_WEEKDAYS = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 工具描述不再手写，交给 generate_tool_description 从 execute() 的签名 + docstring 自动生成
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

def build_system() -> str:
    """生成系统提示词，里面带上「此刻」的日期与时间

    时间放在这里而不是写死在常量里，是因为 REPL 一轮对话可能跑很久，
    跨天后旧的日期会把模型带偏；需要最新时间时重新调用本函数即可。
    """
    now = datetime.now()
    stamp = f"{now.strftime('%Y-%m-%d %H:%M:%S')} {_WEEKDAYS[now.weekday()]}"
    return (
        f"You are a coding agent at {os.getcwd()}. "
        f"Current date and time: {stamp}. "
        "Use the bash tool to solve tasks. Act, don't explain. "
        "The shell is Windows cmd.exe: use dir / type / echo / findstr / copy / del, "
        "never Unix commands like ls / cat / grep / wc."
    )


SYSTEM = build_system()  # 模块级快照 = 本文件被 import 那一刻的时间；要最新时间请调 build_system()

MAX_ROUNDS = 40  # 安全上限，防止模型不停调用工具烧钱


# ---------- 公共小工具（与工具定义方式无关，保持和 s01 一致） ----------
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


# ---------- ① BaseTool：所有工具的父类 ----------
class BaseTool:
    """工具父类：一个类 = 一个工具

    子类只需要：
        name      -> 暴露给大模型的工具名（必填，因为方法名固定叫 execute）
        execute() -> 真正干活的方法，写清类型注解和 docstring 即可

    schema 全自动：to_openai_schema() 用 generate_tool_description 去解析
    self.execute（bound method，签名里已经没有 self），自动拿到：
        description -> docstring 第一行
        parameters  -> 类型注解 + :param xxx: 说明 + 有无默认值 => required

    想手工覆盖时，才在子类里写同名类属性：
        description -> 覆盖自动生成的描述
        parameters  -> 完全手写 JSON Schema
    """

    name: str = ""
    description: str = ""       # 可选覆盖，留空则用 execute 的 docstring
    parameters: Dict[str, Any] = {}  # 可选覆盖，留空则自动生成

    def execute(self, **kwargs: Any) -> str:
        """执行工具并返回字符串结果（出错就返回 Error: ... 文本）"""
        raise NotImplementedError

    def to_openai_schema(self) -> Dict[str, Any]:
        """转成 OpenAI 规范的 tools 条目

        generate_tool_description 返回的已经是完整的 OpenAI tools 条目
        （type / function.name / function.description / function.parameters），
        这里不重建字典，只在需要覆盖时就地改字段，避免丢掉生成器新增的内容。
        """
        # bound method：self 已绑定，sig/type hints/docstring 都和 s01 的普通函数一样解析
        spec = generate_tool_description(self.execute)
        # 唯一的必改项：bound method 的 __name__ 恒为 "execute"，工具名得用类属性
        spec["function"]["name"] = self.name or type(self).__name__
        if self.description:                 # 有手写描述才覆盖自动生成的 docstring 首行
            spec["function"]["description"] = self.description
        if self.parameters:                  # 有手写 JSON Schema 才覆盖自动生成的参数
            spec["function"]["parameters"] = self.parameters
        return spec


# ---------- ② ToolRegistry：名字 -> 工具实例 ----------
class ToolRegistry:
    """管理所有工具类实例（写法参考 simple_react_agent.py）"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> "ToolRegistry":
        """注册一个工具实例，返回 self 以便链式调用"""
        if not tool.name:
            raise ValueError(f"Tool class {type(tool).__name__} 缺少 name")
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Optional[BaseTool]:
        """按名字取工具，没有则返回 None"""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """工具是否已注册"""
        return name in self._tools

    def execute(self, name: str, params: Dict[str, Any]) -> str:
        """执行工具；未知工具或工具内部异常都转成 Error 文本回给模型"""
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: Unknown tool {name}"
        try:
            return tool.execute(**(params or {}))
        except Exception as e:  # 工具本身炸了（含参数给错）也要把错误喂回模型
            return f"Error: {e}"

    def get_definitions(self) -> List[Dict]:
        """发给大模型的工具描述列表"""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def as_map(self) -> Dict[str, BaseTool]:
        """工具名 -> 工具实例（保持和 s01 的 TOOL_MAP 同名同用途）"""
        return dict(self._tools)


# ---------- ③ 具体的工具类：一个类对应一个工具 ----------
class RunBashTool(BaseTool):
    """执行一条 shell 命令并返回它的合并输出（Windows cmd.exe）"""

    name = "run_bash"
    # description / parameters 都不写：交给 execute() 的 docstring 与类型注解自动生成

    def __init__(self, workdir: str = None, timeout: int = 120):
        # 共享配置挂在 self 上，这就是「用类写工具」相对散函数的最大好处
        self.workdir = workdir or os.getcwd()
        self.timeout = timeout
        self.dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/", "format "]

    def execute(self, command: str) -> str:
        """执行一条 shell 命令并返回它的输出

        :param command: 要执行的 cmd 命令，例如 dir /b *.py、type a.txt、echo hi > a.txt
        :return: 命令的标准输出与错误输出（合并），无输出时返回占位文本
        """
        if any(d in command for d in self.dangerous):
            return "Error: Dangerous command blocked"
        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=self.workdir,
                capture_output=True,
                timeout=self.timeout,
            )
            # 拿到原始字节后自己解码：Windows 中文输出多为 GBK，直接按 utf-8 解会得到替换符
            raw = (r.stdout or b"") + (r.stderr or b"")
            out = _decode(raw).strip()
            if not out:
                return f"(no output, exit code {r.returncode})"
            return out[:50000]
        except subprocess.TimeoutExpired:
            return f"Error: Timeout ({self.timeout}s)"
        except (FileNotFoundError, OSError) as e:
            return f"Error: {e}"


# ---------- 注册工具，产出 agent_loop 需要的两份数据 ----------
REGISTRY = ToolRegistry()
REGISTRY.register(RunBashTool(os.getcwd()))

TOOL_MAP = REGISTRY.as_map()      # 工具名 -> 工具实例（对应 s01 里的 名字->函数）
TOOLS = REGISTRY.get_definitions()  # OpenAI 规范的工具描述列表


# ---------- 核心 Agent Loop（与 s01 同构，只把分发换成 REGISTRY） ----------
def agent_loop(
    messages: list,
    *,
    client=None,                      # 可外部注入：自己的 key / 别的 OpenAI 兼容服务 / 测试用假 client
    registry: ToolRegistry = None,    # 可外部注入：想用自己的工具集就传进来
    tools: list = None,               # 默认由 registry 生成 OpenAI 规范的工具描述
    model: str = None,                # 默认用模块级 MODEL
    max_rounds: int = None,           # 默认用模块级 MAX_ROUNDS
    max_tokens: int = 8000,
    verbose: bool = True,             # False 时全程不 print，方便被别的程序调用
) -> str:
    """反复调用 LLM，直到它不再请求工具为止

    外部依赖全部可注入，所以本文件被 import 后可以直接复用这个函数；
    什么都不传时行为与原来完全一致（用模块级的 client / REGISTRY / MODEL），
    即 agent_loop(messages) 这种老调用方式不受影响。

    :param messages: 会话消息列表，会被原地追加 assistant / tool 消息
    :param client: OpenAI 兼容客户端，不传则用本模块顶部的 client
    :param registry: 工具注册表，不传则用本模块的 REGISTRY
    :param tools: OpenAI 规范的工具描述列表，不传则取 registry.get_definitions()
    :param model: 模型名，不传则用本模块的 MODEL
    :param max_rounds: 安全轮数上限，不传则用本模块的 MAX_ROUNDS
    :param max_tokens: 单次回复的 token 上限
    :param verbose: 是否打印模型回复、命令与工具输出
    :return: 模型最后一轮的纯文本回复；超过轮数上限时返回 [Stopped] 提示
    """
    # 没显式传的，退回模块级默认值；注意 client 与模块级同名，只能走 globals() 取
    if client is None:
        client = globals()["client"]
    if registry is None:
        registry = REGISTRY
    if tools is None:
        tools = registry.get_definitions()
    if model is None:
        model = MODEL
    if max_rounds is None:
        max_rounds = MAX_ROUNDS

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=max_tokens,
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
            if verbose and message.content:
                print(message.content)

            func_name = tc.function.name
            try:
                func_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                func_args = {}

            if not registry.has(func_name):
                output = f"Error: Unknown tool {func_name}"
                if verbose:
                    print(f"\033[31m! {output}\033[0m")
            else:
                if verbose:
                    print(f"\033[33m$ {func_args.get('command', '')}\033[0m")
                output = registry.execute(func_name, func_args)
                if verbose:
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
    print("s020: Agent Loop (OpenAI) - 一个工具一个类")
    print(f"model  = {MODEL}")
    print(f"base   = {client.base_url}")
    print(f"cwd    = {os.getcwd()}")
    print(f"tools  = {', '.join(TOOL_MAP) or '(none)'}")
    print("输入问题并回车发送，输入 q 退出。\n")

    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("\033[36ms020 >> \033[0m")
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
