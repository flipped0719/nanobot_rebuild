"""几个起步用的工具。

每个工具都是普通的 LangChain `@tool` 装饰函数。文件类工具上的工作区路径校验，
是 nanobot `WorkspaceScopeResolver` 的一个极简替代版（完整版留给后面的步骤，
如果最终还需要的话）。
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    raise ValueError(f"不支持的表达式节点: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """计算一个基础算术表达式（+ - * / ** %）并返回结果。"""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except Exception as exc:
        return f"错误：无法计算表达式 '{expression}'：{exc}"


@tool
def get_current_time() -> str:
    """返回当前的 UTC 时间（ISO 8601 格式）。"""
    return datetime.now(timezone.utc).isoformat()


def _resolve_workspace_path(workspace_root: Path, path: str) -> Path:
    target = (workspace_root / path).resolve()
    if target != workspace_root and workspace_root not in target.parents:
        raise ValueError("路径超出了工作区范围，拒绝访问")
    return target


def make_basic_tools(workspace_root: str | Path):
    """构造绑定到配置工作区的基础工具，避免在 import 时误用进程 cwd。"""
    root = Path(workspace_root).resolve()

    @tool("read_text_file")
    def read_text_file(path: str) -> str:
        """读取相对于工作区根目录的一个 UTF-8 文本文件，返回其内容。"""
        try:
            target = _resolve_workspace_path(root, path)
            return target.read_text(encoding="utf-8")
        except Exception as exc:
            return f"读取 '{path}' 出错：{exc}"

    @tool("write_text_file")
    def write_text_file(path: str, content: str) -> str:
        """向相对于工作区根目录的文件写入 UTF-8 文本，并创建所需父目录。"""
        try:
            target = _resolve_workspace_path(root, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"已写入 {len(content)} 个字符到 {path}"
        except Exception as exc:
            return f"写入 '{path}' 出错：{exc}"

    return [calculator, get_current_time, read_text_file, write_text_file]


# 保留旧导入接口；实际运行时会使用配置目录重新构造。
ALL_TOOLS = make_basic_tools(Path.cwd())
