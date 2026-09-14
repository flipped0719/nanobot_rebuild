"""立即提交异步子任务的 spawn 工具。"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from ..subagents import SubagentManager


def make_spawn_tool(manager: SubagentManager):
    """构造只负责快速提交任务的 spawn 工具。"""

    @tool
    async def spawn(task: str, config: RunnableConfig) -> str:
        """派生一个子代理异步处理任务，并立即返回 task_id。

        子代理看不到主对话历史，任务描述必须包含完成任务所需的全部信息。结果会在
        后台完成后自动注入当前父会话，无需等待或反复查询。
        """
        configurable = config.get("configurable") or {}
        parent_session_id = configurable.get("thread_id")
        if not isinstance(parent_session_id, str) or not parent_session_id:
            raise ValueError("运行配置缺少有效的 configurable.thread_id")
        record = manager.submit(parent_session_id, task)
        return f"子任务已提交，task_id：{record.task_id}"

    return spawn
