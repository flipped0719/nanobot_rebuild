"""临时子代理的异步任务管理。

子代理不共享主代理的检查点和对话历史。任务结束后，结果通过父会话的待处理
事件队列回送，由父代理在后续轮次消费。
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Protocol

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from .prompts import build_subagent_prompt

_FORBIDDEN_TOOL_NAMES = {"spawn", "create_goal", "update_goal"}


def _utc_now() -> str:
    """返回可直接序列化的 UTC 时间。"""
    return datetime.now(timezone.utc).isoformat()


class SubagentStatus(StrEnum):
    """子任务状态。"""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class SubagentTask:
    """可查询、可序列化的子任务快照。"""

    task_id: str
    parent_session_id: str
    task: str
    status: SubagentStatus
    created_at: str
    completed_at: str | None = None
    result: str | None = None
    error: str | None = None
    injection_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """属性转换为普通字典。"""
        return asdict(self)


class SessionEventQueue(Protocol):
    """子任务结果回送所需的最小会话接口。"""

    async def enqueue(
        self,
        thread_id: str,
        event_type: str,
        content: Any,
        metadata: Mapping[str, Any] | None = None,
        *,
        event_id: str | None = None,
    ) -> Any:
        """向父会话加入一个幂等事件。"""
        ...


class SubagentRunner(Protocol):
    """LangChain agent 所需的最小异步调用接口。"""

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """运行一次临时子代理。"""
        ...

# Callable 类型：可调用对象类型，接受 BaseChatModel 和 list[BaseTool] 作为参数，并返回一个 SubagentRunner 实例。
RunnerFactory = Callable[[BaseChatModel, list[BaseTool]], SubagentRunner]


def _default_runner_factory(
    model: BaseChatModel,
    tools: list[BaseTool],
) -> SubagentRunner:
    """建立无检查点的临时 LangChain agent。"""
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=build_subagent_prompt(),
        checkpointer=None,
    )


class SubagentManager:
    """管理临时子代理的提交、执行、查询、取消和结果回送。"""

    def __init__(
        self,
        model: BaseChatModel,
        tools: Sequence[BaseTool],
        sessions: SessionEventQueue,
        *,
        # 最大并发数：同时运行的子代理数量上限，超过则排队等待。
        max_concurrent: int = 3,
        timeout: float = 300.0,
        runner_factory: RunnerFactory | None = None,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent 必须大于 0")
        if timeout <= 0:
            raise ValueError("timeout 必须大于 0")

        safe_tools = [tool for tool in tools if tool.name not in _FORBIDDEN_TOOL_NAMES]
        self._runner = (runner_factory or _default_runner_factory)(model, safe_tools)
        self._sessions = sessions
        # Semaphore：异步信号量，控制并发数量
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._timeout = timeout
        self._tasks: dict[str, SubagentTask] = {}
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._closed = False

    def submit(self, parent_session_id: str, task: str) -> SubagentTask:
        """提交任务并立即返回任务快照，不等待子代理执行。"""
        if self._closed:
            raise RuntimeError("SubagentManager 已关闭")
        if not parent_session_id:
            raise ValueError("父会话 ID 不能为空")
        if not task.strip():
            raise ValueError("子任务描述不能为空")

        task_id = uuid.uuid4().hex
        record = SubagentTask(
            task_id=task_id,
            parent_session_id=parent_session_id,
            task=task,
            status=SubagentStatus.QUEUED,
            created_at=_utc_now(),
        )
        self._tasks[task_id] = record
        worker = asyncio.create_task(self._run(record), name=f"subagent-{task_id}")
        self._workers[task_id] = worker
        worker.add_done_callback(lambda _worker, key=task_id: self._workers.pop(key, None))
        return record

    def get(self, task_id: str) -> SubagentTask | None:
        """查询单个任务；任务结束后仍可查询。"""
        return self._tasks.get(task_id)

    def list(self) -> list[SubagentTask]:
        """按创建顺序列出当前进程中的全部任务。"""
        return list(self._tasks.values())

    def cancel(self, task_id: str) -> bool:
        """请求取消排队中或运行中的任务。"""
        record = self._tasks.get(task_id)
        worker = self._workers.get(task_id)
        if (
            record is None
            or worker is None
            or worker.done()
            or record.status
            in {
                SubagentStatus.COMPLETED,
                SubagentStatus.FAILED,
                SubagentStatus.CANCELLED,
            }
        ):
            return False
        # 延后一拍取消，确保刚创建、尚未开始执行的协程也能进入统一的收尾与注入逻辑。
        asyncio.get_running_loop().call_soon(worker.cancel)
        return True

    async def close(self) -> None:
        """停止接收新任务，取消并等待全部后台任务完成。"""
        self._closed = True
        # 让刚提交的 worker 至少启动一次，避免取消发生在协程首行之前而跳过收尾。
        await asyncio.sleep(0)
        workers = list(self._workers.values())
        for worker in workers:
            if not worker.done():
                worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)

    async def _run(self, record: SubagentTask) -> None:
        try:
            async with self._semaphore:
                record.status = SubagentStatus.RUNNING
                async with asyncio.timeout(self._timeout):
                    response = await self._runner.ainvoke(
                        {"messages": [HumanMessage(content=record.task)]}
                    )
                record.result = _extract_result(response)
                record.status = SubagentStatus.COMPLETED
        except TimeoutError:
            record.status = SubagentStatus.FAILED
            record.error = f"子任务执行超时（限制 {self._timeout:g} 秒）"
        except asyncio.CancelledError:
            record.status = SubagentStatus.CANCELLED
            record.error = "子任务已取消"
        except Exception as exc:
            record.status = SubagentStatus.FAILED
            record.error = f"{type(exc).__name__}：{exc}"
        finally:
            record.completed_at = _utc_now()
            await self._inject_safely(record)

    async def _inject_safely(self, record: SubagentTask) -> None:
        """回送最终状态；队列故障只记录，不泄漏后台异常。"""
        try:
            await self._sessions.enqueue(
                record.parent_session_id,
                "subagent_result",
                _result_text(record),
                metadata={
                    "task_id": record.task_id,
                    "status": record.status.value,
                    "created_at": record.created_at,
                    "completed_at": record.completed_at,
                },
                event_id=f"subagent:{record.task_id}:result",
            )
        except BaseException as exc:
            record.injection_error = f"{type(exc).__name__}：{exc}"


def _extract_result(response: Any) -> str:
    """从 create_agent 的标准返回状态提取最终文本。"""
    if isinstance(response, Mapping):
        messages = response.get("messages")
        if isinstance(messages, Sequence) and messages:
            content = getattr(messages[-1], "content", messages[-1])
            return _content_to_text(content)
    return _content_to_text(response)


def _content_to_text(content: Any) -> str:
    """把模型可能返回的文本块转换成稳定字符串。"""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts = []
        for item in content:
            if isinstance(item, Mapping) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _result_text(record: SubagentTask) -> str:
    """生成注入父会话的中文结果文本。"""
    if record.status is SubagentStatus.COMPLETED:
        return f"子任务 {record.task_id} 已完成：\n{record.result or '（无文本结果）'}"
    if record.status is SubagentStatus.CANCELLED:
        return f"子任务 {record.task_id} 已取消。"
    return f"子任务 {record.task_id} 执行失败：{record.error or '未知错误'}"
