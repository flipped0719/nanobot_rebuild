"""AgentService：消费总线上的消息，跑图，把结果发回总线。

对应 nanobot 的 `AgentLoop`（`nanobot/agent/loop.py`）—— 简化版：完整的 8 状态
状态机（RESTORE → COMPACT → COMMAND → BUILD → RUN → SAVE → RESPOND → DONE）
里，RESTORE/SAVE 被 LangGraph 的 checkpointer 隐式接管了，COMPACT 变成了图里
的 `maybe_compact` 节点，BUILD/RUN 就是图本身；这里只保留一层显式判断：
「这是斜杠命令还是普通对话」，对应原版的 COMMAND 状态。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from .bus import InboundMessage, MessageBus, OutboundMessage
from .session import RunStatus
from .state import AgentContext

if TYPE_CHECKING:
    from .graph import AppRuntime

logger = logging.getLogger("mini_nanobot")


class AgentService:
    def __init__(self, bus: MessageBus, runtime: AppRuntime) -> None:
        self.bus = bus
        self.runtime = runtime
        self._tasks: set[asyncio.Task[None]] = set()
        self._dream_lock = asyncio.Lock()

    async def run(self) -> None:
        """长期运行的主循环：不断从 inbound 取消息处理。"""
        try:
            while True:
                msg = await self.bus.consume_inbound()
                try:
                    await self._dispatch(msg)
                except Exception as exc:  # noqa: BLE001 - 单条消息出错不能打断整个服务
                    logger.exception("处理消息失败")
                    await self._reply(msg, f"处理失败：{exc}")
        finally:
            for task in self._tasks:
                task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _dispatch(self, msg: InboundMessage) -> None:
        """快速分派消息，不在总线消费循环里等待整次模型运行。"""
        if self.runtime.sessions.get(msg.session_id) is None:
            self.runtime.sessions.create("外部会话", thread_id=msg.session_id)

        content = msg.content.strip()
        if content.startswith("/") and await self._maybe_handle_command(msg, content):
            return
        goal_objective = self._goal_objective(content)
        if goal_objective is not None:
            if not goal_objective:
                await self._reply(msg, "用法：/goal <目标>")
                return
            if self.runtime.sessions.run_status(msg.session_id) is not RunStatus.IDLE:
                await self._reply(msg, "当前会话正在运行，请先用 /stop 停止后再创建目标。")
                return
            content = f"请创建并持续执行以下目标：\n{goal_objective}"
            self._start_run(msg, content, goal_creation_allowed=True)
            return

        if self.runtime.sessions.run_status(msg.session_id) is not RunStatus.IDLE:
            await self.runtime.sessions.enqueue(
                msg.session_id,
                "user_message",
                content,
                {"channel": msg.channel},
            )
            return
        self._start_run(msg, content, goal_creation_allowed=False)

    def _start_run(
        self,
        msg: InboundMessage,
        content: str,
        *,
        goal_creation_allowed: bool,
    ) -> None:
        """登记并启动一个可取消的会话任务。"""
        task = asyncio.create_task(
            self._run_managed(msg, content, goal_creation_allowed=goal_creation_allowed),
            name=f"mini-nanobot:{msg.session_id}",
        )
        try:
            self.runtime.sessions.start_run(msg.session_id, task)
        except BaseException:
            task.cancel()
            raise
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _start_maintenance(
        self,
        msg: InboundMessage,
        operation: Callable[[], Awaitable[None]],
    ) -> None:
        """后台执行 Dream/压缩，避免阻塞总线对其他会话的消息消费。"""
        task = asyncio.create_task(
            self._run_maintenance(msg, operation),
            name=f"mini-nanobot:maintenance:{msg.session_id}",
        )
        try:
            self.runtime.sessions.start_run(msg.session_id, task)
        except BaseException:
            task.cancel()
            raise
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_maintenance(
        self,
        msg: InboundMessage,
        operation: Callable[[], Awaitable[None]],
    ) -> None:
        try:
            async with self.runtime.sessions.lock_for(msg.session_id):
                await operation()
        except asyncio.CancelledError:
            logger.info("会话维护任务已取消：%s", msg.session_id)
        except Exception as exc:  # noqa: BLE001 - 维护失败不能终止服务
            logger.exception("会话维护任务失败")
            await self._reply(msg, f"维护任务失败：{exc}")
        finally:
            self.runtime.sessions.finish_run(msg.session_id)

    async def _run_managed(
        self,
        msg: InboundMessage,
        content: str,
        *,
        goal_creation_allowed: bool,
    ) -> None:
        """在会话锁内运行图，并保证取消后状态一定复位。"""
        try:
            async with self.runtime.sessions.lock_for(msg.session_id):
                async with asyncio.timeout(self.runtime.run_timeout_seconds):
                    await self._run_once(
                        msg,
                        content,
                        goal_creation_allowed=goal_creation_allowed,
                    )
                await self._maybe_run_auto_dream()
        except asyncio.CancelledError:
            logger.info("会话运行已取消：%s", msg.session_id)
        except TimeoutError:
            await self.runtime.hooks.on_error(
                msg.session_id,
                TimeoutError("本次 Agent 运行超过总超时"),
            )
            await self._reply(
                msg,
                f"本次运行超过 {self.runtime.run_timeout_seconds:g} 秒，已自动停止。",
            )
        except Exception as exc:  # noqa: BLE001 - 子任务异常不能影响主消费循环
            logger.exception("处理消息失败")
            await self._reply(msg, f"处理失败：{exc}")
        finally:
            self.runtime.sessions.finish_run(msg.session_id)

    async def _maybe_run_auto_dream(self) -> None:
        """累计的新摘要达到阈值时串行运行 Dream；失败只记日志，不影响已完成回复。"""
        threshold = self.runtime.dream_auto_threshold
        if threshold <= 0 or self._dream_lock.locked():
            return
        async with self._dream_lock:
            try:
                history = await self.runtime.memory.read_history()
                cursor = await self.runtime.memory.get_dream_cursor()
                if len(history) - cursor < threshold:
                    return
                from .memory.dream import run_dream

                result = await run_dream(self.runtime.llm, self.runtime.memory)
                logger.info("自动 Dream 完成：%s", result)
            except Exception:  # noqa: BLE001 - 自动维护失败不能污染用户回复
                logger.exception("自动 Dream 失败")

    async def _run_once(
        self,
        msg: InboundMessage,
        content: str,
        *,
        goal_creation_allowed: bool,
    ) -> None:
        config = {"configurable": {"thread_id": msg.session_id}}
        context = AgentContext(
            session_id=msg.session_id,
            memory=self.runtime.memory,
            pending=self.runtime.sessions,
            goal_creation_allowed=goal_creation_allowed,
        )
        await self.runtime.hooks.before_run(msg.session_id)
        final_content = ""
        try:
            if msg.metadata.get("supports_stream", False):
                final_content = await self._run_streamed(msg, content, config, context)
            else:
                result = await self.runtime.graph.ainvoke(
                    {"messages": [HumanMessage(content=content)]},
                    config=config,
                    context=context,
                )
                final_content = self._message_text(result["messages"][-1])
                if not final_content:
                    final_content = "模型连续返回空响应，请稍后重试。"
                await self._reply(msg, final_content)
        except Exception as exc:
            await self.runtime.hooks.on_error(msg.session_id, exc)
            raise
        finally:
            await self.runtime.hooks.after_run(msg.session_id, final_content)
            await self.runtime.hooks.on_finally(msg.session_id)

    async def _run_streamed(
        self,
        msg: InboundMessage,
        content: str,
        config: dict,
        context: AgentContext,
    ) -> str:
        """用 `stream_mode="messages"` 拿到模型 token 级别的增量，边生成边发出去。"""
        parts: list[str] = []

        async for event in self.runtime.graph.astream(
            {"messages": [HumanMessage(content=content)]},
            config=config,
            context=context,
            stream_mode="messages",
            subgraphs=True,
        ):
            payload = event
            if (
                isinstance(event, tuple)
                and len(event) == 2
                and isinstance(event[0], tuple)
                and isinstance(event[1], tuple)
            ):
                # subgraphs=True 时 LangGraph 在消息事件外再包一层命名空间。
                payload = event[1]
            if not isinstance(payload, tuple) or len(payload) != 2:
                continue
            chunk, _metadata = payload
            if not isinstance(chunk, (AIMessage, AIMessageChunk)):
                continue
            delta = self._message_text(chunk)
            if not delta:
                continue
            parts.append(delta)
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    session_id=msg.session_id,
                    content=delta,
                    event="delta",
                )
            )

        final_content = "".join(parts)
        if not final_content:
            state = await self.runtime.graph.aget_state(config)
            messages = state.values.get("messages", []) if state else []
            final_content = self._message_text(messages[-1]) if messages else ""
            if not final_content:
                final_content = "模型连续返回空响应，请稍后重试。"
            await self._reply(msg, final_content)

        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel, session_id=msg.session_id, content="", event="stream_end"
            )
        )
        return final_content

    @staticmethod
    def _message_text(message: object) -> str:
        """把字符串或结构化消息内容统一提取为可显示文本。"""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
            )
        return str(content) if content is not None else ""

    @staticmethod
    def _goal_objective(content: str) -> str | None:
        if content == "/goal":
            return ""
        if content.startswith("/goal "):
            return content.removeprefix("/goal ").strip()
        return None

    async def _maybe_handle_command(self, msg: InboundMessage, content: str) -> bool:
        config = {"configurable": {"thread_id": msg.session_id}}

        if content == "/stop":
            task = self.runtime.sessions.run_task(msg.session_id)
            if self.runtime.sessions.cancel_run(msg.session_id):
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)
                state = await self.runtime.graph.aget_state(config)
                goal = dict(state.values.get("goal_state") or {}) if state else {}
                if goal.get("status") == "active":
                    goal.update({"status": "cancelled", "detail": "用户通过 /stop 取消"})
                    await self.runtime.graph.aupdate_state(config, {"goal_state": goal})
                await self._reply(msg, "已请求停止当前会话的运行。")
            else:
                await self._reply(msg, "当前会话没有正在运行的任务。")
            return True

        if content == "/status":
            state = await self.runtime.graph.aget_state(config)
            values = state.values if state else {}
            messages = values.get("messages", [])
            goal = values.get("goal_state") or {}
            goal_desc = f"{goal.get('status', '无')}"
            if goal.get("objective"):
                goal_desc += f"（{goal['objective']}）"
            await self._reply(
                msg,
                f"会话：{msg.session_id}\n"
                f"运行状态：{self.runtime.sessions.run_status(msg.session_id).value}\n"
                f"待处理消息：{self.runtime.sessions.pending_count(msg.session_id)}\n"
                f"消息条数：{len(messages)}\n"
                f"目标状态：{goal_desc}",
            )
            return True

        if content == "/compact":
            if self.runtime.sessions.run_status(msg.session_id) is not RunStatus.IDLE:
                await self._reply(msg, "当前会话正在运行，不能同时压缩上下文。")
                return True
            self._start_maintenance(
                msg,
                lambda: self._compact_session(msg, config),
            )
            return True

        if content == "/dream":
            if self.runtime.sessions.run_status(msg.session_id) is not RunStatus.IDLE:
                await self._reply(msg, "当前会话正在运行，不能同时执行 Dream。")
                return True
            self._start_maintenance(msg, lambda: self._dream_session(msg))
            return True

        return False

    async def _compact_session(self, msg: InboundMessage, config: dict) -> None:
        state = await self.runtime.graph.aget_state(config)
        messages = state.values.get("messages", []) if state else []
        additions = await self.runtime.consolidator.compact(messages, force=True)
        if additions:
            await self.runtime.graph.aupdate_state(config, {"messages": additions})
            await self._reply(msg, "已手动触发一次上下文压缩。")
        else:
            await self._reply(msg, "当前历史还太短，没什么可压缩的。")

    async def _dream_session(self, msg: InboundMessage) -> None:
        from .memory.dream import run_dream

        async with self._dream_lock:
            changelog = await run_dream(self.runtime.llm, self.runtime.memory)
        await self._reply(msg, f"Dream 完成：{changelog}")

    async def _reply(self, msg: InboundMessage, content: str) -> None:
        await self.bus.publish_outbound(
            OutboundMessage(channel=msg.channel, session_id=msg.session_id, content=content)
        )
