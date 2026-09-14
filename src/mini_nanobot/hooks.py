"""Hook 生命周期系统。

对应 nanobot 的 `nanobot/agent/hook.py`。让「在关键节点插入自定义逻辑」这件事
不需要改 graph.py 本身——写一个新的 Hook 子类，塞进 hooks 列表就行。

模型、工具、压缩等执行期钩子交给 LangChain middleware；这里只保留产品层的一轮
运行事件，避免维护两套重复生命周期。
"""

from __future__ import annotations

import logging
import time
from typing import Any

_fallback_logger = logging.getLogger("mini_nanobot")


class AgentHook:
    """钩子基类，什么都不做。子类按需覆盖需要的方法。"""

    async def before_run(self, session_id: str) -> None:
        return

    async def after_run(self, session_id: str, final_content: str | None) -> None:
        return

    async def on_error(self, session_id: str, error: Exception) -> None:
        return

    async def on_finally(self, session_id: str) -> None:
        return


class LoggingHook(AgentHook):
    """最基础的实现：把每个生命周期节点打印到日志，方便学习/调试时观察循环过程。"""

    def __init__(self) -> None:
        self._t0: dict[str, float] = {}

    async def before_run(self, session_id: str) -> None:
        self._t0[session_id] = time.monotonic()
        _fallback_logger.info("[hook] session=%s 开始处理一轮对话", session_id)

    async def after_run(self, session_id: str, final_content: str | None) -> None:
        elapsed = time.monotonic() - self._t0.pop(session_id, time.monotonic())
        _fallback_logger.info("[hook] session=%s 结束，耗时 %.2fs", session_id, elapsed)

    async def on_error(self, session_id: str, error: Exception) -> None:
        _fallback_logger.warning("[hook] session=%s 出错：%s", session_id, error)

    async def on_finally(self, session_id: str) -> None:
        _fallback_logger.debug("[hook] session=%s finally 清理", session_id)


class HookRegistry:
    """把多个 hook 合并成一组，逐个调用。任何一个 hook 抛异常都不应该打断主流程。"""

    def __init__(self, hooks: list[AgentHook] | None = None) -> None:
        self.hooks = hooks or []

    async def _call(self, method: str, *args: Any) -> None:
        for hook in self.hooks:
            try:
                await getattr(hook, method)(*args)
            except Exception:  # noqa: BLE001 - hook 本身出错不应该影响主流程
                _fallback_logger.exception("hook %s 执行失败", method)

    async def before_run(self, session_id: str) -> None:
        await self._call("before_run", session_id)

    async def after_run(self, session_id: str, final_content: str | None) -> None:
        await self._call("after_run", session_id, final_content)

    async def on_error(self, session_id: str, error: Exception) -> None:
        await self._call("on_error", session_id, error)

    async def on_finally(self, session_id: str) -> None:
        await self._call("on_finally", session_id)
