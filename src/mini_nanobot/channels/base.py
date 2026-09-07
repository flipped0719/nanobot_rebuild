"""
定义所有消息通道的基类，子类需要实现 start、stop、send 方法。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from ..bus import InboundMessage, MessageBus

class BaseChannel(ABC):
    name: str = "base"
    supports_streaming: bool = False

    def __init__(self, bus: MessageBus) -> None:
        self.bus = bus
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        """开始监听平台消息（长期运行的协程）。"""

    @abstractmethod
    async def stop(self) -> None:
        """停止监听、清理资源。"""

    @abstractmethod
    async def send(self, session_id: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """发送一条完整的回复。"""

    async def send_delta(
        self,
        session_id: str,
        delta: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        return

    async def send_delta_end(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        return

    async def _handle_message(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        meta = dict(metadata or {})
        meta.setdefault("supports_stream", self.supports_streaming)
        await self.bus.publish_inbound(
            InboundMessage(
                channel=self.name,
                session_id=session_id,
                content=content,
                metadata=meta,
            )
        )

    @property
    def is_running(self) -> bool:
        return self._running
    # 统一查看不同channel的_running的接口，同时限制对示例属性的访问
