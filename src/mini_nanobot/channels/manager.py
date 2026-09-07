"""
消息通道管理器，实现入站与出站消息的分发
"""

from __future__ import annotations

import asyncio
from ..bus import MessageBus
from .base import BaseChannel


class ChannelManager:
    def __init__(self, bus: MessageBus, channels: list[BaseChannel]) -> None:
        self.bus = bus
        # 字典：{BaseChannel.name:BaseChannel}
        self.channels = {c.name: c for c in channels}
        # 元素为asyncio.Task的列表，Task[None]表示每个任务执行完后返回None
        self._tasks: list[asyncio.Task[None]] = []
        self._channel_tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        for channel in self.channels.values():
            # 创建不同channel消息监听任务
            task = asyncio.create_task(
                channel.start(),
                name=f"channel:{channel.name}",
            )
            self._channel_tasks.append(task)
            self._tasks.append(task)
        self._tasks.append(asyncio.create_task(self._dispatch_outbound()))
	
    async def stop(self) -> None:
        # 一次停止多个channel的监听
        for channel in self.channels.values():
            await channel.stop()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            # 并发运行多个协程：回复任务
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _dispatch_outbound(self) -> None:
        while True:
            # 获取回复消息
            msg = await self.bus.consume_outbound()
            # BaseChannel的某个子类
            channel = self.channels.get(msg.channel)
            if channel is None:
                continue
            if msg.event == "delta":
                await channel.send_delta(msg.session_id, msg.content, msg.metadata)
            elif msg.event == "stream_end":
                await channel.send_delta_end(msg.session_id, msg.metadata)
            else:
                await channel.send(msg.session_id, msg.content, msg.metadata)

    async def wait_until_all_stopped(self) -> None:
        """阻塞直到所有channel都停止运行，确认所有channel都已关闭"""
        if self._channel_tasks:
            await asyncio.gather(*self._channel_tasks)