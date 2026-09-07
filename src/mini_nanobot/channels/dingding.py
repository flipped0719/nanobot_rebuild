"""钉钉机器人渠道"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from contextlib import suppress
from typing import Any

import httpx

from ..bus import MessageBus
from .base import BaseChannel

try:
    import dingtalk_stream
except ImportError:  # 允许未安装可选依赖时继续导入其他渠道。
    dingtalk_stream = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

TOKEN_URL = "https://api.dingtalk.com/v1.0/oauth2/accessToken"
SEND_URL = "https://api.dingtalk.com/v1.0/robot/oToMessages/batchSend"


class DingdingChannel(BaseChannel):
    name = "dingding"
    supports_streaming = False

    def __init__(
        self,
        bus: MessageBus,
        client_id: str | None = None,
        client_secret: str | None = None,
        robot_code: str | None = None,
    ) -> None:
        super().__init__(bus)
        self.client_id = self._setting(client_id, "DINGTALK_CLIENT_ID", "CLIENT_ID")
        self.client_secret = self._setting(
            client_secret,
            "DINGTALK_CLIENT_SECRET",
            "CLIENT_SECRET",
        )
        self.robot_code = self._setting(
            robot_code,
            "DINGTALK_ROBOT_CODE",
            "ROBOT_CODE",
        )

        self._http: httpx.AsyncClient | None = None
        self._stream_client: Any = None
        self._start_task: asyncio.Task[None] | None = None
        self._access_token = ""
        self._token_expires_at = 0.0
        self._token_lock = asyncio.Lock()

    async def start(self) -> None:
        """连接钉钉 Stream，并把收到的文本消息放入 MessageBus。"""
        self._validate_config()
        if dingtalk_stream is None:
            raise RuntimeError("钉钉渠道需要安装 dingtalk-stream")

        current_task = asyncio.current_task()
        self._start_task = current_task
        self._running = True
        self._http = httpx.AsyncClient(timeout=30, follow_redirects=True)

        credential = dingtalk_stream.Credential(
            self.client_id,
            self.client_secret,
        )
        self._stream_client = dingtalk_stream.DingTalkStreamClient(credential)
        self._stream_client.register_callback_handler(
            dingtalk_stream.ChatbotMessage.TOPIC,
            self._create_message_handler(),
        )

        try:
            while self._running:
                try:
                    await self._stream_client.start()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    if not self._running:
                        break
                    logger.exception("钉钉 Stream 连接异常")
                if self._running:
                    await asyncio.sleep(3)
        finally:
            self._running = False
            if self._start_task is current_task:
                self._start_task = None
            await self._close_http()

    async def stop(self) -> None:
        self._running = False
        await self._close_stream()

        # 部分版本的 Stream SDK 会吞掉第一次 CancelledError，取消两次。
        task = self._start_task
        if task and task is not asyncio.current_task() and not task.done():
            task.cancel()
            await asyncio.sleep(0)
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        self._stream_client = None
        await self._close_http()

    async def send(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """向 ``session_id`` 对应的钉钉员工发送单聊文本。"""
        if not content:
            return
        if self._http is None:
            raise RuntimeError("钉钉渠道尚未启动")

        token = await self._get_access_token()
        response = await self._http.post(
            SEND_URL,
            headers={"x-acs-dingtalk-access-token": token},
            json={
                "robotCode": self.robot_code,
                "userIds": [session_id],
                "msgKey": "sampleText",
                "msgParam": json.dumps(
                    {"content": content},
                    ensure_ascii=False,
                ),
            },
        )
        response.raise_for_status()
        self._raise_for_api_error(response, "发送钉钉消息失败")

    async def _receive_callback(self, data: dict[str, Any]) -> None:
        """解析 Stream 回调中的员工 ID 和文本。"""
        user_id = str(data.get("senderStaffId") or "")
        if not user_id:
            raise ValueError("钉钉消息缺少 senderStaffId")
        if data.get("msgtype") != "text":
            raise ValueError("当前仅支持钉钉文本消息")

        content = str((data.get("text") or {}).get("content") or "").strip()
        if not content:
            raise ValueError("钉钉消息内容为空")

        # 与 ConsoleChannel 一致：统一通过 BaseChannel 写入 inbound bus。
        await self._handle_message(
            session_id=user_id,
            content=content,
            metadata={
                "message_id": str(data.get("msgId") or ""),
                "corp_id": str(data.get("senderCorpId") or ""),
                "conversation_id": str(data.get("conversationId") or ""),
                "sender_name": str(data.get("senderNick") or ""),
            },
        )

    def _create_message_handler(self) -> Any:
        channel = self

        class MessageHandler(dingtalk_stream.ChatbotHandler):  # type: ignore[union-attr]
            async def process(self, callback: Any) -> tuple[int, str]:
                try:
                    await channel._receive_callback(callback.data)
                except Exception:
                    logger.exception("处理钉钉消息失败")
                    return dingtalk_stream.AckMessage.STATUS_SYSTEM_EXCEPTION, "failed"
                return dingtalk_stream.AckMessage.STATUS_OK, "OK"

        return MessageHandler()

    async def _get_access_token(self) -> str:
        now = time.monotonic()
        if self._access_token and now < self._token_expires_at:
            return self._access_token

        async with self._token_lock:
            now = time.monotonic()
            if self._access_token and now < self._token_expires_at:
                return self._access_token
            if self._http is None:
                raise RuntimeError("钉钉渠道尚未启动")

            response = await self._http.post(
                TOKEN_URL,
                json={
                    "appKey": self.client_id,
                    "appSecret": self.client_secret,
                },
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise TypeError("钉钉 Token 接口返回了无效数据")

            token = str(result.get("accessToken") or "")
            if not token:
                detail = result.get("message") or result
                raise RuntimeError(f"获取钉钉 AccessToken 失败: {detail}")

            expires_in = int(result.get("expireIn", 7200))
            self._access_token = token
            self._token_expires_at = now + max(expires_in - 200, 60)
            return token

    @staticmethod
    def _raise_for_api_error(response: httpx.Response, message: str) -> None:
        if not response.content:
            return
        try:
            result = response.json()
        except ValueError:
            return
        if not isinstance(result, dict):
            return

        code = result.get("code", result.get("errcode"))
        if code not in (None, "", 0, "0"):
            detail = result.get("message") or result.get("errmsg") or result
            raise RuntimeError(f"{message}: code={code}, message={detail}")

    async def _close_stream(self) -> None:
        client = self._stream_client
        if client is None:
            return
        close = getattr(client, "close", None)
        if close is None:
            close = getattr(getattr(client, "websocket", None), "close", None)
        if close is None:
            return
        with suppress(Exception):
            result = close()
            if inspect.isawaitable(result):
                await result

    async def _close_http(self) -> None:
        client, self._http = self._http, None
        if client is not None:
            await client.aclose()

    def _validate_config(self) -> None:
        missing = [
            name
            for name, value in (
                ("client_id", self.client_id),
                ("client_secret", self.client_secret),
                ("robot_code", self.robot_code),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(f"钉钉渠道缺少配置: {', '.join(missing)}")

    @staticmethod
    def _setting(value: str | None, *environment_names: str) -> str:
        if value:
            return value.strip()
        for name in environment_names:
            configured = os.getenv(name, "").strip()
            if configured:
                return configured
        return ""


# 同时提供官方英文拼写，方便调用方选择。
DingTalkChannel = DingdingChannel
