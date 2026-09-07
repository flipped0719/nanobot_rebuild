"""
个人微信消息渠道
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import uuid
from typing import Any

import httpx

from ..bus import MessageBus
from .base import BaseChannel

logger = logging.getLogger(__name__)

BASE_URL = "https://ilinkai.weixin.qq.com"
BASE_INFO = {"channel_version": "2.1.1"}
BOT_MESSAGE = 2
TEXT_ITEM = 1
MAX_MESSAGE_LENGTH = 4000


class WeixinChannel(BaseChannel):
    name = "weixin"
    supports_streaming = False

    def __init__(
        self,
        bus: MessageBus,
        token: str | None = None,
        *,
        base_url: str = BASE_URL,
        poll_timeout: int = 35,
    ) -> None:
        super().__init__(bus)
        self.token = (token or os.getenv("WEIXIN_BOT_TOKEN", "")).strip()
        self.base_url = base_url.rstrip("/")
        self.poll_timeout = poll_timeout
        self._client: httpx.AsyncClient | None = None
        self._updates_cursor = ""
        self._context_tokens: dict[str, str] = {}

    async def start(self) -> None:
        if not self.token:
            raise RuntimeError("未配置微信 token，请设置 WEIXIN_BOT_TOKEN")

        self._running = True
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.poll_timeout + 10, connect=30),
            follow_redirects=True,
        )

        try:
            while self._running:
                try:
                    await self._poll_once()
                except httpx.TimeoutException:
                    # 长轮询超时表示暂时没有新消息，继续请求即可。
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception:
                    if not self._running:
                        break
                    logger.exception("微信消息轮询失败")
                    await asyncio.sleep(2)
        finally:
            await self._close_client()
            self._running = False

    async def stop(self) -> None:
        self._running = False
        await self._close_client()

    async def send(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not content:
            return
        context_token = self._context_tokens.get(session_id, "")
        if metadata:
            context_token = str(metadata.get("context_token") or context_token)
        if not context_token:
            raise RuntimeError(f"找不到用户 {session_id} 的 context_token，无法回复")

        for start in range(0, len(content), MAX_MESSAGE_LENGTH):
            await self._send_text(
                session_id,
                content[start : start + MAX_MESSAGE_LENGTH],
                context_token,
            )

    async def _poll_once(self) -> None:
        data = await self._post(
            "ilink/bot/getupdates",
            {"get_updates_buf": self._updates_cursor},
        )

        self._raise_for_api_error(data, "获取微信消息失败")
        if data.get("get_updates_buf"):
            self._updates_cursor = str(data["get_updates_buf"])

        for message in data.get("msgs") or []:
            await self._receive(message)

    async def _receive(self, message: dict[str, Any]) -> None:
        """提取一条微信消息中的文本并投递给上层服务。"""
        if message.get("message_type") == BOT_MESSAGE:
            return

        user_id = str(message.get("from_user_id") or "")
        if not user_id:
            return

        context_token = str(message.get("context_token") or "")
        if context_token:
            self._context_tokens[user_id] = context_token

        text_parts: list[str] = []
        for item in message.get("item_list") or []:
            if item.get("type") != TEXT_ITEM:
                continue
            text = str((item.get("text_item") or {}).get("text") or "")
            if text:
                text_parts.append(text)

        content = "\n".join(text_parts).strip()
        if not content:
            return

        await self._handle_message(
            session_id=user_id,
            content=content,
            metadata={
                "message_id": str(
                    message.get("message_id") or message.get("seq") or ""
                ),
                "context_token": context_token,
            },
        )

    async def _send_text(
        self,
        user_id: str,
        content: str,
        context_token: str,
    ) -> None:
        message = {
            "from_user_id": "",
            "to_user_id": user_id,
            "client_id": f"mini-nanobot-{uuid.uuid4().hex[:12]}",
            "message_type": BOT_MESSAGE,
            "message_state": 2,
            "context_token": context_token,
            "item_list": [
                {"type": TEXT_ITEM, "text_item": {"text": content}},
            ],
        }
        data = await self._post("ilink/bot/sendmessage", {"msg": message})
        self._raise_for_api_error(data, "发送微信消息失败")

    async def _post(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("微信渠道尚未启动")

        payload = dict(body)
        payload.setdefault("base_info", BASE_INFO)
        response = await self._client.post(
            f"{self.base_url}/{endpoint}",
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise TypeError("微信接口返回了无效数据")
        return result

    def _headers(self) -> dict[str, str]:
        random_uin = int.from_bytes(os.urandom(4), "big")
        return {
            "Authorization": f"Bearer {self.token}",
            "AuthorizationType": "ilink_bot_token",
            "Content-Type": "application/json",
            "iLink-App-Id": "bot",
            "iLink-App-ClientVersion": str(0x020101),
            "X-WECHAT-UIN": base64.b64encode(str(random_uin).encode()).decode(),
        }

    @staticmethod
    def _raise_for_api_error(data: dict[str, Any], message: str) -> None:
        ret = data.get("ret", 0)
        errcode = data.get("errcode", 0)
        if ret not in (None, 0, "0") or errcode not in (None, 0, "0"):
            detail = data.get("errmsg", "")
            raise RuntimeError(
                f"{message}: ret={ret}, errcode={errcode}, errmsg={detail}"
            )

    async def _close_client(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()
