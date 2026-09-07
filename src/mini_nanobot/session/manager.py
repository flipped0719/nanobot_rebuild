"""
会话管理器：创建、删除、切换会话，管理会话元数据和运行时状态，以及待处理事件队列。
"""


import asyncio
import json
import os
import tempfile
import threading
import uuid
from collections import deque # 双端队列，是一种数据结构；异步队列是消息对立，二者不是同一种的概念
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


def _utc_now() -> str:
    """返回便于 JSON 保存、带时区的 UTC 时间。"""
    return datetime.now(timezone.utc).isoformat()

# frozen=True：一旦创建了对象，属性就不能再修改；slots=True：限制实例只能有dataclass中定义的属性，节省内存
@dataclass(frozen=True, slots=True)
class SessionInfo:
    """可持久化的会话元数据。"""
    thread_id: str
    title: str
    created_at: str
    updated_at: str

@dataclass(frozen=True, slots=True)
class PendingEvent:
    """待处理事件"""
    event_id: str
    type: str
    content: Any
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class SessionEvent:
    """会话事件描述"""
    thread_id: str
    event: str
    data: dict[str, Any]
    created_at: str

class RunStatus(StrEnum):
    """会话在当前进程中的运行状态。"""
    IDLE = "idle" # 会话当前没有在运行任务，处于“空闲/待机”状态
    RUNNING = "running"
    CANCELLING = "cancelling"

@runtime_checkable # 让一个Protocol 在运行时也能被 isinstance() 检查
class SessionMetadataStore(Protocol):
    """元数据存储协议；Redis 后端只需实现这两个同步短 I/O 方法。"""

    def load(self) -> dict[str, Any]:
        """加载完整元数据快照。"""
        ...

    def save(self, data: Mapping[str, Any]) -> None:
        """原子保存完整元数据快照。"""
        ...


class JsonSessionMetadataStore:
    """使用本地原子 JSON 文件保存会话元数据。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            # version是存储文件/数据结构的版本，后续如果格式变了，可以知道如何兼容读取
            return {"version": 1, "active_thread_id": None, "sessions": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取会话元数据：{self.path}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("sessions"), dict):
            raise ValueError(f"会话元数据格式无效：{self.path}")
        return data

    # Mapping：一个“可通过 key 取值”的对象，像字典那样，这里其实就是一个字典嵌套
    def save(self, data: Mapping[str, Any]) -> None:
        # parents=True：递归创建父目录
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # ensure_ascii=False：不转义非 ASCII 字符，保持原样输出；indent=2：缩进 2 个空格，便于阅读；sort_keys=True：按 key 排序，便于 diff
        payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
        # 创建临时文件，写入数据后再原子替换原文件，避免写入过程中程序崩溃导致文件损坏
        # fd：文件描述符，temp_name：临时文件名
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            # fpopen()：用文件描述符创建一个文件对象，指定编码和换行符
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush() # 刷新缓冲区，确保数据写入磁盘
                os.fsync(handle.fileno()) # 强制将缓冲区数据写入磁盘，确保数据持久化
            # Windows 下必须先关闭临时文件句柄，os.replace 才能稳定工作。
            os.replace(temp_name, self.path)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(temp_name) # 删除临时文件
            except FileNotFoundError:
                pass
            raise


@dataclass(slots=True)
class _SessionRuntime:
    """仅在当前进程存在的会话运行数据。"""
    # 异步锁：在单线程、并发执行的协程之间，建立一个“独享屏障”，防止多个协程同时修改同一份共享数据，从而保证数据的一致性和正确性。
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending: deque[PendingEvent] = field(default_factory=deque)
    events_by_id: dict[str, PendingEvent] = field(default_factory=dict)
    status: RunStatus = RunStatus.IDLE
    task: asyncio.Task[Any] | None = None


class SessionManager:
    """会话管理器。"""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        store: SessionMetadataStore | None = None,
    ) -> None:
        if store is None:
            if data_dir is None:
                raise ValueError("data_dir 和 store 至少需要提供一个")
            store = JsonSessionMetadataStore(Path(data_dir) / "sessions.json")
        elif data_dir is not None:
            raise ValueError("data_dir 和 store 不能同时提供")

        self._store = store
        # 可重入锁：允许同一线程多次获取锁（例如，函数嵌套中外层和内层都有锁），避免死锁；在多线程环境下保护共享数据的访问，确保线程安全
        self._guard = threading.RLock()
        self._data = self._normalise(store.load())
        self._runtime: dict[str, _SessionRuntime] = {
            thread_id: _SessionRuntime() for thread_id in self._data["sessions"]
        }
        # 字典的遍历实际是遍历字典的key，因此thread_id是存入store的thread_id
    @staticmethod
    def _normalise(raw: Mapping[str, Any]) -> dict[str, Any]:
        """规范化原始元数据。"""
        sessions: dict[str, dict[str, str]] = {}
        for thread_id, value in raw.get("sessions", {}).items():
            if not isinstance(thread_id, str) or not isinstance(value, Mapping):
                continue
            try:
                info = SessionInfo(
                    thread_id=thread_id,
                    title=str(value["title"]),
                    created_at=str(value["created_at"]),
                    updated_at=str(value["updated_at"]),
                )
            except KeyError:
                continue
            # asdict()：将 SessionInfo 实例转换为字典，方便存储和序列化
            sessions[thread_id] = asdict(info)

            # 当前选中的会话，只能有一个对话处于activate状态
            active = raw.get("active_thread_id")
            if active not in sessions:
                active = None
            return {"version": 1, "active_thread_id": active, "sessions": sessions}


    def _save_locked(self) -> None:
        # 原子写
        self._store.save(self._data)

    # 获取会话运行时数据，如果不存在则抛出 KeyError
    def _require_runtime(self, thread_id: str) -> _SessionRuntime:
        with self._guard:
            if thread_id not in self._data["sessions"]:
                raise KeyError(f"会话不存在：{thread_id}")
            return self._runtime[thread_id]

    # 获取会话元数据，如果不存在则抛出 KeyError
    def _info_locked(self, thread_id: str) -> SessionInfo:
        return SessionInfo(**self._data["sessions"][thread_id])

    

    def create(self, title: str = "新会话", *, thread_id: str | None = None) -> SessionInfo:
        """创建新会话。"""
        if thread_id is None:
            thread_id = uuid.uuid4().hex

        now = _utc_now()
        with self._guard:
            if thread_id in self._data["sessions"]:
                raise ValueError(f"thread_id 已存在：{thread_id}")
            info = SessionInfo(thread_id, title.strip() or "新会话", now, now)
            # 新增session元数据
            self._data["sessions"][thread_id] = asdict(info)
            self._runtime[thread_id] = _SessionRuntime()
            previous_active = self._data["active_thread_id"]
            # 如果当前没有激活的会话，则将新创建的会话设为激活状态
            if self._data["active_thread_id"] is None:
                self._data["active_thread_id"] = thread_id
            try:
                # 将self._data原子写为json文件
                self._save_locked()
            except BaseException:
                del self._data["sessions"][thread_id]
                del self._runtime[thread_id]
                self._data["active_thread_id"] = previous_active
                raise
            return info

    def list(self) -> list[SessionInfo]:
        """按最近更新时间倒序列出会话。"""
        with self._guard:
            values = [SessionInfo(**value) for value in self._data["sessions"].values()]
        # 先按 updated_at 排序，再按 thread_id 排序
        return sorted(values, key=lambda item: (item.updated_at, item.thread_id), reverse=True)

    # _info_locked是内部快速取值，默认认为数据一定存在;get是外部接口，可能取不到数据，所以返回值是可选的
    def get(self, thread_id: str) -> SessionInfo | None:
        # 获取指定 thread_id 的会话元数据，如果不存在则返回 None
        with self._guard:
            value = self._data["sessions"].get(thread_id)
            return SessionInfo(**value) if value is not None else None

    @property
    def active_thread_id(self) -> str | None:
        with self._guard:
            return self._data["active_thread_id"]

    def get_active(self) -> SessionInfo | None:
        with self._guard:
            thread_id = self._data["active_thread_id"]
            return self._info_locked(thread_id) if thread_id is not None else None

    def activate(self, thread_id: str) -> SessionInfo:
        """切换当前会话，并原子保存 active_thread_id。"""
        with self._guard:
            if thread_id not in self._data["sessions"]:
                raise KeyError(f"会话不存在：{thread_id}")
            previous = self._data["active_thread_id"]
            self._data["active_thread_id"] = thread_id
            try:
                self._save_locked()
            except BaseException:
                self._data["active_thread_id"] = previous
                raise
            return self._info_locked(thread_id)

    def delete(self, thread_id: str) -> bool:
        """删除会话；正在运行的任务必须先取消或结束。"""
        with self._guard:
            if thread_id not in self._data["sessions"]:
                return False
            runtime = self._runtime[thread_id]
            # 在cancel_run调用时，状态会被设置为CANCELLING，但该函数调用时后面会接着finish_run，最终状态会变为IDLE，因此一般不会出现状态是CANCELLING的情况
            if runtime.status is not RunStatus.IDLE:
                raise RuntimeError("不能删除正在运行的会话")

            old_info = self._data["sessions"].pop(thread_id)
            old_active = self._data["active_thread_id"]
            del self._runtime[thread_id]
            if old_active == thread_id:
                remaining = self.list()
                self._data["active_thread_id"] = remaining[0].thread_id if remaining else None
            try:
                self._save_locked()
            except BaseException:
                self._data["sessions"][thread_id] = old_info
                self._data["active_thread_id"] = old_active
                self._runtime[thread_id] = runtime
                raise
            return True

    # 获取会话专属锁，供未来执行图串行化同一会话请求，避免多个协程同时修改同一会话的运行状态或待处理事件，从而保证数据的一致性和正确性。
    def lock_for(self, thread_id: str) -> asyncio.Lock:
        """取得会话专属锁，供未来执行图串行化同一会话请求。"""
        return self._require_runtime(thread_id).lock
    

    # 待处理事件管理
    async def enqueue(
        self,
        thread_id: str,
        event_type: str,
        content: Any,
        metadata: Mapping[str, Any] | None = None,
        *,
        event_id: str | None = None,
    ) -> PendingEvent:
        """加入待处理事件；同一 event_id 重复入队时保持幂等。幂等：重复执行同一个操作，结果不会产生额外副作用。"""
        runtime = self._require_runtime(thread_id)
        if event_id is None:
            event_id = uuid.uuid4().hex
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id 必须是非空字符串")
        async with runtime.pending_lock:
            # 避免重复事件
            if event_id in runtime.events_by_id:
                return runtime.events_by_id[event_id]
            event = PendingEvent(event_id, event_type, content, dict(metadata or {}))
            # 加入队列
            runtime.pending.append(event)
            runtime.events_by_id[event_id] = event
            self._touch(thread_id)
            return event

    async def drain(self, thread_id: str, limit: int | None = None) -> list[PendingEvent]:
        """按 FIFO 顺序取走待处理事件。"""
        if limit is not None and limit < 0:
            raise ValueError("limit 不能小于 0")
        runtime = self._require_runtime(thread_id)
        async with runtime.pending_lock:
            count = len(runtime.pending) if limit is None else min(limit, len(runtime.pending))
            return [runtime.pending.popleft() for _ in range(count)]

    async def drain_pending(
        self,
        session_id: str,
        *,
        limit: int = 3,
    ) -> list[PendingEvent]:
        """供 Agent middleware 使用的协议别名。"""
        return await self.drain(session_id, limit=limit)

    def pending_count(self, thread_id: str) -> int:
        """返回队列长度；精确修改仍由会话异步锁保护。"""
        return len(self._require_runtime(thread_id).pending)

    # 触碰会话，更新 updated_at 时间戳
    def _touch(self, thread_id: str) -> None:
        with self._guard:
            old = self._data["sessions"][thread_id]["updated_at"]
            self._data["sessions"][thread_id]["updated_at"] = _utc_now()
            try:
                self._save_locked()
            except BaseException:
                self._data["sessions"][thread_id]["updated_at"] = old
                raise

    def start_run(
        self,
        thread_id: str,
        task: asyncio.Task[Any] | None = None,
    ) -> None:
        """标记运行中，并可登记用于取消的 asyncio.Task。"""
        runtime = self._require_runtime(thread_id)
        if runtime.status is not RunStatus.IDLE:
            raise RuntimeError("会话已经在运行")
        if task is not None and task.done():
            raise ValueError("不能登记已结束的任务")
        runtime.status = RunStatus.RUNNING
        runtime.task = task

    def finish_run(self, thread_id: str) -> None:
        runtime = self._require_runtime(thread_id)
        runtime.status = RunStatus.IDLE
        runtime.task = None

    def run_status(self, thread_id: str) -> RunStatus:
        return self._require_runtime(thread_id).status

    def run_task(self, thread_id: str) -> asyncio.Task[Any] | None:
        return self._require_runtime(thread_id).task

    def cancel_run(self, thread_id: str) -> bool:
        """请求取消任务；调用方应在任务收尾后调用 finish_run。"""
        runtime = self._require_runtime(thread_id)
        task = runtime.task
        if runtime.status is RunStatus.IDLE or task is None or task.done():
            return False
        runtime.status = RunStatus.CANCELLING
        # 异步任务里，task.cancel() 不是“立刻终止”；它只是发出取消信号。
        task.cancel()
        return True