from .consolidator import Consolidator, estimate_tokens
from .store import FileMemoryBackend, MemoryBackend, MemorySnapshot, MemoryStore


def __getattr__(name: str):
    """延迟加载 Dream，保持公共导入接口不变。"""
    if name == "run_dream":
        from .dream import run_dream

        return run_dream
    raise AttributeError(name)

__all__ = [
    "Consolidator",
    "FileMemoryBackend",
    "MemoryBackend",
    "MemorySnapshot",
    "MemoryStore",
    "estimate_tokens",
    "run_dream",
]
