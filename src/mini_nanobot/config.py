"""
1. 模型的配置参数校验与加载
2. 应用程序配置参数加载
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ConfigurationError(RuntimeError): # RuntimeError运行时错误异常类
    """用户可修复的配置错误。"""
    pass

class ProviderConfig(BaseModel):
    model_config = ConfigDict(validate_default=True) # ConfigDict是校验行为的配置类型

    # 这里创建的是类属性，如果当前直接给默认值，那么它在创建类时就固定了
    # default_factory用于在每次创建对象时动态生成默认值
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    api_base: str = Field(
        default_factory=lambda: os.getenv(
            "OPENAI_API_BASE",
            "https://api.openai.com/v1",
        )
    )
    model: str = Field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o-mini"))
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("MODEL_TEMPERATURE", "0.7")))
    max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MODEL_MAX_TOKENS", "4096")),
        ge=1,)
    timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("MODEL_TIMEOUT_SECONDS", "120")),
        gt=0,)

    @field_validator("api_base") # 把下面的validate_api_base方法注册为api_base字段的验证方法
    @classmethod # 把下面的方法变成类方法，第一个参数必须是cls
    def validate_api_base(cls, value: str) -> str:

        value = value.strip().rstrip("/") # 去除两边空格和最右边的"/"

        if not value:
            raise ValueError("api_base不能为空")

        # 某些云平台为不同工作空间分配独立的API地址，例如：
        # https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
        if any(
            m in value
            for m in ("[workspace-id]", "<workspace-id>", "{workspace_id}")
        ):raise ValueError("仍包含 workspace ID 占位符，请替换为真实值")

        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("OPENAI_API_BASE 必须是包含主机名的 http/https URL")
        return value


class AppConfig(BaseModel):
    provider: ProviderConfig = Field(default_factory=ProviderConfig)

    workspace_dir: Path = Field(
        # resolve():把路径转换为绝对路径
        default_factory=lambda: Path(os.getenv("WORKSPACE_DIR", "./workspace")).resolve())

    context_window: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_WINDOW", "32000")))

    # 一次任务中最多循环执行多少轮
    max_iterations: int = Field(default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "25")))

    # 一个目标完成后，允许智能体继续自动推进目标的最大次数。
    max_goal_continuations: int = Field(
        default_factory=lambda: int(os.getenv("MAX_GOAL_CONTINUATIONS", "12")),
        ge=0, # >=0
    )
    run_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("RUN_TIMEOUT_SECONDS", "600")),
        gt=0,
    )
    # 最多允许同时运行多少个子智能体
    max_concurrent_subagents: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_SUBAGENTS", "3")),
        ge=1, 
    )
    subagent_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("SUBAGENT_TIMEOUT_SECONDS", "300")),
        gt=0, # >0
    )
    # 触发自动“dream”（后台思考或记忆整理）功能所需达到的阈值。
    dream_auto_threshold: int = Field(
        default_factory=lambda: int(os.getenv("DREAM_AUTO_THRESHOLD", "10")),
        ge=0,
    )
    mcp_config_path: str = Field(default_factory=lambda: os.getenv("MCP_CONFIG_PATH", ""))

    @property
    def db_path(self) -> Path:
        return self.workspace_dir / "sessions.db"

    @property
    def memory_dir(self) -> Path:
        return self.workspace_dir / "memory"

    def ensure_dirs(self) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

def load_config() -> AppConfig:
    load_dotenv()
    try:
        cfg = AppConfig()
    except ValidationError as e:
        raise ConfigurationError(f"配置验证失败: {e}") from e

    if not cfg.provider.api_key:
        raise ConfigurationError("OPENAI_API_KEY 未设置")   

    cfg.ensure_dirs()

    return cfg