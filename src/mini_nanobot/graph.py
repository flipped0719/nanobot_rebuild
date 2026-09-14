"""核心运行时：外层业务编排 + 内层 LangChain `create_agent`。

内层 agent 负责标准 ReAct 循环，重试、压缩、调用预算、动态提示词等横切能力由
middleware 组合；外层 StateGraph 只保留 mini-nanobot 特有的 sustained-goal
续跑。这样不会重复实现 LangChain 已经提供的模型/工具循环，同时仍能清楚看到
nanobot 的业务状态机。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from .config import AppConfig
from .hooks import HookRegistry, LoggingHook
from .memory import Consolidator, FileMemoryBackend, MemoryBackend, MemoryStore
from .middleware import build_agent_middleware
from .session import SessionManager
from .state import AgentContext, AgentState
from .subagents import SubagentManager
from .tools import GOAL_TOOLS, load_mcp_tools, make_basic_tools, make_spawn_tool


def _build_llm(cfg: AppConfig) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=cfg.provider.api_key,
        base_url=cfg.provider.api_base,
        model=cfg.provider.model,
        temperature=cfg.provider.temperature,
        max_tokens=cfg.provider.max_tokens,
        timeout=cfg.provider.timeout_seconds,
    )


@dataclass
class AppRuntime:
    """AgentService 需要的一切：编译好的图 + 旁路能力（压缩、记忆、hooks）。"""

    graph: CompiledStateGraph
    llm: BaseChatModel
    memory: MemoryBackend
    sessions: SessionManager
    subagents: SubagentManager
    consolidator: Consolidator
    hooks: HookRegistry
    max_iterations: int
    run_timeout_seconds: float
    dream_auto_threshold: int


def _build_graph(
    llm: BaseChatModel,
    all_tools: list[BaseTool],
    cfg: AppConfig,
    max_iterations: int,
    checkpointer,
) -> CompiledStateGraph:
    inner_agent = create_agent(
        llm,
        tools=all_tools,
        middleware=build_agent_middleware(
            llm,
            context_window=cfg.context_window,
            consolidation_ratio=cfg.consolidation_ratio,
            max_model_calls=max_iterations,
        ),
        state_schema=AgentState,
        context_schema=AgentContext,
        name="react_agent",
    )

    async def prepare_run(
        state: AgentState,
        runtime: Runtime[AgentContext],
    ) -> dict:
        """重置单次预算，并把服务层的显式授权复制到可注入工具状态。"""
        return {
            "continuation_count": 0,
            "goal_creation_allowed": runtime.context.goal_creation_allowed,
        }

    async def count_goal_continuation(state: AgentState) -> dict:
        return {"continuation_count": state.get("continuation_count", 0) + 1}

    async def mark_goal_budget_exhausted(state: AgentState) -> dict:
        goal = dict(state.get("goal_state") or {})
        goal.update(
            {
                "status": "blocked",
                "detail": f"本轮已达到 {cfg.max_goal_continuations} 次自动续跑上限",
            }
        )
        return {"goal_state": goal}

    def route_after_agent(state: AgentState) -> str:
        goal = state.get("goal_state") or {}
        count = state.get("continuation_count", 0)
        if goal.get("status") == "active":
            if count < cfg.max_goal_continuations:
                return "goal_continue"
            return "goal_exhausted"
        return END

    graph = StateGraph(AgentState, context_schema=AgentContext)
    graph.add_node("prepare_run", prepare_run)
    graph.add_node("agent", inner_agent)
    graph.add_node("goal_continue", count_goal_continuation)
    graph.add_node("goal_exhausted", mark_goal_budget_exhausted)

    graph.add_edge(START, "prepare_run")
    graph.add_edge("prepare_run", "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"goal_continue": "goal_continue", "goal_exhausted": "goal_exhausted", END: END},
    )
    graph.add_edge("goal_continue", "agent")
    graph.add_edge("goal_exhausted", END)

    return graph.compile(checkpointer=checkpointer)


@asynccontextmanager
async def create_app(cfg: AppConfig) -> AsyncIterator[AppRuntime]:
    """组装整个运行时：provider、工具（基础 + goal + MCP + spawn）、记忆、图。

    是一个 async context manager，因为 `AsyncSqliteSaver` 需要在 `async with`
    块里维持数据库连接——退出这个函数的作用域，连接就会被正确关闭。
    """
    llm = _build_llm(cfg)
    memory = FileMemoryBackend(memory_dir=cfg.memory_dir)
    await memory.initialize()
    sync_memory = MemoryStore(memory_dir=cfg.memory_dir)
    sessions = SessionManager(cfg.workspace_dir / "sessions")
    consolidator = Consolidator(
        llm=llm,
        memory=sync_memory,
        context_window=cfg.context_window,
        consolidation_ratio=cfg.consolidation_ratio,
    )
    hooks = HookRegistry([LoggingHook()])

    mcp_tools = await load_mcp_tools(cfg.mcp_config_path)
    basic_tools = make_basic_tools(cfg.workspace_dir)
    subagent_tools = [*basic_tools, *mcp_tools]
    subagents = SubagentManager(
        llm,
        subagent_tools,
        sessions,
        max_concurrent=cfg.max_concurrent_subagents,
        timeout=cfg.subagent_timeout_seconds,
    )
    spawn_tool = make_spawn_tool(subagents)
    all_tools = [*basic_tools, *GOAL_TOOLS, *mcp_tools, spawn_tool]

    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(cfg.db_path)) as checkpointer:
        graph = _build_graph(llm, all_tools, cfg, cfg.max_iterations, checkpointer)
        try:
            yield AppRuntime(
                graph=graph,
                llm=llm,
                memory=memory,
                sessions=sessions,
                subagents=subagents,
                consolidator=consolidator,
                hooks=hooks,
                max_iterations=cfg.max_iterations,
                run_timeout_seconds=cfg.run_timeout_seconds,
                dream_auto_threshold=cfg.dream_auto_threshold,
            )
        finally:
            await subagents.close()
