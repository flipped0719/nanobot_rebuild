"""MCP（Model Context Protocol）工具加载。

对应 nanobot 的 `nanobot/agent/tools/mcp.py` + build-guide 第 17 步。

用官方的 `langchain-mcp-adapters` 库做协议层的活（stdio/http 连接、JSON-RPC、
把 MCP 工具转换成 LangChain 工具），我们自己只负责：从一个 JSON 配置文件读出
「要连哪些 MCP server」，没配置就跳过（可选功能，不影响核心功能可用性）。

配置文件格式（`mcp_servers.example.json` 有一个例子）：

```json
{
  "filesystem": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "./workspace"],
    "transport": "stdio"
  }
}
```
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.tools import BaseTool

logger = logging.getLogger("mini_nanobot")


async def load_mcp_tools(mcp_config_path: str) -> list[BaseTool]:
    """按配置连接 MCP server 并返回它们的工具列表；未配置则返回空列表。"""
    if not mcp_config_path:
        return []

    path = Path(mcp_config_path)
    if not path.exists():
        logger.warning("MCP_CONFIG_PATH=%s 指向的文件不存在，跳过 MCP 加载。", mcp_config_path)
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        logger.warning("未安装 langchain-mcp-adapters，跳过 MCP 加载。运行 `uv sync` 后重试。")
        return []

    servers = json.loads(path.read_text(encoding="utf-8"))
    if not servers:
        return []

    client = MultiServerMCPClient(servers)
    try:
        tools = await client.get_tools()
    except Exception:
        logger.exception("连接 MCP server 失败，本次启动不加载 MCP 工具。")
        return []

    logger.info("已从 %s 个 MCP server 加载 %s 个工具。", len(servers), len(tools))
    return tools
