"""所有 system prompt 的定义与拼装。

用 `langchain_core.prompts.PromptTemplate` 而不是手写字符串拼接、也不是外部
的 Jinja2/markdown 模板文件。理由：

- `langchain-core` 本来就带这个能力，不用再加 Jinja2 依赖；
- `{变量}` 占位符 + `.format()` 时会做输入校验——漏传一个变量会直接抛
  `KeyError`，比手写 f-string/字符串拼接更容易在开发阶段发现问题；
- 提示词内容还是普通的 Python 代码：走 git blame、走 lint，不需要额外的
  「模板文件加载 + 找路径」逻辑，改一个字都能在 diff 里直接看到。

如果以后提示词变复杂（比如要循环渲染一个动态长度的 skills 列表、或者要支持
条件分支的模板逻辑），再考虑要不要换成 Jinja2（`PromptTemplate` 也支持
`template_format="jinja2"`，到时候不用换库，只是换个参数）；现在的复杂度
（插入长期记忆内容、插入目标状态）用最基础的 f-string 占位符完全够用。
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate

_MAIN_TEMPLATE = PromptTemplate.from_template(
    "你是 mini-nanobot，一个小巧、有帮助的 AI agent。\n"
    "当工具能帮到你时就调用工具，回答尽量简洁，默认使用中文回复。\n"
    "只有当用户明确要求你持续处理、一直做到完成的长期任务时，才调用 create_goal；"
    "普通的一次性问答不需要创建目标。\n"
    "\n"
    "## 长期记忆（Dream 巩固产生，仅作为你的背景参考，不要原文念给用户）\n"
    "### 行为规则（SOUL.md）\n{soul}\n\n"
    "### 用户画像（USER.md）\n{user}\n\n"
    "### 项目知识（MEMORY.md）\n{memory}\n"
    "{goal_section}"
)

_SUBAGENT_TEMPLATE = PromptTemplate.from_template(
    "你是被主 agent 派生出来的子任务处理者（subagent）。\n"
    "只专注完成分配给你的这一个任务，给出简洁明确的结果；不要反问，不要闲聊，"
    "不知道的信息就在结果里说明「未知」。"
)

_DREAM_TEMPLATE = PromptTemplate.from_template(
    "你是 mini-nanobot 的长期记忆巩固程序（Dream）。\n"
    "\n"
    "你会看到三个当前的长期记忆文件（SOUL.md / USER.md / MEMORY.md），以及自上次\n"
    "巩固以来新产生的一批对话历史摘要。你的任务：把这些新信息合理地归档进三个文件\n"
    "里，而不是简单堆砌。\n"
    "\n"
    "规则：\n"
    "1. MECE —— 每个事实只存一份，存到最合适的那个文件里：\n"
    "   - SOUL.md：agent 应该遵守的行为规则、工具使用策略\n"
    "   - USER.md：用户是谁、偏好什么、沟通风格\n"
    "   - MEMORY.md：项目上下文、长期有效的事实性知识\n"
    "2. 如果新信息和已有内容冲突，用新信息覆盖旧信息。\n"
    "3. 删除过时、重复、或者随手一查就能查到的信息（不要囤积无用信息）。\n"
    "4. 如果新历史里没有值得记录的新信息，对应文件原样返回即可，不要编造内容。\n"
    "5. 三个文件都用简洁的 Markdown 小节组织。\n"
    "\n"
    "## 当前 SOUL.md\n{soul}\n\n"
    "## 当前 USER.md\n{user}\n\n"
    "## 当前 MEMORY.md\n{memory}\n\n"
    "## 新增的历史摘要（自上次 Dream 以来，共 {entry_count} 条）\n{entries_text}"
)


def build_system_prompt(memory_files: dict[str, str], goal_state: dict[str, object] | None) -> str:
    """主 agent 的 system prompt：身份规则 + 当前长期记忆 + （如果有）进行中的目标。

    `memory_files` 就是 `MemoryStore.read_all_memory_files()` 的返回值，调用方
    每次构造 prompt 前都重新读一次文件，所以 Dream 刚更新完记忆，下一轮对话立刻
    就能用上最新内容——不需要额外的缓存失效逻辑。
    """
    goal_section = ""
    if goal_state and goal_state.get("status") == "active":
        goal_section = f"\n## 当前进行中的目标\n{goal_state.get('objective', '')}\n"
    return _MAIN_TEMPLATE.format(
        soul=memory_files.get("SOUL.md") or "（暂无）",
        user=memory_files.get("USER.md") or "（暂无）",
        memory=memory_files.get("MEMORY.md") or "（暂无）",
        goal_section=goal_section,
    )


def build_subagent_prompt() -> str:
    """子 agent 的 system prompt。目前没有可变部分，但保持函数形式方便以后扩展
    （比如以后想给子 agent 也传一段「精简版长期记忆」）。"""
    return _SUBAGENT_TEMPLATE.format()


def build_dream_prompt(memory_files: dict[str, str], entries_text: str, entry_count: int) -> str:
    """Dream 巩固时喂给 `with_structured_output` 的那条 prompt。"""
    return _DREAM_TEMPLATE.format(
        soul=memory_files.get("SOUL.md") or "",
        user=memory_files.get("USER.md") or "",
        memory=memory_files.get("MEMORY.md") or "",
        entries_text=entries_text,
        entry_count=entry_count,
    )
