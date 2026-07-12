"""
=========================================================================
 graph.py — 多智能体工作流图的"蓝图"
=========================================================================

【这个文件是整个项目的"骨架"】

它负责"画"出整个 AI Agent 的工作流程图。
想象你在画一张带分支的流程图——每个方框是"工作台（Node）"，
箭头是"传送带（Edge）"，条件分支是"分拣员（Conditional Edge）"。

对比原项目（7 个图构建函数），这里精简为 2 个：
  1. compile_graph — 纯图构建（核心）
  2. build_persistent_graph — 加上 SQLite 持久化的完整图

工作流程图概览：

                    START
                      │
                      ▼
              ┌───────────────┐
              │ analyze_query  │  ← 意图分析（关键词匹配）
              │ （意图分析）     │
              └───────┬───────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
    ┌──────────┐          ┌──────────────┐
    │  clarify  │          │ recall_memory│  ← 召回长期记忆
    │ （追问澄清）│          └──────┬───────┘
    └─────┬────┘                 │
          │                      ▼
          │               ┌──────────────┐
          │               │ select_agent  │  ← 选择专职 Agent
          │               │ （路由分配）    │
          │               └──────┬───────┘
          │                      │
          │     ┌────────────────┼────────────────┐
          │     ▼                ▼                ▼
          │ ┌────────┐    ┌──────────┐    ┌────────────┐
          │ │weather │    │  travel   │    │  general    │
          │ │ Agent  │    │  Agent    │    │  Agent      │
          │ └───┬────┘    └────┬─────┘    └─────┬──────┘
          │     │              │                │
          │     ▼              ▼                │
          │ ┌────────┐    ┌──────────┐          │
          │ │weather │    │  travel  │          │
          │ │ tools  │    │  tools   │          │
          │ └───┬────┘    └────┬─────┘          │
          │     │              │                │
          │     └──────┬───────┘                │
          │            ▼                        │
          │    ┌──────────────┐                 │
          │    │finalize_memory│  ← 收尾：保存记忆  │
          │    └──────┬───────┘                 │
          │           │                         │
          │           ▼                         │
          │        END ◄────────────────────────┘
          └──────────────────────────────────┘

核心循环（Agent Loop）：
  1. 分析用户输入 → 2. 如果信息不足则追问
  3. 召回长期记忆 → 4. 选择专职 Agent
  5. Agent 工作 → 6. 如需工具则调用 → 7. 回到 Agent 继续
  8. 收尾保存 → 完成
=========================================================================
"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Iterable

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from config import CHECKPOINT_DB_PATH, LONG_TERM_MEMORY_DB_PATH, LONG_TERM_MEMORY_TOP_K
from llm import get_model
from memory import (
    build_sqlite_memory_manager,
    create_finalize_memory_node,
    create_recall_memory_node,
    route_after_specialist_response,
)
from mcp_tools import load_amap_tools
from nodes import (
    analyze_query,
    clarify_query,
    create_specialist_node,
    route_after_analysis,
    route_to_specialist,
    select_specialist_agent,
)
from state import TravelAssistantState


# ============================================================
# 辅助函数
# ============================================================

async def create_sqlite_checkpointer(db_path: str = CHECKPOINT_DB_PATH) -> AsyncSqliteSaver:
    """
    创建 SQLite 检查点存储器。

    Checkpointer（检查点）是 LangGraph 持久化对话状态的机制。
    每次对话结束后，它会把 TravelAssistantState 的所有字段
    保存到 SQLite 数据库。

    下次用同一个 thread_id 调用时，自动恢复之前的状态，
    所以 AI 助手能"记住"上轮说了什么。

    注意：不是保存整段对话文本，而是保存"共享黑板"的完整状态。
    """
    cp = Path(db_path)
    cp.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(cp)
    return AsyncSqliteSaver(conn)


def tool_name(tool) -> str:
    """获取工具的名称，兼容不同的命名属性。"""
    return getattr(tool, "name", getattr(tool, "__name__", str(tool)))


def select_tools(tools: Iterable, allowed: set[str]) -> list:
    """
    从全量工具列表中筛选出某个 Agent 允许使用的工具。

    比如天气 Agent 只能用 weather 和 input_tips，
    不能用地理解析工具——这是一种安全边界，
    确保每个 Agent 不会"越权"调用不该用的工具。
    """
    return [t for t in tools if tool_name(t) in allowed]


def clone_model(model):
    """
    浅拷贝模型实例。

    为什么要拷贝？因为每个 Agent 需要绑定不同的工具集。
    如果不拷贝，model.bind_tools(weather_tools) 会修改原模型，
    导致其他 Agent 也被绑上不相关的工具。
    """
    try:
        return copy(model)
    except Exception:
        return model


def bind_agent_model(model, tools):
    """
    把工具集"绑"到模型上。

    绑了工具的模型在对话时，会自动判断是否该调用工具。
    没绑工具的模型（如 General Agent）永远不会调用工具。
    """
    m = clone_model(model)
    resolved = list(tools)
    return m.bind_tools(resolved) if resolved else m


# ============================================================
# 【核心函数】compile_graph：编译多智能体图
#
# 这个函数做了以下几件事：
#   1. 为每个专职 Agent 分配允许使用的工具
#   2. 创建所有节点（工作台）
#   3. 在节点之间连接边（箭头指向）
#   4. 设置条件边（分支判断）
#   5. 编译生成可调用的图
#
# 参数：
#   model: AI 模型实例（默认使用 DeepSeek）
#   tools: 工具列表（从高德 MCP 获取）
#   checkpointer: SQLite 检查点（用于持久化）
#   memory_manager: 长期记忆管理器
# ============================================================

def compile_graph(model=None, tools=None, checkpointer=None, memory_manager=None):
    """编译多智能体图。"""
    resolved_tools = list(tools or [])
    resolved_model = model or get_model()
    resolved_mm = memory_manager or build_sqlite_memory_manager()

    # ==== 为每个 Agent 分配工具集 ====
    # Weather Agent：天气 + 地点提示
    weather_tools = select_tools(resolved_tools, {"weather", "input_tips"})
    # Travel Agent：天气 + 地点提示（规划需要天气和搜索能力）
    travel_tools = select_tools(resolved_tools, {"weather", "input_tips"})
    # General Agent：没有工具（纯聊天，不需要调任何外部 API）

    # ==== 创建 3 个专职 Agent ====
    weather_agent = create_specialist_node(
        bind_agent_model(resolved_model, weather_tools), "weather_agent",
    )
    travel_agent = create_specialist_node(
        bind_agent_model(resolved_model, travel_tools), "travel_agent",
    )
    general_agent = create_specialist_node(
        bind_agent_model(resolved_model, []), "general_agent",
    )

    # ==== 构建图结构 ====
    builder = StateGraph(TravelAssistantState)

    # ---- 添加节点 ----
    builder.add_node("analyze_query", analyze_query)          # 意图分析
    builder.add_node("clarify", clarify_query)                # 追问澄清
    builder.add_node("recall_memory", create_recall_memory_node(resolved_mm))  # 召回记忆
    builder.add_node("select_agent", select_specialist_agent)  # 选择 Agent
    builder.add_node("weather_agent", weather_agent)           # 天气 Agent
    builder.add_node("travel_agent", travel_agent)             # 旅行 Agent
    builder.add_node("general_agent", general_agent)           # 通用 Agent
    builder.add_node("weather_tools", ToolNode(weather_tools)) # 天气工具
    builder.add_node("travel_tools", ToolNode(travel_tools))   # 旅行工具
    builder.add_node("finalize_memory", create_finalize_memory_node(resolved_mm))  # 收尾

    # ---- 连接边（画箭头）----

    # 1. 入口 → 意图分析
    builder.add_edge(START, "analyze_query")

    # 2. 意图分析后分岔：
    #    - 信息不足 → clarify（追问）
    #    - 信息足够 → recall_memory（召回记忆）
    builder.add_conditional_edges(
        "analyze_query", route_after_analysis,
        {"clarify": "clarify", "select_agent": "recall_memory"},
    )

    # 3. 追问后直接收尾（等用户下一轮回复）
    builder.add_edge("clarify", "finalize_memory")

    # 4. 记忆召回 → 选择 Agent
    builder.add_edge("recall_memory", "select_agent")

    # 5. 选择 Agent → 分派到具体的专职 Agent
    builder.add_conditional_edges(
        "select_agent", route_to_specialist,
        {"weather_agent": "weather_agent", "travel_agent": "travel_agent",
         "general_agent": "general_agent"},
    )

    # 6. 专职 Agent 输出后判断：
    #    - 有工具调用（tool_calls）→ 去工具节点
    #    - 直接回答 → 去收尾
    builder.add_conditional_edges(
        "weather_agent", route_after_specialist_response,
        {"tools": "weather_tools", "finalize_memory": "finalize_memory"},
    )
    builder.add_conditional_edges(
        "travel_agent", route_after_specialist_response,
        {"tools": "travel_tools", "finalize_memory": "finalize_memory"},
    )
    builder.add_conditional_edges(
        "general_agent", route_after_specialist_response,
        {"tools": END, "finalize_memory": "finalize_memory"},
    )

    # 7. 工具执行完后回到对应的 Agent（形成 Agent + Tool 循环）
    #    这个循环可以执行多次：Agent 可以依次调用多个工具
    builder.add_edge("weather_tools", "weather_agent")
    builder.add_edge("travel_tools", "travel_agent")

    # 8. 收尾 → 结束
    builder.add_edge("finalize_memory", END)

    # 编译图
    graph = builder.compile(checkpointer=checkpointer)
    setattr(graph, "memory_manager", resolved_mm)
    return graph


# ============================================================
# build_persistent_graph：构建"生产级"的持久化图
#
# 这是 CLI 和 Web 应用实际使用的函数。
# 它给 compile_graph 加上了两样东西：
#   1. SQLite Checkpointer — 持久化对话状态
#   2. SQLite 长期记忆管理器 — 跨会话保留用户偏好
# ============================================================

async def build_persistent_graph(model=None, tools=None):
    """构建带 SQLite checkpoint 和长期记忆的持久化图。"""
    # 创建 checkpoint 存储器
    checkpointer = await create_sqlite_checkpointer()
    # 加载高德 MCP 工具（如果没有外部传入的话）
    resolved_tools = list(tools) if tools else list(await load_amap_tools())
    # 创建长期记忆管理器
    mm = build_sqlite_memory_manager(
        db_path=LONG_TERM_MEMORY_DB_PATH,
        top_k=LONG_TERM_MEMORY_TOP_K,
    )
    return compile_graph(
        model=model,
        tools=resolved_tools,
        checkpointer=checkpointer,
        memory_manager=mm,
    )
