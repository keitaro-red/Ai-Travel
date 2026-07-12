"""
=======================================================================
 memory.py — 记忆系统（三层记忆架构）
=======================================================================

这个文件合并了原项目的两个记忆文件：
  1. integrations/memory.py → 长期存储层（SQLite 向量搜索）
  2. assistant/memory.py → 短期窗口 + 图节点工厂

记忆的三层结构（从短到长）：

  短期记忆（对话窗口）
    最近 N 条消息（N=8）。超过后旧的压缩成摘要。
    存放位置：TravelAssistantState.conversation_summary
    代码位置：summarize_archived_messages()

  中期记忆（TaskMemory）
    当前会话的任务状态：目标城市、预算、偏好等。
    存放位置：TravelAssistantState.task_memory
    代码位置：update_task_memory()

  长期记忆（SQLite 向量数据库）
    跨会话持久化。用哈希嵌入把文本转成向量，
    算余弦相似度找相关内容。
    存放位置：data/long_term_memory.sqlite
    代码位置：SQLiteLongTermMemoryStore

文件内部按顺序分 5 节：
  第 1 节：哈希嵌入工具函数
  第 2 节：SQLite 长期记忆存储
  第 3 节：记忆管理器（封装读写操作）
  第 4 节：短期/中期记忆管理函数
  第 5 节：LangGraph 节点工厂
=======================================================================
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# LangChain 消息类型
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage
# RunnableLambda 把普通函数包装成 LangGraph 节点
from langchain_core.runnables import RunnableLambda
# REMOVE_ALL_MESSAGES 是 LangGraph 的"清空消息"标记
from langgraph.graph.message import REMOVE_ALL_MESSAGES
# RunnableConfig 包含了 thread_id 等运行时可配置参数
from langgraph.types import RunnableConfig

from config import (
    LONG_TERM_MEMORY_DB_PATH,
    LONG_TERM_MEMORY_TOP_K,
    MEMORY_MAX_SUMMARY_CHARS,
    MEMORY_SHORT_TERM_WINDOW,
    MEMORY_SUMMARY_TRIGGER_MESSAGES,
)
from state import QueryContext, TaskMemory, TravelAssistantState
from llm import get_embedding_model


# ============================================================
# 第 1 节：语义嵌入（文本 → 向量）
#
# 使用阿里云 DashScope Embedding API 把文本转成语义向量。
# 和之前手写的 SHA-256 哈希嵌入不同，真正的 Embedding 模型
# 能理解语义——"北京天气"和"上海天气"的向量距离会很近，
# 而它们和"你好"的距离会很远。
#
# 为什么这很重要：
#   长期记忆搜索不是简单的关键词匹配，而是要找到"语义相近"的内容。
#   手写哈希做不到这一点，只有真正的 Embedding API 才行。
# ============================================================

def utc_now_iso() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(timezone.utc).isoformat()


def embed_text(text: str) -> list[float]:
    """
    使用阿里 DashScope Embedding API 把文本转成语义向量。

    调用流程：
      文本 "北京天气" → DashScopeEmbeddings.embed_query()
      → 阿里云端模型处理 → 返回一个浮点数列表（如 1024 维）

    这个向量会被存到 SQLite 的 embedding_json 字段，
    搜索时用余弦相似度找到最相关的记忆。
    """
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return []
    model = get_embedding_model()
    return model.embed_query(cleaned)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """
    计算两个向量之间的余弦相似度。

    完整公式：cos(θ) = (A · B) / (|A| × |B|)

    其中：
      A·B = sum(aᵢ × bᵢ)        ← 点积（内积）
      |A|  = sqrt(sum(aᵢ²))      ← A 的模长（长度）
      |B|  = sqrt(sum(bᵢ²))      ← B 的模长（长度）

    DashScope Embedding API 返回的向量不一定归一化，
    所以不能直接用点积代替余弦相似度。

    结果范围：-1 到 1
      1  = 方向完全一致
      0  = 互相垂直（不相关）
      -1 = 方向完全相反
    """
    if not left or not right:
        return 0.0

    dot = sum(l * r for l, r in zip(left, right))
    norm_left = sum(v * v for v in left) ** 0.5
    norm_right = sum(v * v for v in right) ** 0.5

    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0

    return dot / (norm_left * norm_right)


# ============================================================
# 第 2 节：长期记忆存储层
#
# 用 SQLite 存储记忆，配合哈希嵌入进行向量搜索。
# 虽然不如专业的向量数据库快，但对个人项目完全够用。
#
# 表结构：
#   long_term_memories (
#     id           TEXT PRIMARY KEY,  -- 唯一标识（由 scope:type:content 的哈希生成）
#     scope        TEXT NOT NULL,     -- 作用域（通常 = thread_id）
#     memory_type  TEXT NOT NULL,     -- 类型：summary / goal / preference
#     content      TEXT NOT NULL,     -- 内容
#     metadata_json TEXT,             -- 元数据（JSON）
#     embedding_json TEXT,            -- 向量（JSON 数组）
#     created_at   TEXT,              -- 创建时间
#     updated_at   TEXT               -- 更新时间
#   )
# ============================================================

@dataclass(slots=True)
class MemoryRecord:
    """一条记忆记录的数据结构。@dataclass 自动生成 __init__ 等方法。"""
    id: str
    scope: str
    memory_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0  # 搜索时填入的相似度分数


class SQLiteLongTermMemoryStore:
    """
    SQLite 实现的长期记忆存储。

    搜索流程：
      1. 把查询文本也转成向量
      2. 从 SQLite 读出所有相关 scope 的记忆
      3. 逐一算余弦相似度
      4. 按分数排序，返回 top_k
    """

    def __init__(self, db_path: str = LONG_TERM_MEMORY_DB_PATH) -> None:
        self.db_path = str(db_path)
        # 确保数据库目录存在
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """打开 SQLite 连接。row_factory = sqlite3.Row 让查询结果可以用列名访问。"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        """创建表（如果不存在）。"""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            # 索引：按作用域快速查找
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ltm_scope
                ON long_term_memories(scope)
            """)
            # 联合索引：按作用域+类型查找
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ltm_scope_type
                ON long_term_memories(scope, memory_type)
            """)
            conn.commit()

    def search(self, *, scope: str, query: str, top_k: int,
               memory_types: Iterable[str] | None = None) -> list[MemoryRecord]:
        """
        搜索记忆：把 query 转成向量，和库里的每条记录算相似度。

        参数：
          scope: 作用域（通常 = thread_id）
          query: 查询文本（如"杭州旅游"）
          top_k: 最多返回几条
          memory_types: 限制只搜索某些类型（如只搜 preference）

        SQLite 的局限：没有原生向量索引，所以需要全量扫描。
        数据多了（>1万条）会变慢，这时需要换成 Milvus 等专业向量库。
        """
        query_vector = embed_text(query)
        # 构建 WHERE 条件
        clauses = ["scope = ?"]
        params: list[Any] = [scope]
        allowed = list(memory_types or [])
        if allowed:
            placeholders = ",".join("?" for _ in allowed)
            clauses.append(f"memory_type IN ({placeholders})")
            params.extend(allowed)
        sql = ("SELECT id, scope, memory_type, content, metadata_json, embedding_json "
               "FROM long_term_memories WHERE " + " AND ".join(clauses))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        # 算相似度
        records: list[MemoryRecord] = []
        for row in rows:
            emb = json.loads(row["embedding_json"])
            score = cosine_similarity(query_vector, emb)
            if score <= 0:  # 余弦相似度 ≤ 0 的不计入
                continue
            records.append(MemoryRecord(
                id=row["id"], scope=row["scope"], memory_type=row["memory_type"],
                content=row["content"], metadata=json.loads(row["metadata_json"]), score=score,
            ))
        records.sort(key=lambda r: r.score, reverse=True)
        return records[:top_k]

    def upsert(self, *, scope: str, memory_type: str, content: str,
               metadata: dict[str, Any] | None = None, record_id: str | None = None) -> str:
        """
        插入或更新一条记忆。

        upsert = update + insert
          如果 id 存在 → 更新（UPDATE）
          如果 id 不存在 → 插入（INSERT）

        id 由 scope + type + content 的 SHA-1 哈希生成，
        所以同样的内容只会存储一次，不会重复。
        """
        normalized = " ".join((content or "").strip().split())
        if not normalized:
            return record_id or ""
        now = utc_now_iso()
        rid = record_id or hashlib.sha1(
            f"{scope}:{memory_type}:{normalized}".encode("utf-8")).hexdigest()
        payload = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
        emb = json.dumps(embed_text(normalized))
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT created_at FROM long_term_memories WHERE id = ?", (rid,)).fetchone()
            created_at = existing["created_at"] if existing else now
            conn.execute("""
                INSERT OR REPLACE INTO long_term_memories
                (id, scope, memory_type, content, metadata_json, embedding_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (rid, scope, memory_type, normalized, payload, emb, created_at, now))
            conn.commit()
        return rid

    def delete_scope(self, scope: str) -> None:
        """删除某个 scope（通常是一个 thread）下的所有记忆。"""
        with self._connect() as conn:
            conn.execute("DELETE FROM long_term_memories WHERE scope = ?", (scope,))
            conn.commit()

    # ---- 异步版本（底层用 asyncio.to_thread 把同步操作丢到线程池里）----
    async def asearch(self, **kwargs) -> list[MemoryRecord]:
        return await asyncio.to_thread(self.search, **kwargs)

    async def aupsert(self, **kwargs) -> str:
        return await asyncio.to_thread(self.upsert, **kwargs)

    async def adelete_scope(self, scope: str) -> None:
        await asyncio.to_thread(self.delete_scope, scope)


# ============================================================
# 第 3 节：长期记忆管理器
#
# 封装了存储层的细节，提供更"贴近业务"的接口。
# recall：召回相关记忆（给图节点调用）
# remember：保存记忆（给图节点调用）
# ============================================================

class LongTermMemoryManager:
    def __init__(self, store: SQLiteLongTermMemoryStore, top_k: int = 3) -> None:
        self.store = store
        self.top_k = top_k

    def recall(self, *, scope: str, query: str,
               memory_types: Iterable[str] | None = None, top_k: int | None = None) -> list[str]:
        """召回记忆，直接返回文本列表（不暴露 MemoryRecord 的细节）。"""
        if not scope or not query.strip():
            return []
        records = self.store.search(scope=scope, query=query, top_k=top_k or self.top_k,
                                     memory_types=memory_types)
        return [r.content for r in records]

    async def arecall(self, **kwargs) -> list[str]:
        if not kwargs.get("scope") or not kwargs.get("query", "").strip():
            return []
        if "top_k" not in kwargs:
            kwargs["top_k"] = self.top_k
        records = await self.store.asearch(**kwargs)
        return [r.content for r in records]

    def remember(self, *, scope: str, conversation_summary: str, task_memory: dict[str, Any]) -> None:
        """
        保存本轮对话的记忆。

        保存三类记忆：
          1. summary — 对话摘要（如果有的话）
          2. goal — 当前目标（如果有的话）
          3. preference — 用户偏好（可能有多个）
        """
        if not scope:
            return
        if conversation_summary.strip():
            self.store.upsert(scope=scope, memory_type="summary", content=conversation_summary,
                              metadata={"kind": "conversation_summary"}, record_id=f"{scope}:summary")
        goal = str(task_memory.get("current_goal", "")).strip()
        if goal:
            self.store.upsert(scope=scope, memory_type="goal", content=goal,
                              metadata={"kind": "current_goal"}, record_id=f"{scope}:goal")
        for pref in task_memory.get("user_preferences", []):
            cleaned = " ".join(str(pref).split())
            if not cleaned:
                continue
            digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()
            self.store.upsert(scope=scope, memory_type="preference", content=cleaned,
                              metadata={"kind": "user_preference"},
                              record_id=f"{scope}:preference:{digest}")

    async def aremember(self, **kwargs) -> None:
        await asyncio.to_thread(self.remember, **kwargs)

    def delete_scope(self, scope: str) -> None:
        self.store.delete_scope(scope)

    async def adelete_scope(self, scope: str) -> None:
        await self.store.adelete_scope(scope)


def build_sqlite_memory_manager(db_path: str = LONG_TERM_MEMORY_DB_PATH,
                                top_k: int = LONG_TERM_MEMORY_TOP_K) -> LongTermMemoryManager:
    """快捷工厂函数：创建一个 SQLite 后端的记忆管理器。"""
    return LongTermMemoryManager(SQLiteLongTermMemoryStore(db_path=db_path), top_k=top_k)


# ============================================================
# 第 4 节：短期/中期记忆管理函数
#
# 这些函数操作 TravelAssistantState 中的任务记忆字段。
# 不涉及 SQLite，只处理当前会话的数据。
# ============================================================

# 三个正则表达式：从用户输入中提取结构化信息
BUDGET_PATTERN = re.compile(r"(预算[^\n,，。.!?；;]{0,16})")       # "预算3000"
TRIP_DAYS_PATTERN = re.compile(r"([0-9一二三四五六七八九十两]+天)")  # "3天"
PREFERENCE_PATTERN = re.compile(                                    # "喜欢自然风光"
    r"(喜欢[^,，。.!?\n]{1,18}|偏好[^,，。.!?\n]{1,18}|尽量不要[^,，。.!?\n]{1,18}|不要[^,，。.!?\n]{1,18})")


def ensure_unique_items(values: list[str], limit: int = 6) -> list[str]:
    """去重并限制列表长度，保留最新的（数组末尾）。"""
    ordered: list[str] = []
    seen: set[str] = set()
    for v in values:
        cleaned = " ".join(str(v).split())
        if not cleaned or cleaned in seen:
            continue
        ordered.append(cleaned)
        seen.add(cleaned)
    return ordered[-limit:] if len(ordered) > limit else ordered


def extract_latest_assistant_text(messages: list[BaseMessage]) -> str:
    """从消息列表中找到最新的助手回复文本。"""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            return msg.content.strip()
        # 如果 content 是列表（有些模型返回的 content 是复杂结构），跳过
    return ""


def summarize_message(message: BaseMessage) -> str:
    """
    把一条消息压缩成一行摘要。

    输出格式：
      User: 你好
      Assistant: 北京明天天气晴，温度...
    超长部分截断到 120 字。
    """
    if isinstance(message, HumanMessage):
        role = "User"
    elif isinstance(message, AIMessage):
        role = "Assistant"
    else:
        role = type(message).__name__  # 比如 ToolMessage
    content = getattr(message, "content", "")
    text = content if isinstance(content, str) else str(content)
    compact = " ".join(text.split())
    if len(compact) > 120:
        compact = f"{compact[:117]}..."
    return f"{role}: {compact}"


def summarize_archived_messages(existing_summary: str, archived: list[BaseMessage],
                                max_chars: int = MEMORY_MAX_SUMMARY_CHARS) -> str:
    """
    把一批旧消息压缩成摘要文本。

    会保留已有的历史摘要（跨多次压缩），然后追加新归档的消息摘要。
    控制总长度不超过 max_chars。
    """
    lines = [summarize_message(m) for m in archived if summarize_message(m)]
    if not lines:
        return existing_summary
    parts: list[str] = []
    if existing_summary.strip():
        parts.append(existing_summary.strip())
    parts.append("Archived history: " + " | ".join(lines))
    combined = "\n".join(parts)
    return combined if len(combined) <= max_chars else combined[-max_chars:]


# ---- 从纯文本中提取结构化信息的辅助函数 ----
def extract_budget_text(text: str) -> str:
    m = BUDGET_PATTERN.search(text)
    return m.group(1).strip() if m else ""

def extract_trip_days(text: str) -> str:
    m = TRIP_DAYS_PATTERN.search(text)
    return m.group(1).strip() if m else ""

def extract_preferences(text: str) -> list[str]:
    return [m.strip() for m in PREFERENCE_PATTERN.findall(text)]


def update_task_memory(state: TravelAssistantState) -> TaskMemory:
    """
    从当前对话状态中提取信息，更新到 task_memory。

    这个函数在每一轮对话结束时（finalize_memory 节点）调用。
    提取的信息包括：城市、预算、天数、偏好等。

    这些信息会：
      1. 保存到 TaskMemory（供下轮使用）
      2. 持久化到长期记忆 SQLite（跨会话使用）
    """
    tm: TaskMemory = dict(state.get("task_memory", {}))
    qc: QueryContext = state.get("query_context", {})
    messages = state.get("messages", [])
    latest_user = qc.get("raw_user_input", "")
    latest_reply = extract_latest_assistant_text(messages)

    if latest_user:
        tm["latest_user_request"] = latest_user
        tm["current_goal"] = latest_user
    if latest_reply:
        tm["latest_assistant_reply"] = latest_reply
    if qc.get("intent"):
        tm["latest_intent"] = qc["intent"]
    if qc.get("time_text"):
        tm["latest_time_text"] = qc["time_text"]
    if state.get("active_agent"):
        tm["last_active_agent"] = state["active_agent"]

    city = qc.get("normalized_city") or qc.get("location_text", "")
    if city:
        tm["confirmed_city"] = city
        recent = list(tm.get("recent_cities", []))
        recent.append(city)
        tm["recent_cities"] = ensure_unique_items(recent)

    budget = extract_budget_text(latest_user)
    if budget:
        tm["budget_text"] = budget
    days = extract_trip_days(latest_user)
    if days:
        tm["trip_days"] = days
    prefs = list(tm.get("user_preferences", []))
    prefs.extend(extract_preferences(latest_user))
    tm["user_preferences"] = ensure_unique_items(prefs)
    return tm


def format_task_memory(tm: TaskMemory) -> str:
    """把结构化的 TaskMemory 转成文字段落，给 AI 模型看。"""
    if not tm:
        return ""
    lines = ["Current structured task memory:"]
    for key, label in (
        ("current_goal", "Current goal"), ("latest_intent", "Latest intent"),
        ("confirmed_city", "Confirmed city"), ("recent_cities", "Recent cities"),
        ("budget_text", "Budget"), ("trip_days", "Trip length"),
        ("latest_time_text", "Time hint"), ("user_preferences", "Preferences"),
    ):
        v = tm.get(key)
        if not v:
            continue
        rendered = ", ".join(str(x) for x in v) if isinstance(v, list) else str(v)
        lines.append(f"- {label}: {rendered}")
    return "\n".join(lines)


def build_memory_context_messages(state: TravelAssistantState) -> list[SystemMessage]:
    """
    构建"记忆上下文"消息列表。

    这三条 SystemMessage 会插入到发给 AI 模型的消息列表里：
      1. 历史对话摘要（如果有）
      2. 当前任务状态（目标、城市等）
      3. 从长期记忆中召回的记录（如果有）
    """
    msgs: list[SystemMessage] = []
    summary = state.get("conversation_summary", "").strip()
    if summary:
        msgs.append(SystemMessage(content=f"Conversation summary from earlier turns:\n{summary}"))
    formatted = format_task_memory(state.get("task_memory", {}))
    if formatted:
        msgs.append(SystemMessage(content=formatted))
    recalled = state.get("recalled_memories", [])
    if recalled:
        lines = "\n".join(f"- {item}" for item in recalled)
        msgs.append(SystemMessage(content=f"Relevant long-term memories:\n{lines}"))
    return msgs


def resolve_memory_scope(state: TravelAssistantState, config: RunnableConfig | None) -> str:
    """
    确定当前记忆的作用域（scope）。

    scope 通常等于 thread_id，作用是隔离不同会话的记忆。
    不同会话的记忆不应该互相看到。
    """
    if state.get("memory_scope"):
        return state["memory_scope"]
    cfg = (config or {}).get("configurable", {})
    return str(cfg.get("memory_scope") or cfg.get("thread_id") or "")


def build_recall_query(state: TravelAssistantState) -> str:
    """构建"搜索查询"：从当前问题和记忆中提取关键词，用于在长期记忆中搜索。"""
    tm = state.get("task_memory", {})
    qc = state.get("query_context", {})
    fragments = [
        qc.get("raw_user_input", ""),
        tm.get("current_goal", ""),
        " ".join(tm.get("user_preferences", [])),
    ]
    return " ".join(f for f in fragments if f).strip()


# ============================================================
# 第 5 节：LangGraph 图节点工厂
#
# 这些函数创建图中使用的"节点"（Node）。
# 一个节点在图中就是一个工作台，执行具体的处理步骤。
# ============================================================

def route_after_specialist_response(state: TravelAssistantState):
    """
    条件路由函数：专职 Agent 完成一次输出后，下一步去哪？

    判断条件：
      检查最新的一条 AIMessage 是否包含 tool_calls。
      - 有工具调用 → 去对应的工具节点（weather_tools / travel_tools）
      - 没有工具调用 → 去 finalize_memory 收尾
    """
    messages = state.get("messages", [])
    if messages:
        latest = messages[-1]
        if isinstance(latest, AIMessage) and getattr(latest, "tool_calls", None):
            return "tools"       # Agent 想调用工具
    return "finalize_memory"     # Agent 直接给出了回答


def create_recall_memory_node(memory_manager: LongTermMemoryManager):
    """
    创建"召回记忆"节点工厂。

    这个节点会在 select_agent 之前执行，
    从长期记忆中检索与当前问题相关的内容。
    检索结果存到 recalled_memories 字段，供 Agent 参考。
    """
    def recall_sync(state: TravelAssistantState, config: RunnableConfig | None = None):
        scope = resolve_memory_scope(state, config)
        query = build_recall_query(state)
        recalled = memory_manager.recall(scope=scope, query=query) if scope else []
        return {"memory_scope": scope, "recalled_memories": recalled}

    async def recall_async(state: TravelAssistantState, config: RunnableConfig | None = None):
        scope = resolve_memory_scope(state, config)
        query = build_recall_query(state)
        recalled = await memory_manager.arecall(scope=scope, query=query) if scope else []
        return {"memory_scope": scope, "recalled_memories": recalled}

    # RunnableLambda 把普通函数包装成 LangGraph 可识别的节点
    return RunnableLambda(recall_sync, afunc=recall_async, name="recall_memory")


def create_finalize_memory_node(memory_manager: LongTermMemoryManager):
    """
    创建"收尾记忆"节点工厂。

    每一轮对话结束时执行，做三件事：
      1. 更新 TaskMemory（提取本轮的新信息）
      2. 如果消息太多，压缩旧消息为摘要
      3. 保存到长期记忆（供以后召回用）
    """
    def _finalize_sync(state: TravelAssistantState, config: RunnableConfig | None = None) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        updated_tm = update_task_memory(state)
        summary = state.get("conversation_summary", "")

        # 如果消息数量超过阈值，压缩旧消息
        if len(messages) > MEMORY_SUMMARY_TRIGGER_MESSAGES:
            keep = max(MEMORY_SHORT_TERM_WINDOW, 1)
            archive = messages[:len(messages) - keep]
            kept = messages[len(messages) - keep:]
            summary = summarize_archived_messages(summary, archive)
            # RemoveMessage 是 LangGraph 用来删除旧消息的特殊消息类型
            msg_update: list[Any] = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept]
        else:
            msg_update = []

        # 保存到长期记忆
        scope = resolve_memory_scope(state, config)
        if scope:
            memory_manager.remember(scope=scope, conversation_summary=summary, task_memory=updated_tm)

        result: dict[str, Any] = {
            "task_memory": updated_tm, "conversation_summary": summary, "memory_scope": scope,
        }
        if msg_update:
            result["messages"] = msg_update
        return result

    async def _finalize_async(state: TravelAssistantState, config: RunnableConfig | None = None) -> dict[str, Any]:
        """异步版本的 _finalize（和同步版本逻辑完全一样，只是用了 await）"""
        messages = list(state.get("messages", []))
        updated_tm = update_task_memory(state)
        summary = state.get("conversation_summary", "")
        if len(messages) > MEMORY_SUMMARY_TRIGGER_MESSAGES:
            keep = max(MEMORY_SHORT_TERM_WINDOW, 1)
            archive = messages[:len(messages) - keep]
            kept = messages[len(messages) - keep:]
            summary = summarize_archived_messages(summary, archive)
            msg_update: list[Any] = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept]
        else:
            msg_update = []
        scope = resolve_memory_scope(state, config)
        if scope:
            await memory_manager.aremember(scope=scope, conversation_summary=summary, task_memory=updated_tm)
        result: dict[str, Any] = {
            "task_memory": updated_tm, "conversation_summary": summary, "memory_scope": scope,
        }
        if msg_update:
            result["messages"] = msg_update
        return result

    return RunnableLambda(_finalize_sync, afunc=_finalize_async, name="finalize_memory")
