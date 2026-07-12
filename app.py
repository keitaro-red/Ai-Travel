"""
=========================================================================
 app.py — FastAPI Web 应用（合并后端 + 会话管理）
=========================================================================

这个文件合并了原项目的三个文件：
  1. app/web.py — uvicorn 入口
  2. backend/api.py — 所有 API 路由
  3. backend/thread_store.py — 会话元数据管理

合并的原因：这三个文件本质上是 "Web 服务的三个部分"，
分开反而增加了不必要的心智负担（比如 api.py 还要从 cli.py 导入函数）。

API 路由一览：
  GET  /                    → 返回 index.html 页面
  GET  /api/threads         → 获取所有会话摘要列表
  POST /api/threads         → 创建新会话
  GET  /api/threads/{tid}   → 获取某个会话的详情和消息
  DELETE /api/threads/{tid} → 删除会话（含 checkpoint 和记忆）
  POST /api/chat            → 发送消息（非流式）
  POST /api/chat/stream     → 发送消息（流式 NDJSON）

前端的 NDJSON 数据流格式：
  {"type":"thread","thread_id":"abc123"}
  {"type":"process","key":"analyze_query","status":"running","title":"分析用户输入",...}
  {"type":"assistant_start"}
  {"type":"assistant_delta","content":"北京"}
  {"type":"done","thread":{...},"messages":[...]}
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

import aiosqlite
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from graph import build_persistent_graph

# 前端静态文件目录（和 app.py 同级的 static/ 文件夹）
FRONTEND_DIR = Path(__file__).resolve().parent / "static"
# 会话元数据存储路径（和 checkpoint 在同一目录）
THREAD_STORE_DB_PATH = "data/thread_store.sqlite"


# ============================================================
# Pydantic 模型：定义 API 请求/响应数据的格式
# Pydantic 自动做 JSON 解析和验证，省去手写校验代码
# ============================================================

class ChatRequest(BaseModel):
    """聊天请求体"""
    message: str                     # 用户消息文本
    thread_id: str | None = None     # 会话 ID（为空时自动创建新会话）

class ChatResponse(BaseModel):
    """聊天响应体"""
    thread_id: str
    reply: str

class ThreadSummary(BaseModel):
    """会话摘要（用于左侧列表展示）"""
    thread_id: str
    title: str
    created_at: str
    updated_at: str
    last_user_message: str
    last_assistant_message: str

class MessageView(BaseModel):
    """单条消息的展示格式"""
    role: str      # "user" 或 "assistant"
    content: str   # 消息文本

class ThreadDetail(BaseModel):
    """会话详情（元数据 + 完整消息历史）"""
    thread: ThreadSummary
    messages: list[MessageView]


# ============================================================
# 第 1 块：会话元数据管理（Thread CRUD）
# 原项目在 backend/thread_store.py 中
# ============================================================

def _utc_now() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(timezone.utc).isoformat()

def _default_title() -> str:
    """新会话的默认标题（在用户发第一条消息之前使用）。"""
    return "新会话"

@dataclass
class ThreadRecord:
    """会话记录（映射到 SQLite chat_threads 表的一行）"""
    thread_id: str
    title: str
    created_at: str
    updated_at: str
    last_user_message: str
    last_assistant_message: str

    def to_dict(self) -> dict[str, str]:
        return {k: getattr(self, k) for k in (
            "thread_id", "title", "created_at", "updated_at",
            "last_user_message", "last_assistant_message")}

async def _connect_store() -> aiosqlite.Connection:
    """连接会话元数据库（自动创建 data 目录）。"""
    path = Path(THREAD_STORE_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(path)
    conn.row_factory = aiosqlite.Row
    return conn

async def _init_store(conn: aiosqlite.Connection):
    """创建 chat_threads 表（如果不存在）。"""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_threads (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_user_message TEXT NOT NULL DEFAULT '',
            last_assistant_message TEXT NOT NULL DEFAULT ''
        )
    """)
    await conn.commit()

async def create_thread(conn: aiosqlite.Connection, tid: str) -> ThreadRecord:
    """创建一条新会话记录。"""
    now = _utc_now()
    await conn.execute(
        "INSERT INTO chat_threads VALUES (?,?,?,?,'','')",
        (tid, _default_title(), now, now))
    await conn.commit()
    return ThreadRecord(tid, _default_title(), now, now, "", "")

async def get_thread(conn: aiosqlite.Connection, tid: str) -> ThreadRecord | None:
    """按 thread_id 查询一条会话记录。"""
    cur = await conn.execute("SELECT * FROM chat_threads WHERE thread_id=?", (tid,))
    row = await cur.fetchone()
    return ThreadRecord(**{k: row[k] for k in row.keys()}) if row else None

async def list_threads(conn: aiosqlite.Connection) -> list[ThreadRecord]:
    """获取所有会话，按最后更新时间倒序（最新在前面）。"""
    cur = await conn.execute("SELECT * FROM chat_threads ORDER BY updated_at DESC")
    return [ThreadRecord(**{k: r[k] for k in r.keys()}) for r in await cur.fetchall()]

async def ensure_thread(conn: aiosqlite.Connection, tid: str) -> ThreadRecord:
    """确保会话存在：有则返回，无则创建。"""
    existing = await get_thread(conn, tid)
    return existing or await create_thread(conn, tid)

async def update_thread_after_chat(conn: aiosqlite.Connection, tid: str,
                                    user_msg: str, assistant_msg: str) -> ThreadRecord:
    """一轮对话结束后更新会话摘要信息。

    如果当前还是默认标题，从用户第一条消息截取 24 字作为标题。
    """
    existing = await ensure_thread(conn, tid)
    title = existing.title
    if title == _default_title() and user_msg.strip():
        title = user_msg.strip()[:24]
    now = _utc_now()
    await conn.execute(
        "UPDATE chat_threads SET title=?,updated_at=?,last_user_message=?,last_assistant_message=? WHERE thread_id=?",
        (title, now, user_msg, assistant_msg, tid))
    await conn.commit()
    return ThreadRecord(tid, title, existing.created_at, now, user_msg, assistant_msg)

async def delete_thread(conn: aiosqlite.Connection, tid: str) -> bool:
    """删除一条会话记录。"""
    cur = await conn.execute("DELETE FROM chat_threads WHERE thread_id=?", (tid,))
    await conn.commit()
    return cur.rowcount > 0


# ============================================================
# 第 2 块：后端工具函数
# 原项目散落在 app/cli.py 和 backend/api.py 中，现统一放在这里
# ============================================================

def next_thread_id() -> str:
    """生成一个新的 thread_id（使用 UUID 十六进制字符串）。"""
    return uuid4().hex

def make_graph_config(tid: str) -> dict:
    """构建 LangGraph 的运行时配置，主要包含 thread_id。

    这是 checkpoint 的"钥匙"——同一个 thread_id 恢复同一个会话。
    """
    return {"configurable": {"thread_id": tid}}

def extract_last_ai_text(messages) -> str:
    """从 LangGraph 返回的消息列表中提取最新的助手文本。"""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""

def chunk_text(text: str, size: int = 24) -> list[str]:
    """把长文本切成小块（用于流式输出的回退方案）。"""
    return [text[i:i+size] for i in range(0, len(text), size)] if text else [""]

def ndjson_line(payload: dict) -> bytes:
    """把字典编码成 NDJSON 的一行（JSON + 换行符）。

    NDJSON = Newline-Delimited JSON
    每行是一个独立的 JSON 对象，用 \n 分隔。
    前端可以逐行读取并解析，不需要等待整个响应结束。
    """
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")

def extract_stream_text(chunk) -> str:
    """
    从模型流式输出的块（chunk）中提取纯文本。

    模型生成回答时是一个个 token 块（chunk）吐出来的。
    每个 chunk 的 content 可能是：
      - 字符串：直接取
      - 列表：[{"text": "北京"}, ...]：遍历取 text 字段
    """
    if chunk is None:
        return ""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return str(content) if content else ""

def summarize_value(value: Any, limit: int = 96) -> str:
    """把任意类型的值压缩成一行可读文本（用于时间线显示）。"""
    if value is None:
        return ""
    # ToolMessage 等 LangChain 消息类型有 content 属性
    if hasattr(value, "content"):
        value = value.content
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except TypeError:
            text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit-1]}..."

def summarize_query_context(qc: dict[str, Any]) -> str:
    """把查询上下文浓缩成一句文字（时间线显示用）。"""
    parts = []
    if qc.get("intent"):
        parts.append(f"意图: {qc['intent']}")
    city = qc.get("normalized_city") or qc.get("location_text")
    if city:
        parts.append(f"地点: {city}")
    if qc.get("time_text"):
        parts.append(f"时间: {qc['time_text']}")
    if qc.get("needs_clarification"):
        parts.append(f"需澄清: {qc.get('clarification_reason', '信息不足')}")
    return " | ".join(parts) if parts else "已完成查询预分析。"

def summarize_agent_selection(data: dict[str, Any]) -> str:
    """把 Agent 选择结果浓缩成一句。"""
    agent = data.get("active_agent", "general_agent")
    reason = data.get("agent_selection_reason", "")
    return f"已选择 {agent}。{reason}".strip()

async def delete_graph_thread_data(graph, tid: str):
    """删除某个会话在 checkpoint 和长期记忆中的所有数据。

    注意：不仅要删 chat_threads 表的记录，还要删：
      - checkpoints 表（会话状态快照）
      - writes 表（待写数据）
      - 长期记忆（用户偏好、历史摘要等）
    """
    cp = getattr(graph, "checkpointer", None)
    conn = getattr(cp, "conn", None)
    if conn is not None:
        await conn.execute("DELETE FROM writes WHERE thread_id=?", (tid,))
        await conn.execute("DELETE FROM checkpoints WHERE thread_id=?", (tid,))
        await conn.commit()
    mm = getattr(graph, "memory_manager", None)
    if mm is not None:
        await mm.adelete_scope(tid)

def message_to_view(msg: BaseMessage) -> MessageView | None:
    """把 LangChain 消息对象转成前端友好的 MessageView 格式。"""
    if isinstance(msg, HumanMessage):
        return MessageView(role="user", content=msg.content if isinstance(msg.content, str) else str(msg.content))
    if isinstance(msg, AIMessage):
        c = msg.content if isinstance(msg.content, str) else str(msg.content)
        return MessageView(role="assistant", content=c) if c else None
    return None  # ToolMessage 等不展示

async def load_thread_detail(graph, record) -> ThreadDetail:
    """从 checkpoint 恢复某个会话的完整消息历史。"""
    snap = await graph.aget_state(make_graph_config(record.thread_id))
    msgs = [v for v in (message_to_view(m) for m in snap.values.get("messages", [])) if v is not None]
    return ThreadDetail(thread=ThreadSummary(**record.to_dict()), messages=msgs)


# ============================================================
# 第 3 块：LangGraph 事件 → 前端时间线事件的映射
#
# LangGraph 的 astream_events 会发出各种底层事件：
#   on_chain_start, on_chain_end, on_chat_model_stream 等
#
# 这个函数把这些底层事件"翻译"成前端时间线组件能理解的格式。
# 前端收到后直接渲染成：
#   [●] 分析用户输入（running）
#   [●] 分析完成（done）
#   [●] 选择专职 Agent（done）
#   [●] weather_agent 开始处理（running）
# ============================================================

def build_process_updates(event: dict[str, Any], tracker: dict[str, Any]) -> list[dict[str, Any]]:
    ev_type = event.get("event")
    name = event.get("name")
    data = event.get("data", {})
    run_id = event.get("run_id", "")
    updates: list[dict[str, Any]] = []

    # ---- 意图分析 ----
    if ev_type == "on_chain_start" and name == "analyze_query":
        updates.append({
            "type": "process", "key": "analyze_query", "stage": "analysis",
            "status": "running", "title": "分析用户输入",
            "detail": "提取意图、地点和时间线索。",
        })
    elif ev_type == "on_chain_end" and name == "analyze_query":
        qc = data.get("output", {}).get("query_context", {})
        updates.append({
            "type": "process", "key": "analyze_query", "stage": "analysis",
            "status": "done", "title": "分析完成",
            "detail": summarize_query_context(qc),
        })

    # ---- 路由选择 ----
    elif ev_type == "on_chain_end" and name == "route_after_analysis":
        route = data.get("output", "")
        detail = "信息不足，先进入澄清节点。" if route == "clarify" else "信息足够，准备选择专职 agent。"
        updates.append({
            "type": "process", "key": "route_after_analysis", "stage": "route",
            "status": "done", "title": "选择下一步", "detail": detail,
        })
    elif ev_type == "on_chain_start" and name == "select_agent":
        updates.append({
            "type": "process", "key": "select_agent", "stage": "route",
            "status": "running", "title": "选择专职 Agent",
            "detail": "根据预分析结果分配给最合适的专职 agent。",
        })
    elif ev_type == "on_chain_end" and name == "select_agent":
        updates.append({
            "type": "process", "key": "select_agent", "stage": "route",
            "status": "done", "title": "专职 Agent 已选定",
            "detail": summarize_agent_selection(data.get("output", {})),
        })

    # ---- 记忆 ----
    elif ev_type == "on_chain_start" and name == "recall_memory":
        updates.append({
            "type": "process", "key": "recall_memory", "stage": "memory",
            "status": "running", "title": "召回长期记忆",
            "detail": "正在检索历史摘要和用户偏好。",
        })
    elif ev_type == "on_chain_end" and name == "recall_memory":
        recalled = data.get("output", {}).get("recalled_memories", [])
        updates.append({
            "type": "process", "key": "recall_memory", "stage": "memory",
            "status": "done", "title": "长期记忆召回完成",
            "detail": "未命中相关记忆。" if not recalled else f"命中 {len(recalled)} 条相关长期记忆。",
        })

    # ---- 澄清 ----
    elif ev_type == "on_chain_start" and name == "clarify":
        updates.append({
            "type": "process", "key": "clarify", "stage": "clarify",
            "status": "running", "title": "请求澄清",
            "detail": "当前信息不足，正在生成追问。",
        })
    elif ev_type == "on_chain_end" and name == "clarify":
        updates.append({
            "type": "process", "key": "clarify", "stage": "clarify",
            "status": "done", "title": "澄清完成",
            "detail": "已生成追问，等待用户补充信息。",
        })

    # ---- 收尾记忆 ----
    elif ev_type == "on_chain_start" and name == "finalize_memory":
        updates.append({
            "type": "process", "key": "finalize_memory", "stage": "memory",
            "status": "running", "title": "整理会话记忆",
            "detail": "正在更新摘要、任务状态并裁剪消息窗口。",
        })
    elif ev_type == "on_chain_end" and name == "finalize_memory":
        output = data.get("output", {})
        parts = []
        if output.get("conversation_summary"): parts.append("摘要已更新")
        if output.get("task_memory"): parts.append("任务状态已更新")
        if output.get("messages"): parts.append("消息窗口已裁剪")
        updates.append({
            "type": "process", "key": "finalize_memory", "stage": "memory",
            "status": "done", "title": "会话记忆已收口",
            "detail": "，".join(parts) if parts else "本轮无需额外记忆更新。",
        })

    # ---- 专职 Agent 处理（3 个 Agent，无 geo_agent）----
    elif ev_type == "on_chain_start" and name in {"weather_agent", "travel_agent", "general_agent"}:
        updates.append({
            "type": "process", "key": name, "stage": "agent",
            "status": "running", "title": f"{name} 开始处理",
            "detail": "专职 agent 正在读取上下文并准备生成决策。",
        })
    elif ev_type == "on_chain_end" and name in {"weather_agent", "travel_agent", "general_agent"}:
        updates.append({
            "type": "process", "key": name, "stage": "agent",
            "status": "done", "title": f"{name} 本轮完成",
            "detail": "专职 agent 已完成当前一轮输出。",
        })

    # ---- 模型推理 ----
    elif ev_type == "on_chat_model_start":
        tracker["ar"] += 1  # assistant round 计数器
        key = f"assistant_{tracker['ar']}"
        tracker["cur"] = key
        updates.append({
            "type": "process", "key": key, "stage": "assistant",
            "status": "running", "title": f"助手回合 {tracker['ar']}",
            "detail": "模型正在生成下一步决策或回答。",
        })
    elif ev_type == "on_chat_model_end":
        key = tracker.get("cur", "assistant_1")
        updates.append({
            "type": "process", "key": key, "stage": "assistant",
            "status": "done", "title": f"助手回合 {tracker.get('ar', 1)}",
            "detail": "模型本轮输出完成，可能包含工具调用或最终回答。",
        })

    # ---- 工具调用 ----
    elif ev_type == "on_tool_start":
        tracker["tr"] += 1  # tool round 计数器
        tool_key = f"tool_{tracker['tr']}"
        tracker["runs"][run_id] = tool_key
        inp = summarize_value(data.get("input"))
        detail = f"输入: {inp}" if inp else "正在调用外部工具。"
        updates.append({
            "type": "process", "key": tool_key, "stage": "tool",
            "status": "running", "title": f"调用工具: {name}", "detail": detail,
        })
    elif ev_type == "on_tool_end":
        tool_key = tracker["runs"].pop(run_id, f"tool_{tracker['tr']}")
        out = summarize_value(data.get("output"))
        detail = f"输出: {out}" if out else "工具调用完成。"
        updates.append({
            "type": "process", "key": tool_key, "stage": "tool",
            "status": "done", "title": f"工具完成: {name}", "detail": detail,
        })

    # ---- 全图完成 ----
    elif ev_type == "on_chain_end" and name == "LangGraph":
        updates.append({
            "type": "process", "key": "graph_complete", "stage": "graph",
            "status": "done", "title": "本轮完成",
            "detail": "Agent 已完成本轮图执行。",
        })

    return updates


# ============================================================
# 第 4 块：FastAPI 应用生命周期 + 路由
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期钩子。
    启动时：构建持久化图 + 连接会话数据库
    关闭时：清理数据库连接
    """
    app.state.graph = await build_persistent_graph()
    app.state.thread_store = await _connect_store()
    await _init_store(app.state.thread_store)
    try:
        yield
    finally:
        # 关闭 checkpoint 连接
        cp = getattr(app.state.graph, "checkpointer", None)
        conn = getattr(cp, "conn", None)
        if conn is not None:
            await conn.close()
        await app.state.thread_store.close()


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例，注册所有路由。"""
    app = FastAPI(title="Travel Assistant (Demo)", lifespan=lifespan)

    # 开发模式：允许 Vite dev server (localhost:5173) 跨域访问
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载静态文件（CSS / JS / 图片等）
    # 生产模式：npm run build 后产物在 static/dist/ 下
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    # -------- 路由 1：首页 --------
    @app.get("/", response_class=FileResponse)
    async def home():
        return FileResponse(FRONTEND_DIR / "index.html")

    # -------- 路由 2：获取所有会话 --------
    @app.get("/api/threads", response_model=list[ThreadSummary])
    async def get_threads():
        records = await list_threads(app.state.thread_store)
        return [ThreadSummary(**r.to_dict()) for r in records]

    # -------- 路由 3：创建新会话 --------
    @app.post("/api/threads", response_model=ThreadSummary)
    async def post_thread():
        r = await create_thread(app.state.thread_store, next_thread_id())
        return ThreadSummary(**r.to_dict())

    # -------- 路由 4：获取会话详情（含消息历史）--------
    @app.get("/api/threads/{tid}", response_model=ThreadDetail)
    async def get_thread_detail(tid: str):
        r = await get_thread(app.state.thread_store, tid)
        if not r:
            raise HTTPException(404, "thread not found")
        return await load_thread_detail(app.state.graph, r)

    # -------- 路由 5：删除会话 --------
    @app.delete("/api/threads/{tid}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_thread_endpoint(tid: str):
        r = await get_thread(app.state.thread_store, tid)
        if not r:
            raise HTTPException(404, "thread not found")
        await delete_graph_thread_data(app.state.graph, tid)
        await delete_thread(app.state.thread_store, tid)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # -------- 路由 6：发送消息（非流式）--------
    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        tid = req.thread_id or next_thread_id()
        await ensure_thread(app.state.thread_store, tid)
        result = await app.state.graph.ainvoke(
            {"messages": [HumanMessage(content=req.message)]},
            config=make_graph_config(tid),
        )
        reply = extract_last_ai_text(result["messages"])
        await update_thread_after_chat(app.state.thread_store, tid, req.message, reply)
        return ChatResponse(thread_id=tid, reply=reply)

    # -------- 路由 7：发送消息（流式输出）--------
    # 这是前端主要使用的 API
    @app.post("/api/chat/stream")
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        """
        流式聊天端点。

        数据格式：NDJSON（每行一个 JSON 对象）
        Stream 中发送的事件类型：
          thread — 告知前端 thread_id
          process — 时间线状态更新
          assistant_start — 助手开始生成文字
          assistant_delta — 助手文字的增量块
          done — 本轮完成，附带完整消息状态

        流式策略：
          1. 优先用模型 astream_events 的真实流事件
          2. 如果模型没有产生可见流块（工具调用阶段），
             回退为分块发送最终回复
        """
        async def event_stream() -> AsyncIterator[bytes]:
            tid = req.thread_id or next_thread_id()
            await ensure_thread(app.state.thread_store, tid)

            # 1. 告诉前端 thread_id
            yield ndjson_line({"type": "thread", "thread_id": tid})

            yielded_text = False      # 是否有文字被流式输出
            started_assistant = False  # 是否已发送 assistant_start
            tracker = {"ar": 0, "cur": "", "tr": 0, "runs": {}}

            yield ndjson_line({
                "type": "process", "key": "graph_start", "stage": "graph",
                "status": "running", "title": "收到新请求",
                "detail": "LangGraph 正在启动本轮执行。",
            })

            # 2. 启动图执行，监听流事件
            async for event in app.state.graph.astream_events(
                {"messages": [HumanMessage(content=req.message)]},
                config=make_graph_config(tid),
                version="v2",
            ):
                # 发送时间线更新
                for pu in build_process_updates(event, tracker):
                    yield ndjson_line(pu)

                # 提取模型生成的文本块
                if event.get("event") != "on_chat_model_stream":
                    continue
                text = extract_stream_text(event.get("data", {}).get("chunk"))
                if not text:
                    continue
                if not started_assistant:
                    yield ndjson_line({"type": "assistant_start"})
                    started_assistant = True
                yielded_text = True
                yield ndjson_line({"type": "assistant_delta", "content": text})

            # 3. 获取最终回复
            detail_before = await load_thread_detail(
                app.state.graph,
                await ensure_thread(app.state.thread_store, tid),
            )
            reply = ""
            for m in reversed(detail_before.messages):
                if m.role == "assistant":
                    reply = m.content
                    break

            # 更新会话元数据
            record = await update_thread_after_chat(
                app.state.thread_store, tid, req.message, reply,
            )
            detail = await load_thread_detail(app.state.graph, record)

            # 4. 如果流式没有产生文字（比如纯工具调用场景），
            #    回退为分块发送最终回复
            if not started_assistant:
                yield ndjson_line({"type": "assistant_start"})
            if not yielded_text:
                for chunk in chunk_text(reply):
                    yield ndjson_line({"type": "assistant_delta", "content": chunk})
                    await asyncio.sleep(0.02)  # 小延迟让前端有"打字效果"

            # 5. 发送完成事件（包含完整的消息列表）
            yield ndjson_line({
                "type": "done",
                "thread": detail.thread.model_dump(),
                "messages": [m.model_dump() for m in detail.messages],
            })

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    return app


# 模块级 app 实例：uvicorn app:app 指向这里
app = create_app()
