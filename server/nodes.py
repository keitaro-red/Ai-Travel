"""
=========================================================================
 nodes.py — 图中所有节点的具体逻辑实现
=========================================================================

这个文件是项目逻辑密度最高的地方，包含了：

  1. create_specialist_node — 创建专职 Agent 的工厂函数
  2. analyze_query — 意图分析（关键词 + 正则）
  3. select_specialist_agent — 根据意图路由到合适的 Agent
  4. clarify_query — 信息不足时生成追问
  5. 各种辅助函数

每个函数对应图中一个"节点"（Node）的具体工作内容。

图中节点的执行顺序：
  START → analyze_query → (clarify | recall_memory) → select_agent
  → weather_agent|travel_agent|general_agent
  → (tools | finalize_memory) → (回到 Agent | END)
"""

from __future__ import annotations

import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, message_chunk_to_message
from langchain_core.runnables import RunnableLambda

from server.config import (
    GENERAL_AGENT_SYSTEM_PROMPT,
    TRAVEL_AGENT_SYSTEM_PROMPT,
    TRAVEL_PLANNER_AGENT_SYSTEM_PROMPT,
    WEATHER_AGENT_SYSTEM_PROMPT,
)
from server.memory import build_memory_context_messages
from server.state import QueryContext, SpecialistAgent, TravelAssistantState


# ============================================================
# 关键字定义区
#
# 这些元组定义"什么词触发什么意图"。
# 本质上是简单的规则匹配——虽然是"笨办法"，但在这个场景下
# 比用大模型做意图分类更快、更省钱、更稳定。
# ============================================================

# 天气相关：只要包含这些词就认为是天气问题
WEATHER_KEYWORDS = ("天气", "气温", "温度", "下雨", "下雪", "冷不冷", "热不热", "适合出门")

# 地点搜索：找附近有什么
PLACE_SEARCH_KEYWORDS = ("附近", "景点", "餐厅", "酒店", "机场", "地铁站", "poi")

# 旅行规划
TRAVEL_KEYWORDS = ("出行", "旅游", "旅行", "行程", "周末去哪", "怎么玩")

# 时间词
TIME_PATTERNS = ("今天", "明天", "后天", "这周末", "周末", "本周", "下周")

# 城市别名表：把用户的非标准说法统一成标准名
# 这个字典同时也是一个"白名单"——只有这些城市能自动识别
CITY_ALIASES = {
    "北京": "北京", "北京市": "北京",
    "上海": "上海", "上海市": "上海",
    "杭州": "杭州", "杭州市": "杭州",
    "广州": "广州", "广州市": "广州",
    "深圳": "深圳", "深圳市": "深圳",
    "南京": "南京", "南京市": "南京",
    "苏州": "苏州", "苏州市": "苏州",
    "成都": "成都", "成都市": "成都",
    "重庆": "重庆", "重庆市": "重庆",
    "武汉": "武汉", "武汉市": "武汉",
    "西安": "西安", "西安市": "西安",
    "天津": "天津", "天津市": "天津",
    "长沙": "长沙", "长沙市": "长沙",
    "青岛": "青岛", "青岛市": "青岛",
    "厦门": "厦门", "厦门市": "厦门",
}

# POI 特征词：如果地点包含这些字，说明是具体地点不是城市名
# POI = Point of Interest（兴趣点），比如"天安门广场"、"望京SOHO"
POI_HINTS = (
    "区", "路", "街", "广场", "大厦", "公园", "景区", "车站", "站",
    "机场", "酒店", "商场", "大学", "医院", "小区", "村", "镇", "乡",
    "山", "湖", "寺", "桥", "塔", "园区", "SOHO", "soho",
)

# Agent 显示名称映射
AGENT_LABELS: dict[SpecialistAgent, str] = {
    "weather_agent": "Weather Agent",
    "travel_agent": "Travel Planner Agent",
    "general_agent": "General Agent",
}

# Agent 提示词映射（每个 Agent 有自己专属的 System Prompt）
AGENT_PROMPTS: dict[SpecialistAgent, str] = {
    "weather_agent": WEATHER_AGENT_SYSTEM_PROMPT,
    "travel_agent": TRAVEL_PLANNER_AGENT_SYSTEM_PROMPT,
    "general_agent": GENERAL_AGENT_SYSTEM_PROMPT,
}


# ============================================================
# run_bound_model：调用 AI 模型（支持流式输出）
#
# 流式输出（stream）：模型是一个字一个字生成回答的。
# 如果模型支持 astream，我们逐块接收再合并。
# 如果不支持，退化为常规 invoke。
# ============================================================

async def run_bound_model(bound_model, messages) -> AIMessage:
    """异步调用 AI 模型，优先使用流式输出。"""
    if hasattr(bound_model, "astream"):
        acc = None
        async for chunk in bound_model.astream(messages):
            acc = chunk if acc is None else acc + chunk
        if acc is not None:
            final = message_chunk_to_message(acc)
            if isinstance(final, AIMessage):
                return final
    # 回退：非流式调用
    if hasattr(bound_model, "ainvoke"):
        resp = await bound_model.ainvoke(messages)
        if isinstance(resp, AIMessage):
            return resp
    return bound_model.invoke(messages)


def run_bound_model_sync(bound_model, messages) -> AIMessage:
    """同步版模型调用（用于 graph.invoke 场景）。"""
    if hasattr(bound_model, "stream"):
        acc = None
        for chunk in bound_model.stream(messages):
            acc = chunk if acc is None else acc + chunk
        if acc is not None:
            final = message_chunk_to_message(acc)
            if isinstance(final, AIMessage):
                return final
    return bound_model.invoke(messages)


# ============================================================
# create_specialist_node：创建专职 Agent 的工厂函数
#
# 这是一个"函数工厂"——不直接做事情，而是"造"节点函数。
# 每个 Agent 节点的核心工作：
#   1. 组装发给模型的消息列表
#   2. 调用模型
#   3. 返回模型的回复
#
# 参数：
#   bound_model：已经绑定了工具集的模型实例
#   agent_name：这个 Agent 的名字（决定用哪个 System Prompt）
# ============================================================

def create_specialist_node(bound_model, agent_name: SpecialistAgent):
    prompt = AGENT_PROMPTS[agent_name]
    label = AGENT_LABELS[agent_name]

    def build_messages(state: TravelAssistantState):
        """组装发给 AI 模型的消息列表。

        顺序很重要（影响了模型的理解）：
          1. 通用旅行助手提示词（总角色）
          2. 专职 Agent 提示词（具体角色）
          3. 路由信息（为什么选这个 Agent）
          4. 查询上下文（意图、地点分析结果）
          5. 记忆信息（历史摘要、任务状态、长期记忆）
          6. 聊天历史（之前的对话）
        """
        ctx = build_query_context_message(state.get("query_context", {}))
        reason = state.get("agent_selection_reason", "")
        route_msg = [f"当前接管角色：{label}。"]
        if reason:
            route_msg.append(f"选择原因：{reason}")

        # 从 checkpoint 恢复的消息可能末尾残留未完成的 tool_calls
        # （上一轮中断或出错导致），直接发给模型会报 400 错误。
        # 这里把末尾所有未收到 ToolMessage 回复的 AIMessage(tool_calls) 剔除。
        raw = list(state["messages"])
        while raw and isinstance(raw[-1], AIMessage) and getattr(raw[-1], "tool_calls", None):
            raw.pop()

        return [
            SystemMessage(content=TRAVEL_AGENT_SYSTEM_PROMPT),
            SystemMessage(content=prompt),
            SystemMessage(content="\n".join(route_msg)),
            SystemMessage(content=ctx),
            *build_memory_context_messages(state),
            *raw,
        ]

    def assistant(state: TravelAssistantState):
        return {"messages": [run_bound_model_sync(bound_model, build_messages(state))]}

    async def assistant_async(state: TravelAssistantState):
        return {"messages": [await run_bound_model(bound_model, build_messages(state))]}

    return RunnableLambda(assistant, afunc=assistant_async, name=agent_name)


# ============================================================
# 节点函数：analyze_query（意图分析）
#
# 用户输入一句话，这个节点把它从"自然语言"转成"结构化数据"。
# 它是整条处理管线的第一步，也是最关键的一步。
# 后面的路由和 Agent 都依赖它的输出。
# ============================================================

def analyze_query(state: TravelAssistantState) -> dict[str, QueryContext]:
    latest = extract_latest_user_text(state)    # 取出用户最新说的话
    ctx = build_query_context(latest, state.get("query_context", {}))  # 分析！
    return {"query_context": ctx}


# ============================================================
# 条件路由函数：route_after_analysis
#
# 分析完成后，决定下一步：
#   - 信息不足（needs_clarification=True）→ 去 clarify 追问
#   - 信息足够 → 去 recall_memory（召回记忆，然后 select_agent）
# ============================================================

def route_after_analysis(state: TravelAssistantState) -> Literal["clarify", "select_agent"]:
    qc = state.get("query_context", {})
    return "clarify" if qc.get("needs_clarification") else "select_agent"


# ============================================================
# 节点函数：select_specialist_agent（选择专职 Agent）
#
# 根据意图类型路由到不同的专职 Agent：
#   weather → weather_agent
#   place_search / travel → travel_agent
#   general → general_agent
# ============================================================

def select_specialist_agent(state: TravelAssistantState) -> dict[str, SpecialistAgent | str]:
    qc = state.get("query_context", {})
    intent = qc.get("intent", "general")
    location = qc.get("normalized_city") or qc.get("location_text", "")
    if intent == "weather":
        return {"active_agent": "weather_agent",
                "agent_selection_reason": f"识别到天气意图，优先交给天气专职 agent。地点线索：{location or '未提供'}。"}
    if intent in {"place_search", "travel"}:
        return {"active_agent": "travel_agent",
                "agent_selection_reason": "识别到旅行规划或地点推荐意图，交给规划 agent。"}
    return {"active_agent": "general_agent",
            "agent_selection_reason": "未命中特定旅行子领域，交给通用 agent。"}


def route_to_specialist(state: TravelAssistantState) -> Literal["weather_agent", "travel_agent", "general_agent"]:
    """条件路由：把流程导向已选中的专职 Agent。"""
    return state.get("active_agent", "general_agent")


# ============================================================
# 节点函数：clarify_query（追问澄清）
#
# 当系统分析后发现信息不足（比如只说了"天气"没提城市），
# 生成一个追问问题让用户补充信息。
# 追问后会去 finalize_memory → END，等用户下一轮输入。
# ============================================================

def clarify_query(state: TravelAssistantState):
    qc = state.get("query_context", {})
    location_text = qc.get("location_text", "")
    reason = qc.get("clarification_reason", "")
    intent = qc.get("intent", "general")

    if intent == "weather" and not location_text:
        content = "你想查询哪个城市或地点的天气？例如北京、上海迪士尼或西湖景区。"
    elif intent == "weather":
        content = (f"你提到的是{location_text}，这更像具体地点而不是标准城市名。"
                   "请补充城市，或直接给出更完整地点，例如北京市朝阳区望京SOHO天气怎么样？")
    else:
        content = "请再具体一点描述你的地点或旅行需求，我再继续帮你处理。"

    if reason:
        content = f"{content}\n当前判断依据：{reason}"
    return {"messages": [AIMessage(content=content)]}


# ============================================================
# 以下是从用户自然语言中提取结构化信息的辅助函数
# ============================================================

def extract_latest_user_text(state: TravelAssistantState) -> str:
    """从消息列表中找到最新一条用户消息的文本。"""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def detect_intent(text: str):
    """通过关键词匹配检测用户意图。"""
    if any(k in text for k in WEATHER_KEYWORDS):
        return "weather"
    if any(k in text for k in PLACE_SEARCH_KEYWORDS):
        return "place_search"
    if any(k in text for k in TRAVEL_KEYWORDS):
        return "travel"
    return "general"


def should_inherit_followup_intent(text: str, prev_ctx: QueryContext, current_intent: str) -> bool:
    """
    判断当前追问是否应该继承上轮的意图。

    场景：用户第一轮说"北京天气"，第二轮只说"那上海呢？"
    这时候"那上海呢？"不包含任何意图关键词，
    但显然应该继承上轮的 weather 意图。

    判断条件：
      1. 当前没有检测到明确意图
      2. 上轮有明确意图
      3. 消息很短（<=8字）或包含追问标记（呢、那、然后等）
    """
    if current_intent != "general":
        return False
    prev = prev_ctx.get("intent", "general")
    if prev not in {"weather", "travel", "place_search"}:
        return False
    if not text:
        return False
    if len(text) <= 8:
        return True
    markers = ("呢", "那", "那边", "那儿", "然后", "还有", "这个呢", "那个呢")
    return any(m in text for m in markers)


def infer_intent(text: str, explicit: str, prev_ctx: QueryContext):
    """
    推断用户意图：优先用显式检测到的，没有则尝试继承上轮。
    """
    if explicit != "general":
        return explicit
    if not should_inherit_followup_intent(text, prev_ctx, explicit):
        return explicit
    return prev_ctx.get("intent", explicit)


def detect_time_text(text: str) -> str:
    """提取时间词。"""
    for t in TIME_PATTERNS:
        if t in text:
            return t
    return ""


def extract_known_city_from_text(text: str) -> str:
    """从文本中找已知城市名（按名称长度从长到短匹配）。"""
    for alias in sorted(CITY_ALIASES, key=len, reverse=True):
        if alias and alias in text:
            return CITY_ALIASES[alias]
    return ""


def extract_followup_location_text(text: str) -> str:
    """
    从简短的追问中提取地点。

    用正则在追问句中找到地点部分。
    例如："上海呢" → "上海"
         "那杭州呢" → "杭州"
    """
    stripped = text.strip()
    for pat in [
        r"^(?:那|那边|那儿)?(?P<location>[一-龥A-Za-z0-9·]{2,20})(?:呢|怎么样|如何|咋样)?$",
        r"^(?P<location>[一-龥A-Za-z0-9·]{2,20})(?:那边|那儿)?(?:呢|怎么样|如何|咋样)?$",
    ]:
        m = re.search(pat, stripped)
        if m:
            return clean_location_text(m.group("location"))
    return ""


def clean_location_text(loc: str) -> str:
    """去掉地点文本中的多余前缀（"帮我看看北京"→"北京"）和标点。"""
    text = loc.strip("，。！？,.!? ")
    for prefix in ("帮我看看", "帮我查查", "查一下", "查下", "看看", "看下"):
        if text.startswith(prefix):
            text = text.removeprefix(prefix)
    if text in TIME_PATTERNS:
        return ""
    return text.strip()


def extract_location_text(text: str, intent: str) -> str:
    """
    根据意图类型提取地点文本。

    天气类：从"北京今天天气"中提取"北京"
    使用正则的命名捕获组 (?P<location>...)。
    """
    if intent == "weather":
        for pat in [
            r"(?:帮我看看|帮我查查|帮我查下|查一下|查下|看看|看下)?(?P<location>[一-龥A-Za-z0-9·]{2,20}?)(?:今天天气|明天天气|后天天气|这周末天气|天气|气温|温度)",
            r"(?P<location>[一-龥A-Za-z0-9·]{2,20}?)(?:适合出门吗|适合旅游吗)",
        ]:
            m = re.search(pat, text)
            if m:
                return clean_location_text(m.group("location"))
    return ""


def normalize_city(loc: str) -> str:
    """标准化城市名：查别名表。不在表中返回空字符串。"""
    return CITY_ALIASES.get(loc, "") if loc else ""


def assess_clarification_need(intent: str, location_text: str, normalized_city: str, text: str) -> tuple[bool, str]:
    """
    判断是否需要追问用户。

    规则：
      - 天气查询，但没有地点 → 需要追问
      - 有标准城市名（匹配了 CITY_ALIASES）→ 不需要
      - 地点是 POI（如"望京SOHO"）而非城市 → 可能需要追问
      - 地点过长但不在城市表中 → 可能存在歧义
    """
    if intent == "weather":
        if not location_text:
            return True, "天气问题缺少明确地点。"
        if normalized_city:
            return False, ""
        if any(h in location_text for h in POI_HINTS):
            return True, "检测到地点更像区县、商圈或 POI，天气接口通常需要标准城市名或更完整地点。"
        if len(location_text) > 6 and "市" not in location_text:
            return True, "地点过长但未识别为标准城市名，存在地点歧义。"
    if intent == "general" and "天气" in text and not location_text:
        return True, "问题提到天气，但未提取到地点。"
    return False, ""


# ============================================================
# build_query_context：从用户文本构建完整的查询上下文
#
# 这是意图分析的"主函数"，按顺序执行：
#   1. 检测关键词意图
#   2. 处理简短追问（继承上轮意图）
#   3. 提取时间和地点
#   4. 标准化城市名
#   5. 判断是否需要追问
#
# 输出是一个 QueryContext 字典，被后续节点使用。
# ============================================================

def build_query_context(user_text: str, prev: QueryContext | None = None) -> QueryContext:
    text = user_text.strip()
    prev_ctx = prev or {}

    # 检测意图
    explicit = detect_intent(text)
    intent = infer_intent(text, explicit, prev_ctx)

    # 提取时间和地点
    time_text = detect_time_text(text)
    location_text = extract_location_text(text, intent)

    # 如果没提取到地点，尝试从简短追问中提取
    if not location_text and should_inherit_followup_intent(text, prev_ctx, intent):
        location_text = extract_followup_location_text(text)

    if not location_text:
        location_text = extract_known_city_from_text(text)

    # 标准化城市名
    normalized = normalize_city(location_text)
    if not normalized:
        known = extract_known_city_from_text(text)
        if known:
            location_text = known
            normalized = known

    # 判断是否需要追问
    needs_clarification, reason = assess_clarification_need(intent, location_text, normalized, text)

    ctx: QueryContext = {
        "raw_user_input": text, "intent": intent, "time_text": time_text,
        "needs_clarification": needs_clarification, "clarification_reason": reason,
    }
    if location_text:
        ctx["location_text"] = location_text
    if normalized:
        ctx["normalized_city"] = normalized
    if needs_clarification and location_text:
        ctx["suggested_tool"] = "input_tips"
    return ctx


def build_query_context_message(qc: QueryContext) -> str:
    """
    把结构化的 QueryContext 转成一段文字描述。
    这段文字会发给 AI 模型，让它知道分析结果。
    """
    lines = [
        "你会收到一个前置解析层生成的查询上下文。",
        "如果上下文里已有标准城市名，天气查询优先使用该城市作为工具参数。",
        "如果上下文提示地点不清晰，不要自行猜测城市。",
    ]
    if qc.get("raw_user_input"):
        lines.append(f"最新用户问题：{qc['raw_user_input']}")
    if qc.get("intent"):
        lines.append(f"解析意图：{qc['intent']}")
    if qc.get("location_text"):
        lines.append(f"原始地点片段：{qc['location_text']}")
    if qc.get("normalized_city"):
        lines.append(f"标准城市名：{qc['normalized_city']}")
    if qc.get("time_text"):
        lines.append(f"时间线索：{qc['time_text']}")
    if qc.get("needs_clarification"):
        lines.append(f"澄清原因：{qc.get('clarification_reason', '')}")
    return "\n".join(lines)
