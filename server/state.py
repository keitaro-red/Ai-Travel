"""
=========================================================================
 state.py — 整个项目的数据结构定义
=========================================================================

这个文件只定义"数据长什么样子"，不写业务逻辑。

类比：
  - State = "共享黑板"——图中每个节点都能读、写这块黑板
  - QueryContext = "当前问题的分析报告"——用户说了什么、想去哪
  - TaskMemory = "便签纸"——记录对话进行到哪了、用户偏好

重要概念：TypedDict 是 Python 给字典加上"列名"和"类型"的语法。
它不像 class 那么重，但 IDE 可以自动提示字段名。
"""

from __future__ import annotations
from typing import Literal, TypedDict

# MessagesState 是 LangGraph 提供的"标准状态"，里面带了一个 messages 列表
# 用于存储聊天历史（UserMessage, AIMessage, ToolMessage 等）
from langgraph.graph import MessagesState


# ============================================================
# TravelIntent：用户意图的精确类型
# Literal 表示这个变量的值只能是下面几个字符串之一
# Python 不会报错但 IDE 和静态检查会拦住错误赋值
# ============================================================

# 原项目有 "geocode"（地理编码），这里删掉了
TravelIntent = Literal["weather", "place_search", "travel", "general"]


# ============================================================
# SpecialistAgent：专职 Agent 的名称
# 每个 Agent 是一个"角色工种"，只负责一类任务
# ============================================================
SpecialistAgent = Literal["weather_agent", "travel_agent", "general_agent"]


# ============================================================
# QueryContext：查询上下文
#
# 当用户说了一句话，analyze_query 节点会把它从"自然语言"
# 转换成"结构化数据"存在这个类型里。
# 后续节点（如 select_agent）就基于这个结构化数据做决策。
#
# total=False 表示所有字段都是可选的
# 不是每次都能提取到时间和地点的
# ============================================================
class QueryContext(TypedDict, total=False):
    raw_user_input: str              # 用户的原始输入，一字不差
    intent: TravelIntent              # 分析出的意图（天气/旅行/通用）
    location_text: str                # 从输入中提取的地点片段（如"北京"）
    normalized_city: str              # 标准化后的城市名（"北京市" → "北京"）
    time_text: str                    # 时间线索（"今天""明天""这周末"）
    needs_clarification: bool         # True = 信息不足，需要追问用户
    clarification_reason: str         # 需要追问的原因（给用户看的）
    suggested_tool: str               # 建议先用哪个工具收集信息
    candidate_locations: list[str]    # 候选地点列表（有歧义时列出几个选项）


# ============================================================
# TaskMemory：任务记忆（中期记忆）
#
# 一轮对话结束后保存到这里，下轮对话开始后读取。
# 这样即使过了几轮，系统还记得"用户在规划去杭州"、"预算3000"等。
#
# 存储方式：每一轮结束时由 finalize_memory 节点写入，
# 下一轮开始时由 recall_memory 节点读取。
# ============================================================
class TaskMemory(TypedDict, total=False):
    current_goal: str                 # 当前任务目标（如"规划杭州三日游"）
    latest_intent: TravelIntent       # 上一轮检测到的意图
    latest_user_request: str          # 用户最新的一条消息
    latest_assistant_reply: str       # 助手最新的一条回复
    confirmed_city: str               # 用户确认过的城市（避免反复问）
    recent_cities: list[str]          # 最近几轮提到的城市列表
    budget_text: str                  # 用户提到的预算（"预算3000"）
    trip_days: str                    # 旅行天数（"3天"）
    latest_time_text: str             # 最新的时间线索
    last_active_agent: SpecialistAgent # 上一轮是哪个 Agent 在处理
    user_preferences: list[str]       # 用户的偏好列表（"喜欢自然风光"）


# ============================================================
# TravelAssistantState：多 Agent 图的"共享黑板"
#
# 这是 LangGraph StateGraph 的核心——所有节点共享的对象。
# 每个节点都可以读取和修改这些字段。
#
# MessagesState 是基类，自带了 messages 字段（聊天消息列表）。
# 我们在这里扩展了更多字段给业务逻辑使用。
# ============================================================
class TravelAssistantState(MessagesState, total=False):
    query_context: QueryContext            # 当前查询的分析结果
    active_agent: SpecialistAgent          # 当前由哪个专职 Agent 在干活
    agent_selection_reason: str            # 为什么选这个 Agent（日志和显示用）
    conversation_summary: str              # 压缩后的历史摘要（消息太多时用）
    task_memory: TaskMemory                # 当前会话的任务记忆
    recalled_memories: list[str]           # 从长期记忆中找到的相关内容
    memory_scope: str                      # 记忆的作用域（通常 = thread_id）
