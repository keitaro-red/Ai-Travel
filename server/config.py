"""
=========================================================================
 config.py — 整个项目的"控制面板"
=========================================================================

所有"可能会改的值"都放这里：
  - API Key 的环境变量名
  - 模型名称和地址
  - 文件存储路径
  - 记忆系统的参数（窗口大小、触发条数等）
  - 各个 Agent 的 System Prompt（角色说明书）

设计原则：
  1. 不要散落到各个源文件里，改配置只改这一个地方
  2. 优先从环境变量读（通过 os.getenv），用户可以在部署时自由覆盖
  3. 有合理的默认值，小白直接跑也能用
"""

from __future__ import annotations

import os


# ============================================================
# DeepSeek 模型配置
# ============================================================

# 模型名称：deepseek-chat 是 DeepSeek 的通用对话模型
# 如果你用 DeepSeek V3，改成 "deepseek-chat"
# 如果你用 DeepSeek R1，改成 "deepseek-reasoner"
DEEPSEEK_MODEL = "deepseek-chat"

# DeepSeek API 的地址
# DeepSeek 兼容 OpenAI 的 API 格式，所以 base_url 指向 DeepSeek 的地址
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 阿里云 DashScope 文本嵌入模型（LLM 用 DeepSeek，Embedding 用阿里）
# text-embedding-v2 是通义千问的嵌入模型，输入文本输出语义向量
DASHSCOPE_EMBEDDING_MODEL = "text-embedding-v2"


# ============================================================
# 高德 MCP 配置
# ============================================================

# MCP Server 的 Python 模块路径
# 注意：当你 cd 到项目目录运行 uvicorn 时，要写 "mcp_server" 而不是带路径前缀
AMAP_MCP_MODULE = "server.mcp_server"


# ============================================================
# 数据持久化路径
#
# 所有数据文件存在 data/ 目录下，自动创建。
# 这个目录已被 .gitignore 忽略，不会提交到仓库。
# 里面会放三种数据：
#   1. checkpoints.sqlite — 对话状态（每轮对话的完整状态快照）
#   2. long_term_memory.sqlite — 长期记忆（用户偏好、历史摘要等）
#   3. thread_store.sqlite — 会话元数据（标题、创建时间等）
# ============================================================
CHECKPOINT_DB_PATH = "data/checkpoints.sqlite"
LONG_TERM_MEMORY_DB_PATH = "data/long_term_memory.sqlite"


# ============================================================
# 记忆系统参数
#
# 这些参数控制"记忆"的行为：
#   SHORT_TERM_WINDOW：保留最近多少条消息（之前的压缩成摘要）
#   SUMMARY_TRIGGER：总消息数超过多少就触发压缩
#   MAX_SUMMARY_CHARS：摘要最多保留多少字符
#   TOP_K：每次从长期记忆中召回多少条
# ============================================================
MEMORY_SHORT_TERM_WINDOW = int(os.getenv("MEMORY_SHORT_TERM_WINDOW", "8"))
MEMORY_SUMMARY_TRIGGER_MESSAGES = int(os.getenv("MEMORY_SUMMARY_TRIGGER_MESSAGES", "12"))
MEMORY_MAX_SUMMARY_CHARS = int(os.getenv("MEMORY_MAX_SUMMARY_CHARS", "1200"))
LONG_TERM_MEMORY_TOP_K = int(os.getenv("LONG_TERM_MEMORY_TOP_K", "3"))


# ============================================================
# System Prompt（系统提示词）= 给 AI 的"岗位说明书"
#
# 每个 AI Agent 在对话开始前都会收到一段 System Prompt，
# 告诉它"你是谁""该做什么""不该做什么"。
#
# 这是影响 AI 行为最重要的配置。改这里比改代码更有效。
# ============================================================

# Travel Agent：所有旅行相关任务的"总入口提示词"
# 不管哪个专职 Agent 被选中，都会先看到这个通用旅行助手提示词
TRAVEL_AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手，负责回答天气、地点、城市出行、景点等问题。

行为规则：
1. 只要问题涉及天气、地点、坐标、行政区、POI、出行前准备，就优先调用高德 MCP 工具。
2. 不要假装知道实时天气或地点信息；没有工具结果时要明确说明。
3. 回答使用中文，先给直接结论，再给简洁解释。
4. 如果用户问题与旅行无关，也可以正常回答，但保持简洁。
5. 如果工具返回的数据不足以支撑结论，要说明不确定性。
""".strip()

# Weather Agent：专职处理天气
# 它的提示词更聚焦，只关注天气，不让它做坐标解析之类的事
WEATHER_AGENT_SYSTEM_PROMPT = """
你的角色是 Weather Agent，只负责天气与出行天气建议。

行为规则：
1. 重点回答天气现状、温度、降水、穿衣和是否适合出门。
2. 如果已有标准城市名，优先直接调用天气工具，不要重复追问。
3. 不要处理坐标解析等非天气主任务，除非这是回答天气结论所必须的辅助信息。
4. 回答应先给结论，再给简洁依据。
""".strip()

# Travel Planner Agent：专职做旅行规划
# 它可以整合天气和地点信息来做规划
TRAVEL_PLANNER_AGENT_SYSTEM_PROMPT = """
你的角色是 Travel Planner Agent，负责旅行规划、城市比较、地点推荐和出行建议。

行为规则：
1. 可以结合天气、地点提示和坐标类工具辅助做规划与比较。
2. 回答优先给建议结论，再给比较理由或简单行程思路。
3. 当需要实时信息时必须依赖工具，不要凭空判断。
4. 如果用户问题本质上只是天气或坐标解析，应沿用当前上下文，但保持规划视角的整合表达。
""".strip()

# General Agent：兜底 Agent
# 当用户的问题不属于以上任何一类时，由它处理（纯聊天模式）
GENERAL_AGENT_SYSTEM_PROMPT = """
你的角色是 General Agent，负责处理不属于天气或旅行规划的普通问答。

行为规则：
1. 优先直接回答，不要无意义调用工具。
2. 回答保持简洁、准确。
3. 如果问题与旅行弱相关，可以给最小必要回答，不要过度扩展。
""".strip()
