"""
=========================================================================
 llm.py — 大语言模型接入层
=========================================================================

这个文件负责"连接 AI 模型"。
因为 DeepSeek 兼容 OpenAI 的 API 格式，所以用 langchain-openai 的 ChatOpenAI 类，
只需要改 base_url 和 model 名字就能对接。

对比原项目（Qwen/Tongyi）：
  原项目用 langchain-community 里的 ChatTongyi
  Demo 换成 langchain-openai 的 ChatOpenAI + base_url 指向 DeepSeek

设计：
  @lru_cache 缓存模型实例，避免重复创建（创建模型涉及网络连接，比较耗时）
"""

from __future__ import annotations

import os
from functools import lru_cache

# ChatOpenAI 默认连接 OpenAI，但可以通过 base_url 指向任何兼容的 API
from langchain_openai import ChatOpenAI

from config import DEEPSEEK_MODEL, DEEPSEEK_BASE_URL, DASHSCOPE_EMBEDDING_MODEL


def has_api_key() -> bool:
    """
    检查环境变量 DEEPSEEK_API_KEY 是否存在。

    注意：不要去"验证" Key 的有效性（比如真的发一个请求）。
    验证在第一次实际调用时由 API 做。这里只检查"有没有设"。
    """
    return bool(os.getenv("DEEPSEEK_API_KEY"))


@lru_cache(maxsize=1)
def get_model(model_name: str = DEEPSEEK_MODEL) -> ChatOpenAI:
    """
    创建（或从缓存取）一个 DeepSeek 模型实例。

    @lru_cache 的作用：
      第一次调用 → 创建实例并缓存
      之后调用 → 直接返回缓存实例，不重复创建

    maxsize=1 表示只缓存 1 个实例（我们只需要 1 个模型）。

    如果用户没设 API Key，这里就报错退出，
    而不是等到运行时才发现——这叫"快速失败"（Fail Fast）。
    """
    if not has_api_key():
        raise RuntimeError(
            "未检测到 DEEPSEEK_API_KEY 环境变量。\n"
            "请设置：conda env config vars set DEEPSEEK_API_KEY=\"你的Key\" -n 环境名"
        )
    return ChatOpenAI(
        model=model_name,       # "deepseek-chat"
        base_url=DEEPSEEK_BASE_URL,  # "https://api.deepseek.com"
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> "DashScopeEmbedding":
    """
    创建阿里云 DashScope 嵌入模型（用 requests 直接调 API，不依赖 dashscope 包）。

    为什么不用 langchain_community.embeddings.DashScopeEmbeddings？
    因为它依赖 dashscope → aiohttp，在 Windows 上有 SSL 证书问题。
    直接用 requests 调 REST API 更稳定、更轻量。
    """
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError(
            "未检测到 DASHSCOPE_API_KEY 环境变量。\n"
            "请设置：conda env config vars set DASHSCOPE_API_KEY=\"你的Key\" -n 环境名"
        )
    return DashScopeEmbedding(model=DASHSCOPE_EMBEDDING_MODEL)


# ============================================================
# 轻量级 DashScope 嵌入类：用 requests 直接调 REST API
# ============================================================

class DashScopeEmbedding:
    """
    对阿里云 DashScope Text Embedding API 的最小封装。

    API: POST https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding

    不做批量、不做重试——只服务于记忆搜索这种低频调用场景。
    """

    def __init__(self, model: str = "text-embedding-v2"):
        self.model = model
        self.url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        self.api_key = os.getenv("DASHSCOPE_API_KEY")

    def embed_query(self, text: str) -> list[float]:
        """
        把一段文字转成一个浮点数向量。

        HTTP 请求:
          POST /api/v1/services/embeddings/text-embedding/text-embedding
          Authorization: Bearer sk-xxx
          {"model": "text-embedding-v2", "input": {"texts": ["北京天气"]}}

        返回:
          {"output": {"embeddings": [{"embedding": [0.12, -0.34, ...]}]}}
        """
        import requests

        resp = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "input": {"texts": [text]},
            },
            timeout=30,
        )
        data = resp.json()

        # 检查 API 是否正常返回
        if data.get("code") != "" and data.get("code") is not None:
            raise RuntimeError(f"DashScope Embedding API 错误: {data.get('message', data)}")

        # 从深层嵌套中提取向量
        embeddings = data["output"]["embeddings"]
        return embeddings[0]["embedding"]


# ============================================================
# 关于为什么 LLM 和 Embedding 用了不同厂商：
#
#   LLM (DeepSeek)   → ChatOpenAI + base_url=https://api.deepseek.com
#                       兼容 OpenAI Chat Completions API
#
#   Embedding (阿里)  → DashScopeEmbeddings
#                       DashScope 是阿里云的模型平台，和 LLM 分离
#
# 两个 Key 各管各的，互不干扰：
#   DEEPSEEK_API_KEY   → 对话模型
#   DASHSCOPE_API_KEY  → 嵌入模型
#   AMAP_API_KEY       → 高德地图 MCP 工具
# ============================================================
