"""
=========================================================================
 mcp_tools.py — MCP 工具发现和加载
=========================================================================

这个文件负责"找到"高德 MCP 服务提供了哪些工具。

重要区别（初学者最容易混淆的点）：
  这个文件是"发现"工具，不是"执行"工具。
  执行是在运行时由 LangGraph 的 ToolNode 自动完成的。

类比：
  mcp_server.py = 商店里的货架，上面摆着工具
  mcp_tools.py = 去商店拍照记下有什么商品
  ToolNode = 顾客真正付钱买商品（调用工具）

加载流程：
  1. 构建 MCP 服务启动配置（build_amap_server_config）
  2. 启动 MCP 服务子进程（mcp_server.py）
  3. 问它："你有什么工具？"
  4. 它回答："我有 weather 和 input_tips"
  5. 把工具列表缓存起来，供图编译时使用

关键设计：
  @lru_cache(maxsize=1) — 只加载一次，后续复用。
  因为启动 MCP 子进程很慢（约 0.5-1 秒），频繁启动没意义。
"""

from __future__ import annotations

import asyncio
import os
import sys
from queue import Queue
from functools import lru_cache
from threading import Thread

from langchain_mcp_adapters.client import MultiServerMCPClient

from server.config import AMAP_MCP_MODULE


def build_amap_server_config() -> dict:
    """
    构建 MCP 服务启动配置。

    这个配置告诉 langchain-mcp-adapters 库：
      1. 运行什么命令 → python -m mcp_server --transport stdio
      2. 传什么环境变量 → API Key 和 PYTHONPATH
      3. 用什么传输方式 → stdio（标准输入输出）

    PYTHONPATH 必须传给子进程，否则子进程找不到 mcp_server 模块。
    """
    env = {
        key: value
        for key in ("AMAP_API_KEY", "GAODE_API_KEY", "PYTHONPATH")
        if (value := os.getenv(key))
    }
    return {
        "transport": "stdio",                    # 通过标准输入输出通信
        "command": sys.executable,                # 使用当前 Python 解释器
        "args": ["-m", AMAP_MCP_MODULE, "--transport", "stdio"],
        "env": env,                               # 传给子进程的环境变量
    }


async def _load_amap_tools_async():
    """异步加载高德 MCP 工具。

    创建一个 MCP 客户端，连接 mcp_server 子进程，
    获取它声明的工具列表。

    这一步不执行任何工具，只是"发现"。
    """
    client = MultiServerMCPClient({"amap": build_amap_server_config()})
    return await client.get_tools()


async def load_amap_tools():
    """公开的异步加载函数，把工具列表转成 tuple。"""
    return tuple(await _load_amap_tools_async())


def _load_amap_tools_in_thread():
    """
    在线程中加载 MCP 工具。

    为什么需要这个？
    当事件循环已经在运行时（比如 uvicorn 启动后），
    不能直接调用 asyncio.run()——会报错"事件循环已在运行"。

    解决方案：创建一个新线程，在新线程里启动新的事件循环。
    用 Queue 在线程间传递结果。
    """
    result_queue: Queue = Queue(maxsize=1)

    def runner():
        try:
            result_queue.put((True, asyncio.run(load_amap_tools())))
        except Exception as exc:
            result_queue.put((False, exc))

    thread = Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    ok, payload = result_queue.get()
    if ok:
        return payload
    raise payload


@lru_cache(maxsize=1)
def get_amap_tools():
    """
    获取 MCP 工具列表（带缓存）。

    @lru_cache 确保：
      第一次调用 → 启动子进程、发现工具、返回结果
      之后调用 → 直接返回缓存结果

    自动选择加载方式：
      如果当前没有事件循环 → asyncio.run()
      如果当前已有事件循环 → 创建新线程
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return tuple(asyncio.run(load_amap_tools()))
    return tuple(_load_amap_tools_in_thread())
