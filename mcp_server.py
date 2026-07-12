"""
=========================================================================
 mcp_server.py — 高德地图 MCP 服务端
=========================================================================

MCP = Model Context Protocol，是 AI 模型和工具之间的"USB 接口"。

这个文件是一个独立的 MCP 服务进程，通过 FastMCP 框架提供两个工具：
  1. weather — 天气查询
  2. input_tips — 地点输入提示

它怎么工作：
  1. 作为一个子进程被 mcp_tools.py 启动（通过 stdio 通信）
  2. 启动后向客户端报告："我有两个工具，定义如下..."
  3. 收到调用请求时，调用高德 REST API
  4. 把结果 JSON 返回给客户端

启动方式：
  被 mcp_tools.py 自动启动（推荐）
  手动启动：python mcp_server.py --transport stdio
  HTTP 模式：python mcp_server.py --transport streamable-http
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

# 高德地图 REST API 基础地址
AMAP_BASE_URL = "https://restapi.amap.com/v3"

# 创建 FastMCP 实例（FastMCP 是 MCP Python SDK 的简化封装）
mcp = FastMCP("Amap MCP Server", json_response=True)


def _get_amap_api_key() -> str:
    """从环境变量读取高德 API Key。

    支持两个变量名，兼容不同用户的命名习惯：
      - AMAP_API_KEY（主要）
      - GAODE_API_KEY（备用）
    """
    api_key = os.getenv("AMAP_API_KEY") or os.getenv("GAODE_API_KEY")
    if not api_key:
        raise RuntimeError("未找到高德 API Key。请设置 AMAP_API_KEY 或 GAODE_API_KEY 环境变量。")
    return api_key


def _request_amap(path: str, params: dict[str, Any], http_get=requests.get) -> dict[str, Any]:
    """
    封装高德 REST API 的 HTTP 请求。

    参数：
      path: API 路径，如 "weather/weatherInfo"
      params: 请求参数（不含 API Key）
      http_get: 可注入的 HTTP 函数（方便单元测试）

    返回统一格式：
      {"status": ..., "info": ..., "data": ...}
    """
    request_params = {"key": _get_amap_api_key(), **params}
    response = http_get(f"{AMAP_BASE_URL}/{path}", params=request_params, timeout=15)
    response.raise_for_status()
    data = response.json()
    return {
        "status": data.get("status"),   # "1" 表示成功
        "info": data.get("info"),        # 说明信息
        "infocode": data.get("infocode"),
        "data": data,                   # 原始响应数据
    }


# ============================================================
# 工具 1：天气查询
#
# 参数：
#   city: 城市名（"北京"）、adcode（"110000"）或 citycode（"010"）
#   extensions: "base" = 实时天气，"all" = 3 天预报
#
# 返回：
#   lives[] — 实时天气数组
#   forecasts[] — 预报数组（仅 extensions=all 时有）
# ============================================================
@mcp.tool()
def weather(city: str, extensions: str = "base") -> dict[str, Any]:
    """查询城市天气。city 可以是城市名、adcode 或 citycode。"""
    result = _request_amap("weather/weatherInfo", {"city": city, "extensions": extensions})
    return {
        "query": {"city": city, "extensions": extensions},
        "lives": result["data"].get("lives", []),
        "forecasts": result["data"].get("forecasts", []),
        "status": result["status"],
        "info": result["info"],
    }


# ============================================================
# 工具 2：地点输入提示（搜索建议）
#
# 类似百度/高德地图的搜索框联想功能。
# 用户输入"望京"，会提示"望京SOHO""望京地铁站"等。
#
# 参数：
#   keywords: 关键字，如"天安门"
#   city: 限定城市范围
#   city_limit: 是否只在指定城市内搜索
# ============================================================
@mcp.tool()
def input_tips(keywords: str, city: str | None = None, city_limit: bool = False) -> dict[str, Any]:
    """根据关键字获取输入提示，适合地点补全。"""
    params: dict[str, Any] = {"keywords": keywords, "citylimit": "true" if city_limit else "false"}
    if city:
        params["city"] = city
    result = _request_amap("assistant/inputtips", params)
    return {
        "query": {"keywords": keywords, "city": city, "city_limit": city_limit},
        "tips": result["data"].get("tips", []),
        "status": result["status"],
        "info": result["info"],
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    --transport 选择通信方式：
      stdio（默认）：通过标准输入输出通信，适合作为子进程运行
      streamable-http：通过 HTTP 通信，适合独立部署
    """
    parser = argparse.ArgumentParser(description="Run the Amap MCP server.")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio",
                        help="MCP transport type.")
    return parser.parse_args()


def main() -> None:
    """启动 MCP 服务。"""
    args = parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
