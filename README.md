# AI-Travel —— 智能旅行助手

基于 **LangGraph 多智能体架构** 的智能旅行助手。集成 DeepSeek 大模型和高德地图 API，支持天气查询、旅行规划和自然语言对话。前端使用 Vue 3，后端使用 FastAPI，通过 **MCP（Model Context Protocol）** 协议连接外部工具。

## 功能特性

- 🌤️ **天气查询** — 查询任意城市的实时天气、温度、穿衣建议
- 🗺️ **旅行规划** — 多智能体协作，整合天气和地点信息做旅行规划
- 🧠 **长期记忆** — 跨会话记住用户偏好（预算、喜好、历史目的地）
- 💬 **自然对话** — 基于 DeepSeek 大模型，中文对话流畅自然
- ⚡ **实时流式输出** — NDJSON 流式传输，回答逐字显示
- 📋 **执行时间线** — 可视化展示 Agent 的每一步思考和工具调用

## 系统架构

```
用户浏览器 (Vue 3)
       │
       ▼
  FastAPI (app.py)          ← HTTP + NDJSON 流式输出
       │
       ▼
  LangGraph StateGraph      ← 多智能体编排引擎
       │
       ├── analyze_query    ← 意图分析（关键词匹配）
       ├── recall_memory    ← 召回长期记忆
       ├── select_agent     ← 路由到专职 Agent
       │
       ├── weather_agent    ← 天气 Agent（调用高德天气 API）
       ├── travel_agent     ← 旅行规划 Agent
       └── general_agent    ← 通用对话 Agent
              │
              ▼
       MCP Server (mcp_server.py)  ← 高德地图工具
              │
              ▼
         高德地图 API
```

**核心循环**：用户输入 → 意图分析 → 召回记忆 → 选择 Agent → Agent 工作（可能需要工具）→ 收尾保存记忆

## 前置要求

| 依赖 | 版本要求 | 说明 |
|---|---|---|
| Python | 3.10 或更高 | 推荐使用 Miniconda 管理 |
| Node.js | 18 或更高 | 前端开发需要（仅使用后端则不需要） |
| DeepSeek API Key | — | [免费获取](https://platform.deepseek.com/api_keys) |
| 阿里云 DashScope API Key | — | [免费获取](https://dashscope.console.aliyun.com/) |
| 高德地图 API Key | — | [免费获取](https://console.amap.com/)（选"Web 服务"类型） |

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/keitaro-red/Ai-Travel.git
cd Ai-Travel
```

### 2. 创建 conda 环境

```bash
# 创建名为 ai-travel 的虚拟环境（Python 3.11）
conda create -n ai-travel python=3.11 -y

# 激活环境
conda activate ai-travel
```

### 3. 设置 API Key

```bash
# 用 conda 设置环境变量（永久生效，仅在此环境中）
conda env config vars set DEEPSEEK_API_KEY="你的DeepSeek-Key"
conda env config vars set DASHSCOPE_API_KEY="你的DashScope-Key"
conda env config vars set AMAP_API_KEY="你的高德Key"

# 重新激活环境使环境变量生效
conda deactivate
conda activate ai-travel
```

> 也可以创建 `.env` 文件（复制 `.env.example` 并填入真实 Key），然后用 `pip install python-dotenv` 加载。

### 4. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 5. 启动后端

```bash
uvicorn app:app --reload
```

看到以下输出表示启动成功：
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### 6. 启动前端（新终端窗口）

```bash
# 激活同一个 conda 环境
conda activate ai-travel

# 进入前端目录
cd frontend

# 安装 npm 依赖（仅第一次）
npm install

# 启动开发服务器
npm run dev
```

浏览器打开 **http://localhost:5173**，看到聊天界面即可使用。

> **生产模式**：运行 `npm run build` 构建前端，然后直接访问 http://localhost:8000（后端会自动托管构建产物）。

## 项目文件说明

| 文件 / 文件夹 | 作用 |
|---|---|
| `config.py` | 所有配置（模型名称、路径、记忆参数、System Prompt） |
| `state.py` | 数据结构定义——LangGraph 的"共享黑板"上有哪些字段 |
| `llm.py` | AI 模型接入（DeepSeek 对话 + 阿里云 Embedding） |
| `memory.py` | 三层记忆系统（短期窗口 / 中期任务状态 / 长期 SQLite 向量搜索） |
| `nodes.py` | 图节点实现（意图分析、追问澄清、Agent 路由） |
| `graph.py` | LangGraph 工作流图的"蓝图"——节点怎么连、箭头怎么画 |
| `mcp_server.py` | 高德 MCP 工具服务端（天气查询、输入提示） |
| `mcp_tools.py` | MCP 客户端——发现并加载高德工具 |
| `app.py` | FastAPI 后端——API 路由、NDJSON 流式输出、静态文件托管 |
| `frontend/` | Vue 3 + Vite 前端源码 |
| `static/` | 旧版原生 JS 前端（仅供参考，已不再维护） |
| `.env.example` | API Key 配置模板 |
| `requirements.txt` | Python 依赖列表 |

## API Key 获取指南

### DeepSeek

1. 打开 [platform.deepseek.com](https://platform.deepseek.com)
2. 注册/登录 → 点击左侧「API Keys」→「创建 API Key」
3. 复制 Key（格式：`sk-xxxx`）
4. 新用户有免费额度

### 阿里云 DashScope

1. 打开 [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com/)
2. 登录阿里云账号 → 开通 DashScope 服务
3. 右上角「API Key 管理」→ 复制 Key
4. 文本嵌入模型有免费额度

### 高德地图

1. 打开 [console.amap.com](https://console.amap.com/)
2. 注册/登录 → 「应用管理」→「我的应用」→「创建新应用」
3. 添加 Key，服务平台选择 **「Web 服务」**
4. 复制 Key（格式：`xxxx`，注意高德的 Key 不带 `sk-` 前缀）

## 常见问题

### Q: 启动报错 "未检测到 DEEPSEEK_API_KEY"

**A:** API Key 没有正确设置。检查：
```bash
conda activate ai-travel
conda env config vars list   # 看看有没有列出三个 Key
```
如果没有，重新执行 `conda env config vars set`。设置后需要 `conda deactivate && conda activate ai-travel` 才能生效。

### Q: 端口 8000 被占用

**A:** 换一个端口启动后端：
```bash
uvicorn app:app --reload --port 8080
```
然后修改 `frontend/vite.config.js` 中的 proxy 目标为 `http://localhost:8080`。

### Q: npm install 太慢或失败

**A:** 使用国内镜像：
```bash
npm install --registry https://registry.npmmirror.com
```

### Q: 前端页面空白

**A:** 确保后端也启动了（前端通过 Vite proxy 把 `/api` 请求转发到后端 8000 端口）。打开浏览器开发者工具（F12）→ Console 查看错误信息。

### Q: ModuleNotFoundError: No module named 'mcp_server'

**A:** 确保在 `Ai-Travel/` 目录下运行 `uvicorn app:app`，不要 cd 到其他目录。

### Q: MCP 工具连接失败

**A:** 检查 `AMAP_API_KEY` 是否正确设置，高德的 Key 是纯字母数字（不像 DeepSeek 的 Key 以 `sk-` 开头）。

## 技术栈

| 层级 | 技术 |
|---|---|
| **AI 模型** | DeepSeek (ChatOpenAI 兼容接口) |
| **嵌入模型** | 阿里云 DashScope text-embedding-v2 |
| **多智能体框架** | LangGraph (StateGraph + Checkpoint) |
| **MCP 协议** | mcp + langchain-mcp-adapters |
| **后端** | FastAPI + uvicorn |
| **数据持久化** | SQLite (aiosqlite) |
| **前端** | Vue 3 (Composition API) + Vite 5 |
| **地图服务** | 高德地图 Web 服务 API |
