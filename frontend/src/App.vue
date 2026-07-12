<!-- ================================================================
 App.vue — 根组件

 职责：
   1. 管理全局状态（会话列表、消息、时间线事件）
   2. 处理所有 API 调用和 NDJSON 流式消费
   3. 通过 provide() 把状态和方法下发给子组件

 组件树：
   App
   ├── SessionSidebar  — 左侧会话列表
   ├── ChatPanel       — 中间对话区（消息 + 输入框）
   └── ProcessTimeline — 右侧 Agent 执行时间线
================================================================ -->

<template>
  <main class="page">
    <SessionSidebar />
    <section class="main">
      <header class="topbar">
        <div>
          <h1>AI 旅行助手</h1>
          <span class="thread-id-label">
            thread_id: <code>{{ state.activeTid || '-' }}</code>
          </span>
        </div>
      </header>
      <div class="workspace">
        <ChatPanel />
        <ProcessTimeline />
      </div>
    </section>
  </main>
</template>

<script setup>
import { reactive, provide, watch, onMounted } from 'vue'
import SessionSidebar from './components/SessionSidebar.vue'
import ChatPanel from './components/ChatPanel.vue'
import ProcessTimeline from './components/ProcessTimeline.vue'

// ============================================================
// 全局状态（reactive 包裹后，任何组件修改它都会触发 UI 更新）
// ============================================================
const state = reactive({
  threads: [],           // 会话列表
  messages: [],          // 当前会话的消息
  events: [],            // 时间线事件
  activeTid: localStorage.getItem('ai-travel-active-thread') || '',
  sending: false,        // 是否正在发送
  timelineHint: '等待执行...',
})

// 存 localStorage
watch(() => state.activeTid, (val) => {
  if (val) localStorage.setItem('ai-travel-active-thread', val)
  else localStorage.removeItem('ai-travel-active-thread')
})

// ============================================================
// 工具函数
// ============================================================

async function fetchJson(url, opts = {}) {
  const r = await fetch(url, opts)
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

function upsertEvent(ev) {
  const idx = state.events.findIndex(e => e.key === ev.key)
  if (idx >= 0) state.events[idx] = { ...state.events[idx], ...ev }
  else state.events.push(ev)
}

// ============================================================
// 会话操作
// ============================================================

async function loadThreads() {
  state.threads = await fetchJson('/api/threads')
  if (!state.threads.length) {
    state.activeTid = ''
    state.messages = []
    state.events = []
    return
  }
  const found = state.threads.find(t => t.thread_id === state.activeTid)
  await selectThread(found ? state.activeTid : state.threads[0].thread_id, false)
}

async function selectThread(tid, refresh = true) {
  const detail = await fetchJson(`/api/threads/${tid}`)
  state.activeTid = detail.thread.thread_id
  state.messages = detail.messages
  state.events = []
  state.timelineHint = '切换会话后等待新消息...'
  if (refresh) state.threads = await fetchJson('/api/threads')
}

async function createThread() {
  const t = await fetchJson('/api/threads', { method: 'POST' })
  state.threads = await fetchJson('/api/threads')
  state.activeTid = t.thread_id
  state.messages = []
  state.events = []
  state.timelineHint = '新会话已创建，发送消息开始'
}

// ============================================================
// 发送消息 + 流式消费 NDJSON
// ============================================================

async function sendMessage(text) {
  if (!text) return
  if (!state.activeTid) await createThread()

  // 立即显示用户消息
  state.messages = [...state.messages, { role: 'user', content: text }]
  state.events = []
  state.timelineHint = 'Agent 正在处理...'
  state.sending = true

  try {
    const res = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, thread_id: state.activeTid }),
    })

    const reader = res.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buf = ''
    let startedAssistant = false

    while (true) {
      const { value, done } = await reader.read()
      if (done && !buf.trim()) break

      buf += decoder.decode(value || new Uint8Array(), { stream: !done })
      const lines = buf.split('\n')
      buf = lines.pop() || ''

      for (const line of lines) {
        if (!line.trim()) continue
        const ev = JSON.parse(line)

        if (ev.type === 'thread') state.activeTid = ev.thread_id
        if (ev.type === 'process') upsertEvent(ev)

        if (ev.type === 'assistant_start' && !startedAssistant) {
          state.messages = [...state.messages, { role: 'assistant', content: '' }]
          startedAssistant = true
        }

        if (ev.type === 'assistant_delta') {
          const msgs = [...state.messages]
          const last = msgs[msgs.length - 1]
          if (!last || last.role !== 'assistant') {
            msgs.push({ role: 'assistant', content: ev.content })
          } else {
            msgs[msgs.length - 1] = { ...last, content: last.content + ev.content }
          }
          state.messages = msgs
        }

        if (ev.type === 'done') {
          state.threads = await fetchJson('/api/threads')
          state.messages = ev.messages
        }
      }
    }
  } catch (err) {
    upsertEvent({ key: 'error', title: '请求失败', detail: `${err}`, status: 'done' })
    state.messages = [...state.messages, { role: 'assistant', content: `请求失败: ${err}` }]
  } finally {
    state.sending = false
  }
}

// ============================================================
// 通过 provide 下发给所有子组件
// ============================================================
provide('state', state)
provide('loadThreads', loadThreads)
provide('selectThread', selectThread)
provide('createThread', createThread)
provide('sendMessage', sendMessage)

onMounted(() => loadThreads())
</script>
