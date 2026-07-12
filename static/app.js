/**
 * =================================================================
 *  app.js — Vue 3 前端（Composition API）
 * =================================================================
 *
 * 相比原版 vanilla JS 的改进：
 *   - 状态用 Vue reactive 管理 → 改了数据 UI 自动更新，不需要手写 render 函数
 *   - 列表用 v-for 渲染 → 不需要手动 innerHTML + createElement
 *   - 条件渲染用 v-if → 不需要 if/else 分支手动写 DOM
 *   - 输入用 v-model 双向绑定 → 不需要手动 .value 读写
 */

const { createApp, ref, reactive, nextTick, watch } = Vue;

createApp({
  setup() {
    // ============================================================
    // 响应式状态（Vue 会自动追踪这些变量，改了 UI 就更新）
    // ============================================================

    const threads = ref([]);          // 会话列表
    const messages = ref([]);         // 当前会话消息
    const events = ref([]);           // 时间线事件
    const activeTid = ref(localStorage.getItem("ai-travel-active-thread") || "");
    const inputText = ref("");        // 输入框内容（v-model 双向绑定）
    const sending = ref(false);       // 是否正在发送中
    const timelineHint = ref("等待执行...");

    // DOM 引用（用于自动滚动）
    const chatBox = ref(null);
    const timelineBox = ref(null);

    // ============================================================
    // 副作用：activeTid 变化时自动写 localStorage
    // ============================================================
    watch(activeTid, (val) => {
      if (val) localStorage.setItem("ai-travel-active-thread", val);
      else localStorage.removeItem("ai-travel-active-thread");
    });

    // ============================================================
    // 工具函数
    // ============================================================

    async function fetchJson(url, opts = {}) {
      const r = await fetch(url, opts);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    }

    function scrollChat() {
      nextTick(() => {
        if (chatBox.value) chatBox.value.scrollTop = chatBox.value.scrollHeight;
      });
    }

    function scrollTimeline() {
      nextTick(() => {
        if (timelineBox.value) timelineBox.value.scrollTop = timelineBox.value.scrollHeight;
      });
    }

    /** 插入或更新一条时间线事件（相同 key 则合并） */
    function upsertEvent(ev) {
      const item = { key: ev.key, title: ev.title, detail: ev.detail, status: ev.status };
      const idx = events.value.findIndex((e) => e.key === ev.key);
      if (idx >= 0) {
        events.value[idx] = { ...events.value[idx], ...item };
      } else {
        events.value.push(item);
      }
      scrollTimeline();
    }

    // ============================================================
    // 会话操作
    // ============================================================

    async function loadThreads() {
      threads.value = await fetchJson("/api/threads");
      if (!threads.value.length) {
        activeTid.value = "";
        messages.value = [];
        events.value = [];
        return;
      }
      const found = threads.value.find((t) => t.thread_id === activeTid.value);
      await selectThread(found ? activeTid.value : threads.value[0].thread_id, false);
    }

    async function selectThread(tid, refresh = true) {
      const detail = await fetchJson(`/api/threads/${tid}`);
      activeTid.value = detail.thread.thread_id;
      messages.value = detail.messages;
      events.value = [];
      timelineHint.value = "切换会话后等待新消息...";
      if (refresh) {
        threads.value = await fetchJson("/api/threads");
      }
    }

    async function createThread() {
      const t = await fetchJson("/api/threads", { method: "POST" });
      threads.value = await fetchJson("/api/threads");
      activeTid.value = t.thread_id;
      messages.value = [];
      events.value = [];
      timelineHint.value = "新会话已创建，发送消息开始";
    }

    // ============================================================
    // 发送消息 + 流式消费 NDJSON（核心逻辑）
    // ============================================================

    async function sendMessage() {
      const text = inputText.value.trim();
      if (!text || sending.value) return;

      if (!activeTid.value) await createThread();

      // 立即显示用户消息
      messages.value = [...messages.value, { role: "user", content: text }];
      events.value = [];
      timelineHint.value = "Agent 正在处理...";
      inputText.value = "";
      sending.value = true;
      scrollChat();

      try {
        const res = await fetch("/api/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, thread_id: activeTid.value }),
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buf = "";
        let startedAssistant = false;

        while (true) {
          const { value, done } = await reader.read();
          if (done && !buf.trim()) break;

          buf += decoder.decode(value || new Uint8Array(), { stream: !done });
          const lines = buf.split("\n");
          buf = lines.pop() || "";

          for (const line of lines) {
            if (!line.trim()) continue;
            const ev = JSON.parse(line);

            if (ev.type === "thread") {
              activeTid.value = ev.thread_id;
            }
            if (ev.type === "process") {
              upsertEvent(ev);
            }
            if (ev.type === "assistant_start" && !startedAssistant) {
              messages.value = [...messages.value, { role: "assistant", content: "" }];
              startedAssistant = true;
              scrollChat();
            }
            if (ev.type === "assistant_delta") {
              // 追加文本到最新一条 assistant 消息
              const msgs = [...messages.value];
              const last = msgs[msgs.length - 1];
              if (!last || last.role !== "assistant") {
                msgs.push({ role: "assistant", content: ev.content });
              } else {
                msgs[msgs.length - 1] = { ...last, content: last.content + ev.content };
              }
              messages.value = msgs;
              scrollChat();
            }
            if (ev.type === "done") {
              threads.value = await fetchJson("/api/threads");
              messages.value = ev.messages;
            }
          }
        }
      } catch (err) {
        upsertEvent({ key: "error", title: "请求失败", detail: `${err}`, status: "done" });
        messages.value = [...messages.value, { role: "assistant", content: `请求失败: ${err}` }];
      } finally {
        sending.value = false;
        nextTick(() => document.querySelector("textarea")?.focus());
      }
    }

    // ============================================================
    // 初始化
    // ============================================================
    loadThreads();

    // ============================================================
    // 暴露给模板
    // ============================================================
    return {
      threads, messages, events, activeTid, inputText, sending, timelineHint,
      chatBox, timelineBox,
      loadThreads, selectThread, createThread, sendMessage,
    };
  },
}).mount("#app");
