<!-- ================================================================
 ChatPanel.vue — 中间对话区（消息列表 + 输入框）
================================================================ -->

<template>
  <div class="chat-area">
    <!-- 消息列表 -->
    <div class="messages" ref="chatBox">
      <div v-if="!state.messages.length" class="empty-state">输入问题开始对话</div>
      <div
        v-for="(m, idx) in state.messages" :key="idx"
        class="message" :class="m.role"
      >
        <span class="role">{{ m.role === 'user' ? 'You' : 'Agent' }}</span>
        <div class="content">{{ m.content }}</div>
      </div>
    </div>

    <!-- 输入框 -->
    <div class="composer">
      <textarea
        ref="inputEl"
        v-model="inputText"
        placeholder="输入旅行问题，如：北京今天天气怎么样？"
        @keydown.enter.exact.prevent="onSend"
      ></textarea>
      <div class="composer-foot">
        <span class="hint">Enter 发送 · Shift+Enter 换行</span>
        <button @click="onSend" :disabled="state.sending">发送</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, inject, nextTick } from 'vue'

const state = inject('state')
const sendMessage = inject('sendMessage')

const inputText = ref('')
const inputEl = ref(null) 
const chatBox = ref(null)

function scrollChat() {
  nextTick(() => {
    if (chatBox.value) chatBox.value.scrollTop = chatBox.value.scrollHeight
  })
}

async function onSend() {
  await sendMessage(inputText.value)
  inputText.value = ''
  nextTick(() => inputEl.value?.focus())
}

// 每当 messages 变化时自动滚到底部
import { watch } from 'vue'
watch(() => state.messages.length, () => scrollChat())
</script>
