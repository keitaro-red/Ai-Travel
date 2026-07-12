<!-- ================================================================
 SessionSidebar.vue — 左侧会话列表 + 新建按钮
================================================================ -->

<template>
  <aside class="sidebar">
    <div class="sidebar-head">
      <h2>会话</h2>
      <button @click="createThread">+ 新建</button>
    </div>
    <div class="session-list">
      <div v-if="!state.threads.length" class="empty-state">暂无会话</div>
      <div
        v-for="t in state.threads" :key="t.thread_id"
        class="session-item"
        :class="{ active: t.thread_id === state.activeTid }"
        @click="selectThread(t.thread_id)"
      >
        <div class="title">{{ t.title }}</div>
        <div class="meta">{{ t.updated_at.slice(0, 16).replace('T', ' ') }}</div>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { inject } from 'vue'

const state = inject('state')
const selectThread = inject('selectThread')
const createThread = inject('createThread')
</script>
