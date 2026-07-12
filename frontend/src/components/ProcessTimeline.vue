<!-- ================================================================
 ProcessTimeline.vue — 右侧 Agent 执行时间线
================================================================ -->

<template>
  <div class="timeline-area">
    <h3>执行时间线</h3>
    <div class="timeline" ref="tlBox">
      <div v-if="!state.events.length" class="empty-state">
        {{ state.timelineHint }}
      </div>
      <div
        v-for="ev in state.events" :key="ev.key"
        class="timeline-step" :class="ev.status"
      >
        <div class="step-title">
          {{ ev.title }}
          <span class="step-status" :class="'step-status-' + ev.status">{{ ev.status }}</span>
        </div>
        <div class="step-detail">{{ ev.detail }}</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, inject, watch, nextTick } from 'vue'

const state = inject('state')
const tlBox = ref(null)

function scrollTl() {
  nextTick(() => {
    if (tlBox.value) tlBox.value.scrollTop = tlBox.value.scrollHeight
  })
}

// 事件变化时自动滚到底部
watch(() => state.events.length, () => scrollTl())
</script>
