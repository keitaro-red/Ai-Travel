import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    // 开发模式：把 /api 请求代理到 FastAPI 后端（8000 端口）
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
  build: {
    // 构建产物输出到 ../static/dist/，让 FastAPI 直接托管
    outDir: '../static/dist',
  },
})
