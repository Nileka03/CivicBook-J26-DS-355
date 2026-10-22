import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// /api is proxied to the FastAPI backend, so the browser only talks to one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
