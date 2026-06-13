import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// La app se sirve desde visualizador/server.py en http://localhost:8080/app/
// (mismo origen que el index.html legacy → el localStorage bwg_* se conserva).
// En dev (vite :5173) los data/*.js se proxyan al server.py de :8080.
export default defineConfig({
  base: '/app/',
  plugins: [react(), tailwindcss()],
  build: {
    outDir: '../visualizador/app',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/data': 'http://localhost:8080',
    },
  },
})
