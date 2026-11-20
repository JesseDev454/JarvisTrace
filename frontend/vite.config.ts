import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  root: fileURLToPath(new URL('./', import.meta.url)),
  plugins: [react()],
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    host: '127.0.0.1',
    proxy: {
      '/api': 'http://127.0.0.1:8770',
      '/media': 'http://127.0.0.1:8770',
      '/evidence': 'http://127.0.0.1:8770',
      '/auth': 'http://127.0.0.1:8770',
    },
  },
});
