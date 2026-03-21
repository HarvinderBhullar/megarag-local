import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxy API calls to the FastAPI backend during development
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path,
      },
      "/batch": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/kg": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/ingest": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/query": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
