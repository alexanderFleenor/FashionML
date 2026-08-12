import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite proxies /api to the backend during local frontend work. The nginx
// config uses the same route after the app is built.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
