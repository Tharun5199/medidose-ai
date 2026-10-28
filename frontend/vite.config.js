import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// WHY a dev proxy: the browser calls /api/... on the Vite origin (localhost:5173) and Vite
// forwards it to FastAPI on :8000. The frontend never hard-codes the backend URL, and in
// development the browser sees a single origin. The backend's CORS settings still allow
// direct calls if VITE_API_URL is set instead.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
