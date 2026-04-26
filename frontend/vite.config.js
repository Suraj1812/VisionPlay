import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const frontendPort = Number(process.env.FRONTEND_PORT || 5173);
const previewPort = Number(process.env.FRONTEND_PREVIEW_PORT || 4173);

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: frontendPort
  },
  preview: {
    host: "0.0.0.0",
    port: previewPort
  }
});

