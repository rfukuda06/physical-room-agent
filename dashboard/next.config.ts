import type { NextConfig } from "next";

// Where the FastAPI backend lives. Read from the env var so it can be
// pointed at a different host if we ever run the dashboard remotely;
// defaults to localhost for local development.
const BACKEND =
  process.env.BACKEND_URL || "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // Pin the workspace root to this directory. Without this, Next.js warns
  // because a stray package-lock.json elsewhere in the user's home dir
  // makes it guess a wrong parent directory.
  turbopack: {
    root: import.meta.dirname,
  },

  // Reverse-proxy the backend under /backend/* so the browser sees
  // all requests as same-origin. Matters most for the MJPEG stream —
  // some browsers refuse to render multipart/x-mixed-replace inside a
  // cross-origin <img> tag even when CORS headers are permissive.
  async rewrites() {
    return [
      {
        source: "/backend/:path*",
        destination: `${BACKEND}/:path*`,
      },
    ];
  },
};

export default nextConfig;
