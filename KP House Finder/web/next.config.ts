import type { NextConfig } from "next";

// Proxy /api/py/* to the local FastAPI trigger service (src/api.py) so the browser calls
// same-origin and we avoid CORS. Start that service with:
//   poetry run uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000
const PY_API_URL = process.env.PY_API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/py/:path*", destination: `${PY_API_URL}/:path*` }];
  },
};

export default nextConfig;
