import type { NextConfig } from "next";

const BACKEND_PORT = process.env.BACKEND_PORT || "8080";
const BACKEND_URL = `http://localhost:${BACKEND_PORT}`;

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${BACKEND_URL}/api/:path*`,
      },
      {
        source: '/health',
        destination: `${BACKEND_URL}/health`,
      },
    ];
  },
};

export default nextConfig;
