import type { NextConfig } from "next";

const BACKEND_URL =
  process.env.BACKEND_URL || `http://localhost:${process.env.BACKEND_PORT || "8080"}`;

const nextConfig: NextConfig = {
  output: "standalone",
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
