/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // No ESLint config is shipped; don't let a missing lint setup fail the build.
  eslint: { ignoreDuringBuilds: true },
};

export default nextConfig;
