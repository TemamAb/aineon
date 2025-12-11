import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#020617", // Slate 950
        surface: "#0f172a",    // Slate 900
        border: "#1e293b",     // Slate 800
        primary: "#3b82f6",    // Blue 500
        success: "#10b981",    // Emerald 500
        warning: "#f59e0b",    // Amber 500
        danger: "#ef4444",     // Red 500
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', "Liberation Mono", "Courier New", 'monospace'],
        sans: ['ui-sans-serif', 'system-ui', 'sans-serif'],
      }
    },
  },
  plugins: [],
};
export default config;
