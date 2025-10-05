/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          DEFAULT: "#6C63FF",
          light: "#918CFF",
          dark: "#3F3AC9"
        }
      }
    }
  },
  plugins: []
};
