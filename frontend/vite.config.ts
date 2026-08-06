import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
    plugins: [
        react(),
        tailwindcss(),
    ],
    server: {
        port: 5173,
        proxy: {
            '/api': {
                target: 'http://localhost:8000',
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api/, ''),
            },
        },
    },
    build: {
        // The WebGL engine is intentionally lazy-loaded; keep warnings focused on
        // unexpected growth in the initial application bundle.
        chunkSizeWarningLimit: 2000,
        rollupOptions: {
            output: {
                manualChunks(id) {
                    if (id.includes('node_modules/three') || id.includes('node_modules/react-globe.gl') || id.includes('node_modules/satellite.js')) {
                        return 'orbital-globe';
                    }
                    if (id.includes('node_modules/recharts')) return 'analytics-charts';
                    if (id.includes('node_modules/react')) return 'react-vendor';
                },
            },
        },
    },
})
