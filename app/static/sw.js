// Deepfake AI Detection Service Worker
const CACHE_NAME = 'deepfake-ai-pwa-v1';
const ASSETS_TO_CACHE = [
    '/',
    '/static/css/style.css',
    '/static/js/app.js',
    '/static/logo.svg',
    '/static/icons/icon-192.png',
    '/static/icons/icon-512.png',
    '/static/icons/icon-maskable-192.png',
    '/static/icons/icon-maskable-512.png',
    '/manifest.json'
];

// Install: Cache essential shell assets
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll(ASSETS_TO_CACHE);
        }).then(() => self.skipWaiting())
    );
});

// Activate: Remove older caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(
                keys.map((key) => {
                    if (key !== CACHE_NAME) {
                        return caches.delete(key);
                    }
                })
            );
        }).then(() => self.clients.claim())
    );
});

// Fetch: Always bypass cache for API calls, use Network-first with cache fallback for static assets
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // Bypass cache for inference APIs, health check, and non-GET requests
    if (url.pathname.startsWith('/api') || event.request.method !== 'GET') {
        event.respondWith(fetch(event.request));
        return;
    }

    // Static assets & HTML navigation: Network-first, fallback to cache
    event.respondWith(
        fetch(event.request)
            .then((networkResponse) => {
                if (networkResponse && networkResponse.status === 200) {
                    const responseClone = networkResponse.clone();
                    caches.open(CACHE_NAME).then((cache) => {
                        cache.put(event.request, responseClone);
                    });
                }
                return networkResponse;
            })
            .catch(() => {
                return caches.match(event.request).then((cachedResponse) => {
                    if (cachedResponse) return cachedResponse;
                    if (event.request.mode === 'navigate') {
                        return caches.match('/');
                    }
                });
            })
    );
});
