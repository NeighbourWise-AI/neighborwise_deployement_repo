# ══════════════════════════════════════════════════════════════════════════════
# nginx/proxy_rules.conf.inc
# Included by both the HTTP and HTTPS server blocks.
# ══════════════════════════════════════════════════════════════════════════════

# ── Security headers ──────────────────────────────────────────────────────────
add_header X-Frame-Options "SAMEORIGIN"         always;
add_header X-Content-Type-Options "nosniff"     always;
add_header X-XSS-Protection "1; mode=block"     always;
add_header Referrer-Policy "strict-origin"      always;

# ── Gzip ──────────────────────────────────────────────────────────────────────
gzip on;
gzip_types text/plain text/css application/json application/javascript
           text/xml application/xml text/javascript;
gzip_min_length 1024;

# ── FastAPI Backend  — /api/* and /docs /redoc /health /openapi.json ──────────
location /api/ {
    rewrite ^/api/(.*)$ /$1 break;
    proxy_pass         http://backend;
    proxy_http_version 1.1;
    proxy_set_header   Host              $host;
    proxy_set_header   X-Real-IP         $remote_addr;
    proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header   X-Forwarded-Proto $scheme;
    proxy_set_header   Connection        "";
    proxy_read_timeout 300s;   # long for report generation / Graph agent
    proxy_send_timeout 60s;
    client_max_body_size 10M;
}

# Rate-limit the heavy query endpoint
location /api/query {
    limit_req zone=query_limit burst=10 nodelay;
    rewrite ^/api/(.*)$ /$1 break;
    proxy_pass         http://backend;
    proxy_http_version 1.1;
    proxy_set_header   Host              $host;
    proxy_set_header   X-Real-IP         $remote_addr;
    proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header   X-Forwarded-Proto $scheme;
    proxy_read_timeout 300s;
    proxy_send_timeout 60s;
}

# Rate-limit report generation (expensive — hits DALL-E + SARIMAX)
location /api/report/generate {
    limit_req zone=report_limit burst=2 nodelay;
    rewrite ^/api/(.*)$ /$1 break;
    proxy_pass         http://backend;
    proxy_http_version 1.1;
    proxy_set_header   Host              $host;
    proxy_set_header   X-Real-IP         $remote_addr;
    proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header   X-Forwarded-Proto $scheme;
    proxy_read_timeout 420s;
    proxy_send_timeout 60s;
}

# Expose FastAPI docs directly at /docs (nice for debugging on EC2)
location ~ ^/(docs|redoc|openapi\.json|health)$ {
    proxy_pass         http://backend;
    proxy_set_header   Host $host;
    proxy_read_timeout 30s;
}

# ── Streamlit Frontend — WebSocket + HTTP ────────────────────────────────────
# Streamlit uses WebSocket for live updates; nginx MUST upgrade the connection.
location /_stcore/ {
    proxy_pass         http://frontend;
    proxy_http_version 1.1;
    proxy_set_header   Upgrade    $http_upgrade;
    proxy_set_header   Connection "upgrade";
    proxy_set_header   Host       $host;
    proxy_read_timeout 86400s;   # keep WS open indefinitely
}

location /stream {
    proxy_pass         http://frontend;
    proxy_http_version 1.1;
    proxy_set_header   Upgrade    $http_upgrade;
    proxy_set_header   Connection "upgrade";
    proxy_set_header   Host       $host;
    proxy_read_timeout 86400s;
}

# All other traffic → Streamlit
location / {
    proxy_pass         http://frontend;
    proxy_http_version 1.1;
    proxy_set_header   Host              $host;
    proxy_set_header   X-Real-IP         $remote_addr;
    proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header   X-Forwarded-Proto $scheme;
    proxy_set_header   Upgrade           $http_upgrade;
    proxy_set_header   Connection        "upgrade";
    proxy_read_timeout 86400s;
    client_max_body_size 50M;
}
