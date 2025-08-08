# ============================================================================
# Dockerfile for Future Vision Transport Frontend - Google Cloud Run Deployment
# ============================================================================

FROM nginx:alpine

# Install curl for health checks
RUN apk add --no-cache curl

# Copy static files
COPY segmentation_test_interface.html /usr/share/nginx/html/index.html

# Create custom nginx configuration for Cloud Run
RUN cat > /etc/nginx/conf.d/default.conf << 'EOF'
server {
    listen 80;
    server_name _;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: ws: wss: data: blob: 'unsafe-inline' 'unsafe-eval'; frame-ancestors 'self';" always;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied expired no-cache no-store private must-revalidate auth;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;

    # Root directory
    root /usr/share/nginx/html;
    index index.html;

    # Cache static assets
    location ~* \.(css|js|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Main application
    location / {
        try_files $uri $uri/ /index.html;
        
        # No cache for HTML
        add_header Cache-Control "no-cache, no-store, must-revalidate";
        add_header Pragma "no-cache";
        add_header Expires "0";
    }

    # Health check endpoint for Cloud Run
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }

    # Error pages
    error_page 404 /index.html;
    error_page 500 502 503 504 /index.html;
}
EOF

# Create startup script to configure API URL from environment variable
RUN cat > /docker-entrypoint.d/30-configure-api-url.sh << 'EOF'
#!/bin/sh
set -e

# Replace API URL placeholder in HTML if BACKEND_URL is provided
if [ -n "$BACKEND_URL" ]; then
    echo "Configuring API URL to: $BACKEND_URL"
    sed -i "s|http://localhost:8000|$BACKEND_URL|g" /usr/share/nginx/html/index.html
else
    echo "Using default API URL: http://localhost:8000"
fi

echo "Frontend configuration completed"
EOF

RUN chmod +x /docker-entrypoint.d/30-configure-api-url.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Expose port 80
EXPOSE 80

# Use default nginx startup
CMD ["nginx", "-g", "daemon off;"]