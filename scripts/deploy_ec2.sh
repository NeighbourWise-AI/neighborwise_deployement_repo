#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# NeighbourWise AI — EC2 Deployment Script
# ══════════════════════════════════════════════════════════════════════════════
# Usage (run once on a fresh EC2 Amazon Linux 2023 / Ubuntu 22.04 instance):
#   chmod +x deploy_ec2.sh
#   ./deploy_ec2.sh
#
# What it does:
#   1. Installs Docker + Docker Compose v2
#   2. Clones your repo (or pulls latest if already present)
#   3. Copies .env (you must have SCP'd it first)
#   4. Builds images + starts all services
#   5. (Optional) installs Certbot for free TLS via Let's Encrypt
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail
IFS=$'\n\t'

# ── Config ────────────────────────────────────────────────────────────────────
REPO_URL="${REPO_URL:-https://github.com/YOUR_ORG/neighbourwise-agents.git}"
APP_DIR="${APP_DIR:-/opt/neighbourwise}"
DOMAIN="${DOMAIN:-}"        # set to your domain for TLS, e.g. app.example.com
INSTALL_CERTBOT="${INSTALL_CERTBOT:-false}"
# ─────────────────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════"
echo " NeighbourWise AI — EC2 Deployment"
echo "═══════════════════════════════════════════════════════"

# ── 1. Detect OS ──────────────────────────────────────────────────────────────
if grep -qi "ubuntu" /etc/os-release 2>/dev/null; then
  OS="ubuntu"
elif grep -qi "amazon" /etc/os-release 2>/dev/null; then
  OS="amazon"
else
  echo "Unsupported OS. Script tested on Ubuntu 22.04 and Amazon Linux 2023."
  exit 1
fi
echo "[1/6] Detected OS: $OS"

# ── 2. Install Docker ─────────────────────────────────────────────────────────
echo "[2/6] Installing Docker..."
if command -v docker &>/dev/null; then
  echo "      Docker already installed: $(docker --version)"
else
  if [[ "$OS" == "ubuntu" ]]; then
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg lsb-release
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
      | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
       https://download.docker.com/linux/ubuntu \
       $(lsb_release -cs) stable" \
      | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
      docker-buildx-plugin docker-compose-plugin
  elif [[ "$OS" == "amazon" ]]; then
    sudo dnf install -y docker
    sudo systemctl enable --now docker
    # Docker Compose v2 plugin for Amazon Linux
    COMPOSE_VERSION="v2.27.1"
    sudo mkdir -p /usr/local/lib/docker/cli-plugins
    sudo curl -SL \
      "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64" \
      -o /usr/local/lib/docker/cli-plugins/docker-compose
    sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
  fi
fi

# Allow current user to run docker without sudo
sudo usermod -aG docker "$USER" || true
echo "      Docker: $(docker --version)"
echo "      Compose: $(docker compose version)"

# ── 3. Clone / update repo ───────────────────────────────────────────────────
echo "[3/6] Fetching application code..."
if [[ -d "$APP_DIR/.git" ]]; then
  echo "      Repo exists — pulling latest..."
  cd "$APP_DIR"
  git pull --rebase origin main
else
  echo "      Cloning $REPO_URL → $APP_DIR"
  sudo git clone "$REPO_URL" "$APP_DIR"
  sudo chown -R "$USER:$USER" "$APP_DIR"
  cd "$APP_DIR"
fi

# ── 4. Ensure .env exists ────────────────────────────────────────────────────
echo "[4/6] Checking .env..."
if [[ ! -f "$APP_DIR/.env" ]]; then
  if [[ -f "$HOME/.env.neighbourwise" ]]; then
    cp "$HOME/.env.neighbourwise" "$APP_DIR/.env"
    echo "      Copied .env from $HOME/.env.neighbourwise"
  else
    echo ""
    echo "  ⚠️  No .env found!"
    echo "  SCP your .env to the instance first:"
    echo "    scp .env ec2-user@<IP>:~/.env.neighbourwise"
    echo "  Then re-run this script."
    exit 1
  fi
else
  echo "      .env present ✓"
fi

# Ensure nginx cert directory exists (Certbot writes here)
mkdir -p "$APP_DIR/nginx/certs"
mkdir -p "$APP_DIR/nginx/html"

# ── 5. Build & start services ────────────────────────────────────────────────
echo "[5/6] Building images and starting services..."
cd "$APP_DIR"

# Export BUILD_DATE so Docker layer cache is busted on deploy
export BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Pull base images first to warm the cache
docker compose pull nginx 2>/dev/null || true

# Build app images (--no-cache on first deploy, cached after that)
docker compose build --parallel

# Bring everything up (detached)
docker compose up -d --remove-orphans

echo ""
echo "  Services:"
docker compose ps

# ── 6. Optional: Certbot TLS ─────────────────────────────────────────────────
if [[ "$INSTALL_CERTBOT" == "true" && -n "$DOMAIN" ]]; then
  echo "[6/6] Installing Certbot for $DOMAIN..."
  if [[ "$OS" == "ubuntu" ]]; then
    sudo apt-get install -y certbot
  elif [[ "$OS" == "amazon" ]]; then
    sudo dnf install -y certbot
  fi

  # Stop nginx briefly so Certbot can bind :80 for the ACME challenge
  docker compose stop nginx
  sudo certbot certonly --standalone \
    -d "$DOMAIN" \
    --non-interactive --agree-tos --email "admin@${DOMAIN}" \
    --cert-path "$APP_DIR/nginx/certs/fullchain.pem" \
    --key-path  "$APP_DIR/nginx/certs/privkey.pem"

  # Now uncomment the HTTPS block in nginx config
  sed -i 's/# server {/server {/g; s/#     /    /g' \
    "$APP_DIR/nginx/neighbourwise.conf"
  # Uncomment the HTTP → HTTPS redirect
  sed -i 's|# return 301|return 301|' "$APP_DIR/nginx/neighbourwise.conf"

  docker compose start nginx

  # Auto-renew cron (runs at 2:30 AM daily)
  (crontab -l 2>/dev/null; \
   echo "30 2 * * * certbot renew --quiet && docker compose -f $APP_DIR/docker-compose.yaml restart nginx") \
   | crontab -
  echo "      TLS certificate installed. Cron set up for auto-renewal."
else
  echo "[6/6] Skipping Certbot (INSTALL_CERTBOT=$INSTALL_CERTBOT)"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
PUBLIC_IP=$(curl -s --max-time 3 http://checkip.amazonaws.com || echo "<your-ec2-ip>")
echo ""
echo "═══════════════════════════════════════════════════════"
echo " ✅  NeighbourWise AI is running!"
echo ""
if [[ -n "$DOMAIN" ]]; then
  echo "  Frontend:  https://$DOMAIN"
  echo "  API docs:  https://$DOMAIN/docs"
else
  echo "  Frontend:  http://$PUBLIC_IP:8501   (or http://$PUBLIC_IP via nginx)"
  echo "  API docs:  http://$PUBLIC_IP:8001/docs"
fi
echo ""
echo "  Logs:  docker compose logs -f"
echo "  Stop:  docker compose down"
echo "═══════════════════════════════════════════════════════"
