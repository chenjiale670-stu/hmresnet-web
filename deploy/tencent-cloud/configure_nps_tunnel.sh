#!/usr/bin/env bash
set -euo pipefail

# Run this script on the Tencent Cloud NPS server as a user with passwordless
# sudo. It reads the existing local NPS web credentials without printing them.
# The client id is the already-connected NPC client on the GPU host.

CLIENT_ID="${CLIENT_ID:-4}"
PUBLIC_PORT="${PUBLIC_PORT:-18011}"
TARGET="${TARGET:-127.0.0.1:8011}"
NPS_BASE="${NPS_BASE:-http://127.0.0.1:8080}"

cookie="$(mktemp)"
trap 'rm -f "$cookie"' EXIT

username="$(sudo -n awk -F= '/^web_username[[:space:]]*=/{gsub(/[[:space:]]/,"",$2); print $2}' /etc/nps/conf/nps.conf)"
password="$(sudo -n awk -F= '/^web_password[[:space:]]*=/{gsub(/[[:space:]]/,"",$2); print $2}' /etc/nps/conf/nps.conf)"

login_response="$(curl -sS --max-time 10 -c "$cookie" -b "$cookie" -X POST \
  --data-urlencode "username=$username" \
  --data-urlencode "password=$password" \
  "$NPS_BASE/login/verify")"

python3 - "$login_response" <<'PY'
import json
import sys

data = json.loads(sys.argv[1])
if not data.get("status"):
    raise SystemExit("NPS login rejected")
print("nps-login-ok")
PY

add_response="$(curl -sS --max-time 10 -c "$cookie" -b "$cookie" -X POST \
  --data-urlencode "type=tcp" \
  --data-urlencode "client_id=$CLIENT_ID" \
  --data-urlencode "remark=HMResNet-web" \
  --data-urlencode "port=$PUBLIC_PORT" \
  --data-urlencode "target=$TARGET" \
  --data-urlencode "server_ip=" \
  "$NPS_BASE/index/add")"

python3 - "$add_response" <<'PY'
import json
import sys

data = json.loads(sys.argv[1])
print({key: data.get(key) for key in ("status", "msg", "id")})
if not data.get("status"):
    raise SystemExit("NPS tunnel add rejected")
PY
