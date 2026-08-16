#!/usr/bin/env bash
set -euo pipefail

TUNNEL_ID="${1:?usage: inspect_nps_tunnel.sh <tunnel-id>}"
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
if not json.loads(sys.argv[1]).get("status"):
    raise SystemExit("NPS login rejected")
PY

curl -sS --max-time 10 -c "$cookie" -b "$cookie" "$NPS_BASE/index/edit?id=$TUNNEL_ID" \
  | grep -E 'name="(type|client_id|remark|port|target|server_ip)"' \
  | sed -E 's/<[^>]+>//g; s/[[:space:]]+/ /g' \
  | sed -n '1,40p'
