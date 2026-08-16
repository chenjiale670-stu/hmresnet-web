# Tencent Cloud + DuckDNS deployment

The existing public route at `https://fgaresnet.duckdns.org` serves FGA-DB
through `127.0.0.1:18000`. HMResNet is mounted alongside it at
`https://fgaresnet.duckdns.org/hmresnet/`; the existing root route is left
unchanged.

1. On the GPU host, clone this repository and run the prediction API on an
   unused loopback port:

   ```bash
   export HMRESNET_DEVICE=cuda:0
   cd hmresnet-web
   python -m uvicorn backend.app:app --host 127.0.0.1 --port 8011 --forwarded-allow-ips=127.0.0.1
   ```

2. Keep a persistent SSH reverse tunnel from the GPU host to Tencent Cloud.
   The included user service `deploy/systemd/hmresnet-reverse-tunnel.service`
   maps Tencent loopback port `18012` to local `127.0.0.1:8011`:

   ```bash
   cp deploy/systemd/hmresnet-reverse-tunnel.service \
      ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable --now hmresnet-reverse-tunnel.service
   ```

   The SSH key referenced by the service must be readable by the account that
   runs the service. The Tencent-side listener remains on `127.0.0.1` and does
   not need a public security-group rule.

3. Add the `/hmresnet/` locations from `nginx/hmresnet-path.conf` to the HTTPS
   server block for `fgaresnet.duckdns.org`, proxying to
   `http://127.0.0.1:18012/`.

4. No new DuckDNS record or certificate is required. Reload Nginx after the
   location is added. DuckDNS supplies DNS only; Nginx owns ports 80/443.

5. Verify from the GPU host first:

   ```bash
   curl http://127.0.0.1:8011/api/health
   ```

6. Do not expose NPS admin ports or local API ports through the security group.
