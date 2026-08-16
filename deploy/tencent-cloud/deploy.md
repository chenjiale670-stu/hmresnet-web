# Tencent Cloud + DuckDNS deployment

The existing public route at `106.54.36.145` currently serves the FGA-DB
application through `127.0.0.1:18000`. Do not replace that route while testing
HMResNet. Use a separate local port and a separate DuckDNS name, for example
`hmresnet.duckdns.org`.

1. On the GPU host, clone this repository and run the prediction API on an
   unused loopback port:

   ```bash
   export HMRESNET_DEVICE=cuda:0
   cd hmresnet-web
   python -m uvicorn backend.app:app --host 127.0.0.1 --port 8011 --forwarded-allow-ips=127.0.0.1
   ```

2. Add an NPS TCP tunnel that maps a Tencent Cloud loopback port to
   `127.0.0.1:8011` on the GPU host. Keep the Tencent-side listener bound to
   `127.0.0.1`; it does not need a public security-group rule.

3. Configure Nginx on the Tencent Cloud server with the example in
   `nginx/hmresnet.conf`, replacing the hostname and internal NPS port with the
   values created for this tunnel.

4. Point the new DuckDNS A record to `106.54.36.145`, issue the certificate,
   then reload Nginx. DuckDNS supplies DNS only; Nginx owns ports 80/443.

5. Verify from the GPU host first:

   ```bash
   curl http://127.0.0.1:8011/api/health
   ```

6. Do not expose NPS admin ports or local API ports through the security group.
