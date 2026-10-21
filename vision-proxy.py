"""vision-proxy.py -- give the text-only 27B eyes by borrowing a remote vision model.

Sits on the public port in front of llama-server and passes every request through
unchanged, except chat requests that carry images: each image is sent to a separate
vision model (OpenAI-compatible), and the image part is replaced by that model's text
description before the request reaches the 27B. The 27B then runs with --no-mmproj,
and the GPU memory the projector used goes to context instead.

Handles OpenAI image_url parts (/v1/chat/completions) and Anthropic image blocks
(/v1/messages), including images nested in Anthropic tool_result blocks.

Descriptions are cached by image hash and generated without reference to the
conversation, so the same image always becomes the same text. That matters twice:
clients resend the whole history every turn, and identical text keeps the 27B's
prompt cache valid across turns.

The trade-off: the 27B reasons over a description, never the pixels. Detail the
vision model leaves out is gone.

Standard library only. Run with the firewall-allowed interpreter, e.g.
    python vision-proxy.py --listen 0.0.0.0:8080 --upstream 127.0.0.1:8079 \
        --vision-url http://<vision-host>:8080/v1 --vision-model qwen3.5:4b-vision
"""

import argparse
import hashlib
import http.client
import json
import sys
import threading
import time
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REWRITE_PATHS = {"/v1/chat/completions", "/chat/completions", "/v1/messages"}
HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
               "proxy-connection", "proxy-authenticate", "proxy-authorization", "host",
               "content-length"}

DESCRIBE_PROMPT = (
    "Describe this image for a reader who cannot see it. Transcribe all visible text "
    "verbatim, including code, numbers, labels and UI text. Then describe layout, "
    "objects, people, charts or diagrams and anything notable. Be thorough and factual; "
    "do not speculate beyond what is visible."
)

args = None
cache = OrderedDict()
cache_lock = threading.Lock()
CACHE_MAX = 512


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def image_ref(part):
    """Return an image URL (data: or http:) for an image part, or None if not an image."""
    if not isinstance(part, dict):
        return None
    t = part.get("type")
    if t == "image_url":
        u = part.get("image_url")
        return u.get("url") if isinstance(u, dict) else u
    if t == "image":
        src = part.get("source") or {}
        if src.get("type") == "base64":
            return f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
        if src.get("type") == "url":
            return src.get("url")
    return None


def describe(url):
    key = hashlib.sha256(url.encode()).hexdigest()
    with cache_lock:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
    t0 = time.time()
    body = {
        "model": args.vision_model,
        "temperature": 0,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": DESCRIBE_PROMPT},
            {"type": "image_url", "image_url": {"url": url}},
        ]}],
    }
    req = urllib.request.Request(args.vision_url.rstrip("/") + "/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=args.vision_timeout) as r:
            msg = json.loads(r.read())["choices"][0]["message"]
        desc = (msg.get("content") or msg.get("reasoning_content") or "").strip()
        if not desc:
            raise ValueError("empty description")
    except Exception as e:  # never fail the user's request over a picture
        log(f"vision FAILED after {time.time() - t0:.1f}s: {e}")
        return f"[An image was attached here, but the vision model could not describe it ({e}).]"
    text = f"[Image attached here. You cannot see it; {args.vision_model} described it as follows.]\n{desc}"
    with cache_lock:
        cache[key] = text
        while len(cache) > CACHE_MAX:
            cache.popitem(last=False)
    log(f"vision described image in {time.time() - t0:.1f}s ({len(desc)} chars)")
    return text


def collect(content, found):
    """Find image parts in a content list (recursing into tool_result blocks)."""
    if not isinstance(content, list):
        return
    for i, part in enumerate(content):
        url = image_ref(part)
        if url:
            found.append((content, i, url))
        elif isinstance(part, dict) and part.get("type") == "tool_result":
            collect(part.get("content"), found)


def rewrite(payload):
    found = []
    for m in payload.get("messages") or []:
        if isinstance(m, dict):
            collect(m.get("content"), found)
    if not found:
        return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        texts = list(pool.map(describe, [url for _, _, url in found]))
    for (content, i, _), text in zip(found, texts):
        content[i] = {"type": "text", "text": text}
    return True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        pass

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def do_PUT(self):
        self.forward()

    def do_DELETE(self):
        self.forward()

    def do_OPTIONS(self):
        self.forward()

    def forward(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        path_only = self.path.split("?", 1)[0]

        if body and self.command == "POST" and path_only in REWRITE_PATHS:
            try:
                payload = json.loads(body)
                if rewrite(payload):
                    body = json.dumps(payload).encode()
            except ValueError:
                pass  # not JSON; let llama-server answer it

        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
        if body is not None:
            headers["Content-Length"] = str(len(body))

        conn = http.client.HTTPConnection(args.upstream_host, args.upstream_port, timeout=3600)
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except OSError as e:
            conn.close()
            msg = json.dumps({"error": {"message": f"upstream llama-server unavailable: {e}"}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return

        try:
            # Clients that check /props before sending images should see vision as available.
            if self.command == "GET" and path_only == "/props" and resp.status == 200:
                data = resp.read()
                try:
                    props = json.loads(data)
                    props.setdefault("modalities", {})["vision"] = True
                    data = json.dumps(props).encode()
                except ValueError:
                    pass
                self.send_response(resp.status)
                self.copy_headers(resp)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            self.send_response(resp.status)
            self.copy_headers(resp)
            length = resp.getheader("Content-Length")
            if length is not None:
                self.send_header("Content-Length", length)
                self.end_headers()
                self.wfile.write(resp.read())
                return
            # Streaming (SSE) or unknown length: relay as chunked, flushing each piece.
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # client went away; closing the upstream socket cancels the generation
        finally:
            conn.close()

    def copy_headers(self, resp):
        for k, v in resp.getheaders():
            if k.lower() not in HOP_HEADERS:
                self.send_header(k, v)


def main():
    global args
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--listen", default="0.0.0.0:8080")
    p.add_argument("--upstream", default="127.0.0.1:8079")
    p.add_argument("--vision-url", required=True, help="OpenAI-compatible base URL, e.g. http://host:8080/v1")
    p.add_argument("--vision-model", required=True)
    p.add_argument("--vision-timeout", type=float, default=180)
    args = p.parse_args()
    args.upstream_host, port = args.upstream.rsplit(":", 1)
    args.upstream_port = int(port)
    host, port = args.listen.rsplit(":", 1)

    server = ThreadingHTTPServer((host, int(port)), Handler)
    server.daemon_threads = True
    log(f"vision-proxy on {args.listen} -> {args.upstream}; images -> {args.vision_model} @ {args.vision_url}")
    server.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
