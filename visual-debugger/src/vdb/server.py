"""The local web server the browser tab talks to.

The debugged program runs on the main thread; this server runs on daemon
threads. They meet in Session: the debugger publishes states, the browser
long-polls for them and posts commands back.
"""

import json
import queue
import secrets
import threading
import tokenize
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .highlight import highlight_lines

INDEX_HTML = Path(__file__).with_name("static").joinpath("index.html")
POLL_SECONDS = 20


class Session:
    def __init__(self):
        self.id = secrets.token_hex(8)
        self.client_seen = threading.Event()
        self.readable_files = set()
        self._cond = threading.Condition()
        self._version = 0
        self._delivered = 0
        self._payload = json.dumps({"version": 0, "state": {"status": "starting"}}).encode()
        self._commands = queue.Queue()
        self._sources = {}

    def publish(self, state):
        with self._cond:
            self._version += 1
            self._payload = json.dumps({"version": self._version, "state": state}).encode()
            self._cond.notify_all()

    def wait_for_state(self, since, timeout=POLL_SECONDS):
        with self._cond:
            self._cond.wait_for(lambda: self._version > since, timeout)
            return self._version, self._payload

    def mark_delivered(self, version):
        """Called once the response bytes are written, so exiting cannot cut them off."""
        with self._cond:
            self._delivered = max(self._delivered, version)
            self._cond.notify_all()

    def wait_until_delivered(self, timeout):
        """Give the browser a moment to receive the final state before the process exits."""
        if not self.client_seen.is_set():
            return
        with self._cond:
            self._cond.wait_for(lambda: self._delivered >= self._version, timeout)

    def post_command(self, command):
        self._commands.put(command)

    def next_command(self):
        while True:
            try:
                return self._commands.get(timeout=0.5)
            except queue.Empty:
                continue

    def source_lines(self, filename):
        if filename not in self.readable_files:
            return None
        if filename not in self._sources:
            with tokenize.open(filename) as f:
                self._sources[filename] = highlight_lines(f.read())
        return self._sources[filename]


class Handler(BaseHTTPRequestHandler):
    server: "DebugServer"

    def log_message(self, *args):
        pass

    def do_GET(self):
        url = urlsplit(self.path)
        if not self._host_allowed():
            return self._send(403, b"forbidden", "text/plain")
        if url.path == "/":
            return self._send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
        if not self.headers.get("X-VDB"):
            return self._send(403, b"forbidden", "text/plain")
        session = self.server.session
        session.client_seen.set()
        query = parse_qs(url.query)
        if url.path == "/api/hello":
            return self._json({"session": session.id})
        if url.path == "/api/state":
            version, payload = session.wait_for_state(int(query.get("since", ["0"])[0]))
            self._send(200, payload, "application/json")
            return session.mark_delivered(version)
        if url.path == "/api/source":
            lines = session.source_lines(query.get("file", [""])[0])
            if lines is None:
                return self._send(404, b"not found", "text/plain")
            return self._json({"lines": lines})
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if not self._host_allowed() or not self.headers.get("X-VDB"):
            return self._send(403, b"forbidden", "text/plain")
        if urlsplit(self.path).path != "/api/command":
            return self._send(404, b"not found", "text/plain")
        length = int(self.headers.get("Content-Length", "0"))
        self.server.session.post_command(json.loads(self.rfile.read(length) or b"{}"))
        return self._json({"ok": True})

    def _host_allowed(self):
        # Rejects DNS-rebinding pages that reach 127.0.0.1 under another name.
        port = self.server.server_address[1]
        return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

    def _json(self, obj):
        self._send(200, json.dumps(obj).encode(), "application/json")

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()


class DebugServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, session, port):
        try:
            super().__init__(("127.0.0.1", port), Handler)
        except OSError:
            super().__init__(("127.0.0.1", 0), Handler)
        self.session = session

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}/"

    def start(self):
        threading.Thread(target=self.serve_forever, name="vdb-server", daemon=True).start()
