import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request


SERVER = """
from pathlib import Path
import sys
import time
from unittest.mock import patch
from llm_index import server

server.PID_DIR = Path(sys.argv[1])
idle = sys.argv[2] == "idle"
sleep = time.sleep

class TestHTTPServer(server.HTTPServer):
    def serve_forever(self, *args, **kwargs):
        (server.PID_DIR / "port").write_text(str(self.server_port))
        return super().serve_forever(*args, **kwargs)

with patch.object(server, "HTTPServer", TestHTTPServer):
    if idle:
        with patch.object(server.time, "sleep", lambda _: sleep(0.1)):
            server.start_server(port=0, timeout=1)
    else:
        server.start_server(port=0)
"""


class ServerShutdownTests(unittest.TestCase):
    def test_shutdown(self):
        for trigger in ("idle", signal.SIGTERM, signal.SIGINT):
            with self.subTest(trigger=trigger), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                process = subprocess.Popen(
                    [sys.executable, "-u", "-c", SERVER, tmp, str(trigger)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    port_file = directory / "port"
                    deadline = time.monotonic() + 10
                    while not port_file.exists():
                        if process.poll() is not None:
                            self.fail(f"Server exited during startup: {process.communicate()}")
                        if time.monotonic() >= deadline:
                            self.fail("Server did not start")
                        time.sleep(0.01)
                    port = int(port_file.read_text())
                    url = f"http://127.0.0.1:{port}/health"
                    with urllib.request.urlopen(url, timeout=2) as response:
                        self.assertEqual(json.load(response)["status"], "ok")
                    self.assertTrue((directory / "server.pid").exists())
                    if trigger != "idle":
                        os.kill(process.pid, trigger)
                    try:
                        stdout, stderr = process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.fail(f"Server hung after {trigger}")
                    self.assertEqual(process.returncode, 0, stderr)
                    self.assertFalse((directory / "server.pid").exists())
                    if trigger == "idle":
                        self.assertIn("Idle for 1s, shutting down.", stdout)
                    with self.assertRaises(OSError):
                        urllib.request.urlopen(url, timeout=1)
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.communicate()


if __name__ == "__main__":
    unittest.main()
