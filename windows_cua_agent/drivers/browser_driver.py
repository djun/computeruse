from __future__ import annotations

import base64
import json
import os
import socket
import ssl
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

from cua_agent.agent.state_manager import ActionResult
from cua_agent.utils.config import Settings
from cua_agent.utils.logger import get_logger


class BrowserDriver:
    """
    Windows browser interaction driver via Chrome DevTools Protocol (CDP).

    Requires launching a Chromium-based browser with:
      --remote-debugging-port=9222
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)

    def execute_browser_action(self, action: dict) -> ActionResult:
        cmd = action.get("command")
        app = action.get("app_name", "Chrome")

        if cmd == "get_page_content":
            return self._get_page_content(app)
        if cmd == "get_links":
            return self._get_links(app)
        if cmd == "navigate":
            return self._navigate(app, action.get("url"))
        if cmd == "fill_form":
            return self._fill_form(app, action.get("selector"), action.get("value"))
        if cmd == "click_element":
            return self._click_element(app, action.get("selector"))
        if cmd == "get_dom_tree":
            return self._get_dom_tree(app)
        if cmd == "run_javascript":
            return self._run_arbitrary_js(app, action.get("value"))
        if cmd in ["go_back", "go_forward", "reload"]:
            return self._handle_nav_command(app, cmd)

        return ActionResult(success=False, reason=f"unknown browser command: {cmd}")

    def get_current_url(self, app_name: str) -> Optional[str]:  # noqa: ARG002
        js = "JSON.stringify({status: 'success', result: location.href})"
        res = self._run_js_with_result("Chromium", f"(function(){{return {js};}})();", "get_current_url")
        if not res.success:
            return None
        data = (res.metadata or {}).get("data") or {}
        if isinstance(data, dict):
            url = data.get("result")
            return url if isinstance(url, str) and url else None
        return None

    # --- CDP-backed implementations ---

    def _navigate(self, app_name: str, url: Optional[str]) -> ActionResult:  # noqa: ARG002
        if not url:
            return ActionResult(success=False, reason="url required")
        try:
            with _CDPClient(timeout_s=self.settings.browser_navigation_timeout_s) as cdp:
                cdp.call("Page.enable")
                cdp.call("Page.navigate", {"url": url})
            return ActionResult(success=True, reason="navigate success")
        except Exception as exc:
            return ActionResult(success=False, reason=f"navigate failed: {exc}")

    def _get_page_content(self, app_name: str) -> ActionResult:  # noqa: ARG002
        wrapped_js = """
        (function() {
            try {
                var content = document.body.innerText;
                return JSON.stringify({status: "success", result: content});
            } catch (e) {
                return JSON.stringify({status: "error", message: e.toString()});
            }
        })();
        """
        return self._run_js_with_result(app_name, wrapped_js, "get_page_content")

    def _get_links(self, app_name: str) -> ActionResult:  # noqa: ARG002
        js_payload = """
        (function() {
            try {
                var links = Array.from(document.links).slice(0, 50).map(l => ({
                    text: l.innerText.replace(/\\n/g, ' ').trim(),
                    url: l.href
                })).filter(l => l.text.length > 0);
                return JSON.stringify({status: "success", result: links});
            } catch (e) {
                return JSON.stringify({status: "error", message: e.toString()});
            }
        })();
        """
        return self._run_js_with_result(app_name, js_payload, "get_links")

    def _fill_form(self, app_name: str, selector: str, value: str) -> ActionResult:  # noqa: ARG002
        if not selector or value is None:
            return ActionResult(success=False, reason="fill_form requires selector and value")

        selector_js = json.dumps(selector)
        value_js = json.dumps(value)
        js_payload = f"""
        (function() {{
            function setNativeValue(el, val) {{
                const directDescriptor = Object.getOwnPropertyDescriptor(el, "value");
                const directSetter = directDescriptor && directDescriptor.set;
                const proto = el instanceof HTMLTextAreaElement
                    ? HTMLTextAreaElement.prototype
                    : el instanceof HTMLSelectElement
                        ? HTMLSelectElement.prototype
                        : HTMLInputElement.prototype;
                const protoDescriptor = Object.getOwnPropertyDescriptor(proto, "value");
                const protoSetter = protoDescriptor && protoDescriptor.set;

                if (directSetter && protoSetter && directSetter !== protoSetter) {{
                    protoSetter.call(el, val);
                }} else if (protoSetter) {{
                    protoSetter.call(el, val);
                }} else if (directSetter) {{
                    directSetter.call(el, val);
                }} else {{
                    el.value = val;
                }}

                const tracker = el._valueTracker;
                if (tracker) tracker.setValue(val);
            }}

            try {{
                var el = document.querySelector({selector_js});
                if (!el) return JSON.stringify({{status: "error", message: "Element not found: " + {selector_js}}});

                setNativeValue(el, {value_js});
                el.dispatchEvent(new Event("input", {{ bubbles: true }}));
                el.dispatchEvent(new Event("change", {{ bubbles: true }}));
                el.dispatchEvent(new Event("blur", {{ bubbles: true }}));
                return JSON.stringify({{status: "success"}});
            }} catch (e) {{
                return JSON.stringify({{status: "error", message: e.toString()}});
            }}
        }})();
        """
        return self._run_js_with_result(app_name, js_payload, "fill_form")

    def _click_element(self, app_name: str, selector: str) -> ActionResult:  # noqa: ARG002
        if not selector:
            return ActionResult(success=False, reason="click_element requires selector")
        selector_js = json.dumps(selector)
        js_payload = f"""
        (function() {{
            try {{
                var el = document.querySelector({selector_js});
                if (!el) return JSON.stringify({{status: "error", message: "Element not found: " + {selector_js}}});
                el.click();
                return JSON.stringify({{status: "success"}});
            }} catch (e) {{
                return JSON.stringify({{status: "error", message: e.toString()}});
            }}
        }})();
        """
        return self._run_js_with_result(app_name, js_payload, "click_element")

    def _get_dom_tree(self, app_name: str) -> ActionResult:  # noqa: ARG002
        js_payload = """
        (function() {{
            const INTERACTIVE = new Set(["a","button","input","select","textarea","option","label"]);
            const CONTAINERS = new Set(["div","span","ul","li","ol","section","header","footer","main","nav","form"]);
            const PRUNE = new Set(["script","style","noscript","meta","link","svg","path"]);

            function distill(node, depth) {{
                if (depth > 6) return null;
                if (node.nodeType === 3) {{
                    var txt = node.textContent.trim();
                    return (txt.length > 0)? txt.substring(0, 200) : null;
                }}
                if (node.nodeType !== 1 && node.nodeType !== 11) return null;

                var tagName = node.tagName ? node.tagName.toLowerCase() : "#fragment";
                if (PRUNE.has(tagName)) return null;

                var isInteractive = INTERACTIVE.has(tagName);
                var isContainer = CONTAINERS.has(tagName) || tagName === "#fragment" || tagName === "#shadow-root";
                if (!isInteractive && !isContainer) return null;

                var rect = node.getBoundingClientRect ? node.getBoundingClientRect() : {{width:1,height:1,x:0,y:0}};
                if (rect.width === 0 || rect.height === 0) return null;

                var obj = {{ tag: tagName }};
                if (node.id) obj.id = node.id;
                if (node.className) obj.class = node.className;
                if (tagName === 'a' && node.href) obj.href = node.href;
                if (tagName === 'input') {{
                    obj.type = node.type;
                    obj.value = node.value;
                    obj.name = node.name;
                    obj.placeholder = node.placeholder;
                }}
                if (node.getAttribute && node.getAttribute('role')) obj.role = node.getAttribute('role');
                if (node.getAttribute && node.getAttribute('aria-label')) obj.ariaLabel = node.getAttribute('aria-label');

                obj.rect = {{
                    x: rect.x,
                    y: rect.y,
                    width: rect.width,
                    height: rect.height
                }};

                var text = (node.innerText || "").trim();
                if (text && text.length > 0) {{
                    obj.text = text.substring(0, 200);
                }}

                var children = [];
                if (node.shadowRoot) {{
                    var shadowChild = distill(node.shadowRoot, depth + 1);
                    if (shadowChild) {{
                        children.push({{ tag: "#shadow-root", children: shadowChild.children || [] }});
                    }}
                }}
                node.childNodes.forEach(c => {{
                    var d = distill(c, depth + 1);
                    if (d) children.push(d);
                }});

                if (children.length > 0) obj.children = children;
                else if (!isInteractive) return null;
                return obj;
            }}
            return JSON.stringify(distill(document.body, 0));
        }})();
        """
        return self._run_js_with_result(app_name, js_payload, "get_dom_tree")

    def _run_arbitrary_js(self, app_name: str, js_code: str) -> ActionResult:  # noqa: ARG002
        if not js_code:
            return ActionResult(success=False, reason="no JS code provided")

        promise_wait_ms = max(500, min(int(self.settings.browser_script_timeout_s * 1000) - 500, 4000))
        wrapped = f"""
        (function() {{
            const runner = (async () => {{
                try {{
                    const result = await (async () => {{ {js_code} }})();
                    return {{status: "success", result: result}};
                }} catch (e) {{
                    const msg = (e && e.stack) ? e.stack : (e && e.message) ? e.message : String(e);
                    return {{status: "error", message: msg}};
                }}
            }})();

            let done = false;
            let value = null;
            runner.then(v => {{ done = true; value = v; }}).catch(err => {{
                done = true;
                const msg = (err && err.stack) ? err.stack : (err && err.message) ? err.message : String(err);
                value = {{status: "error", message: msg}};
            }});

            const start = Date.now();
            const timeout = {promise_wait_ms};
            while (!done && (Date.now() - start) < timeout) {{}}

            if (!done) {{
                return JSON.stringify({{status: "error", message: "Promise unresolved after " + timeout + "ms"}});
            }}
            return JSON.stringify(value);
        }})();
        """
        return self._run_js_with_result(app_name, wrapped, "run_javascript")

    def _handle_nav_command(self, app_name: str, cmd: str) -> ActionResult:  # noqa: ARG002
        if cmd == "go_back":
            js = "history.back()"
        elif cmd == "go_forward":
            js = "history.forward()"
        else:
            js = "location.reload()"

        wrapped = f"""
        (function() {{
            try {{
                {js};
                return JSON.stringify({{status: "success"}});
            }} catch (e) {{
                return JSON.stringify({{status: "error", message: e.toString()}});
            }}
        }})();
        """
        return self._run_js_with_result(app_name, wrapped, cmd)

    def _run_js_with_result(self, app_name: str, js_code: str, label: str, timeout: float | None = None) -> ActionResult:  # noqa: ARG002
        timeout = timeout or self.settings.browser_script_timeout_s
        try:
            with _CDPClient(timeout_s=timeout) as cdp:
                cdp.call("Runtime.enable")
                resp = cdp.call(
                    "Runtime.evaluate",
                    {
                        "expression": js_code,
                        "returnByValue": True,
                        "awaitPromise": True,
                        "userGesture": True,
                    },
                )
            result = (((resp or {}).get("result") or {}).get("value")) if isinstance(resp, dict) else None
            if result is None:
                return ActionResult(success=False, reason=f"{label} returned no value", metadata={"raw": resp})
            # Most payloads return a JSON string.
            if isinstance(result, str):
                try:
                    data = json.loads(result)
                except Exception:
                    data = {"status": "success", "result": result}
            else:
                data = {"status": "success", "result": result}
            ok = isinstance(data, dict) and data.get("status") == "success"
            return ActionResult(success=bool(ok), reason=f"{label} success" if ok else f"{label} failed", metadata={"data": data})
        except Exception as exc:
            return ActionResult(success=False, reason=f"{label} exception: {exc}")


class _CDPError(RuntimeError):
    pass


@dataclass
class _CDPTarget:
    websocket_url: str


class _CDPClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 9222, timeout_s: float = 8.0) -> None:
        self.host = host
        self.port = port
        self.timeout_s = float(timeout_s)
        self._sock: socket.socket | None = None
        self._next_id = 1

    def __enter__(self) -> "_CDPClient":
        target = self._discover_target()
        self._connect_ws(target.websocket_url)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def close(self) -> None:
        try:
            if self._sock:
                try:
                    self._send_close()
                except Exception:
                    pass
                try:
                    self._sock.close()
                finally:
                    self._sock = None
        except Exception:
            self._sock = None

    def call(self, method: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if not self._sock:
            raise _CDPError("CDP websocket not connected")
        msg_id = self._next_id
        self._next_id += 1
        payload = {"id": msg_id, "method": method}
        if params:
            payload["params"] = params
        self._send_text(json.dumps(payload))

        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            message = self._recv_message(timeout=deadline - time.time())
            if not message:
                continue
            try:
                obj = json.loads(message)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("id") == msg_id:
                if "error" in obj:
                    raise _CDPError(str(obj["error"]))
                return obj.get("result") or {}
        raise _CDPError(f"timeout waiting for {method}")

    def _discover_target(self) -> _CDPTarget:
        url = f"http://{self.host}:{self.port}/json"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            raise _CDPError(
                f"CDP discovery failed at {url}. Launch Chrome/Edge with --remote-debugging-port=9222. ({exc})"
            ) from exc

        if not isinstance(data, list) or not data:
            raise _CDPError("CDP discovery returned no targets")

        # Choose the first visible page target.
        for t in data:
            if not isinstance(t, dict):
                continue
            if t.get("type") != "page":
                continue
            ws = t.get("webSocketDebuggerUrl")
            if ws and isinstance(ws, str):
                return _CDPTarget(websocket_url=ws)

        raise _CDPError("No page target found in CDP /json listing")

    def _connect_ws(self, ws_url: str) -> None:
        parsed = urlparse(ws_url)
        if parsed.scheme not in {"ws", "wss"}:
            raise _CDPError(f"unsupported websocket scheme: {parsed.scheme}")
        host = parsed.hostname or self.host
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query

        raw_sock = socket.create_connection((host, port), timeout=self.timeout_s)
        raw_sock.settimeout(self.timeout_s)
        if parsed.scheme == "wss":
            ctx = ssl.create_default_context()
            raw_sock = ctx.wrap_socket(raw_sock, server_hostname=host)

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        raw_sock.sendall(req.encode("ascii"))

        resp = self._recv_http_headers(raw_sock)
        if " 101 " not in resp.split("\r\n", 1)[0]:
            raise _CDPError(f"websocket upgrade failed: {resp.splitlines()[0] if resp else 'no response'}")

        self._sock = raw_sock

    def _recv_http_headers(self, sock: socket.socket) -> str:
        data = b""
        deadline = time.time() + self.timeout_s
        while b"\r\n\r\n" not in data and time.time() < deadline:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
            if len(data) > 65536:
                break
        head = data.split(b"\r\n\r\n", 1)[0]
        return head.decode("utf-8", errors="replace")

    # --- WebSocket framing ---

    def _send_text(self, text: str) -> None:
        if not self._sock:
            raise _CDPError("not connected")
        payload = text.encode("utf-8")
        frame = self._build_frame(opcode=0x1, payload=payload, mask=True)
        self._sock.sendall(frame)

    def _send_close(self) -> None:
        if not self._sock:
            return
        frame = self._build_frame(opcode=0x8, payload=b"", mask=True)
        self._sock.sendall(frame)

    def _recv_message(self, timeout: float) -> str:
        if not self._sock:
            return ""
        self._sock.settimeout(max(0.1, min(timeout, self.timeout_s)))
        while True:
            opcode, payload = self._read_frame()
            if opcode == 0x1:
                return payload.decode("utf-8", errors="replace")
            if opcode == 0x8:
                raise _CDPError("websocket closed")
            if opcode == 0x9:
                # Ping -> Pong
                pong = self._build_frame(opcode=0xA, payload=payload, mask=True)
                self._sock.sendall(pong)
                continue
            if opcode == 0xA:
                continue
            # Ignore other frames/events.

    def _read_frame(self) -> tuple[int, bytes]:
        if not self._sock:
            raise _CDPError("not connected")
        first2 = self._recv_exact(2)
        b1, b2 = first2[0], first2[1]
        fin = (b1 >> 7) & 1
        opcode = b1 & 0x0F
        masked = (b2 >> 7) & 1
        length = b2 & 0x7F

        if fin != 1:
            raise _CDPError("fragmented frames not supported")

        if length == 126:
            length = int.from_bytes(self._recv_exact(2), "big")
        elif length == 127:
            length = int.from_bytes(self._recv_exact(8), "big")

        mask_key = b""
        if masked:
            mask_key = self._recv_exact(4)

        payload = self._recv_exact(length) if length else b""
        if masked:
            payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
        return opcode, payload

    def _recv_exact(self, n: int) -> bytes:
        if not self._sock:
            raise _CDPError("not connected")
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise _CDPError("socket closed")
            buf += chunk
        return buf

    def _build_frame(self, *, opcode: int, payload: bytes, mask: bool) -> bytes:
        fin_opcode = 0x80 | (opcode & 0x0F)
        out = bytearray([fin_opcode])

        length = len(payload)
        mask_bit = 0x80 if mask else 0x00
        if length <= 125:
            out.append(mask_bit | length)
        elif length <= 0xFFFF:
            out.append(mask_bit | 126)
            out.extend(length.to_bytes(2, "big"))
        else:
            out.append(mask_bit | 127)
            out.extend(length.to_bytes(8, "big"))

        if mask:
            key = os.urandom(4)
            out.extend(key)
            out.extend(bytes(b ^ key[i % 4] for i, b in enumerate(payload)))
        else:
            out.extend(payload)
        return bytes(out)

