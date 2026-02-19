"""Live debugging dashboard (optional, FastAPI-backed)."""

from __future__ import annotations

import copy
import threading
import time
import uuid
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from cua_agent.utils.config import Settings


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CUA Live Debug Dashboard</title>
  <style>
    :root {
      --bg: #0b1220;
      --panel: #111a2e;
      --panel-2: #17233d;
      --text: #e5ecff;
      --muted: #9fb2d9;
      --ok: #36d399;
      --warn: #fbbf24;
      --bad: #f87171;
      --accent: #60a5fa;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at top, #132142 0%, var(--bg) 55%);
      color: var(--text);
      font-family: "JetBrains Mono", Consolas, "Liberation Mono", Menlo, monospace;
    }
    .wrap {
      max-width: 1380px;
      margin: 0 auto;
      padding: 14px;
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 12px;
    }
    .panel {
      background: linear-gradient(180deg, var(--panel), var(--panel-2));
      border: 1px solid #253457;
      border-radius: 10px;
      overflow: hidden;
    }
    .hd {
      padding: 8px 10px;
      border-bottom: 1px solid #253457;
      color: var(--muted);
      font-size: 12px;
    }
    .bd { padding: 10px; }
    .screen-wrap {
      position: relative;
      width: 100%;
      background: #03060e;
      border-radius: 8px;
      overflow: hidden;
      min-height: 360px;
      border: 1px solid #263861;
    }
    #screen {
      width: 100%;
      display: block;
    }
    #crosshair {
      position: absolute;
      width: 30px;
      height: 30px;
      margin-left: -15px;
      margin-top: -15px;
      border: 2px solid var(--bad);
      border-radius: 999px;
      pointer-events: none;
      display: none;
    }
    #crosshair::before,
    #crosshair::after {
      content: "";
      position: absolute;
      background: var(--bad);
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
    }
    #crosshair::before { width: 2px; height: 42px; }
    #crosshair::after { width: 42px; height: 2px; }
    .grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }
    .status {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 6px;
      font-size: 11px;
      border: 1px solid #31466f;
      color: var(--muted);
    }
    .status.ok { color: var(--ok); border-color: #2b8c66; }
    .status.fail { color: var(--bad); border-color: #8c3a3a; }
    .status.run { color: var(--warn); border-color: #8b6a20; }
    .mono {
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      font-size: 12px;
      line-height: 1.4;
      color: var(--text);
    }
    ul {
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 6px;
    }
    li {
      border: 1px solid #293a60;
      background: rgba(7, 13, 24, 0.5);
      border-radius: 8px;
      padding: 7px;
      font-size: 12px;
      line-height: 1.4;
    }
    .muted { color: var(--muted); font-size: 12px; }
    @media (max-width: 980px) {
      .wrap { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="panel">
      <div class="hd">Screen + Next Click Preview</div>
      <div class="bd">
        <div class="screen-wrap">
          <img id="screen" alt="latest frame" />
          <div id="crosshair"></div>
        </div>
        <p class="muted" id="meta">Waiting for session...</p>
      </div>
    </section>
    <div class="grid">
      <section class="panel">
        <div class="hd">Current Plan</div>
        <div class="bd"><ul id="plan"></ul></div>
      </section>
      <section class="panel">
        <div class="hd">Cognitive Trace</div>
        <div class="bd"><ul id="thoughts"></ul></div>
      </section>
      <section class="panel">
        <div class="hd">Last Action + Verification</div>
        <div class="bd">
          <pre id="action" class="mono"></pre>
          <pre id="verify" class="mono"></pre>
        </div>
      </section>
      <section class="panel">
        <div class="hd">Event Feed</div>
        <div class="bd"><ul id="events"></ul></div>
      </section>
    </div>
  </div>
  <script>
    const stateUrl = "/api/state";
    const screen = document.getElementById("screen");
    const cross = document.getElementById("crosshair");
    const meta = document.getElementById("meta");
    const plan = document.getElementById("plan");
    const thoughts = document.getElementById("thoughts");
    const action = document.getElementById("action");
    const verify = document.getElementById("verify");
    const events = document.getElementById("events");

    function statusClass(status) {
      if (status === "completed") return "status ok";
      if (status === "failed") return "status fail";
      if (status === "in_progress") return "status run";
      return "status";
    }

    function fillList(el, items, mapper) {
      el.innerHTML = "";
      (items || []).forEach((item) => {
        const li = document.createElement("li");
        li.innerHTML = mapper(item);
        el.appendChild(li);
      });
    }

    function updateCrosshair(snapshot) {
      const c = snapshot.crosshair;
      const d = snapshot.display || {};
      if (!c || !d.logical_width || !d.logical_height || !screen.clientWidth || !screen.clientHeight) {
        cross.style.display = "none";
        return;
      }
      const x = (c.x / d.logical_width) * screen.clientWidth;
      const y = (c.y / d.logical_height) * screen.clientHeight;
      cross.style.display = "block";
      cross.style.left = x + "px";
      cross.style.top = y + "px";
    }

    function render(snapshot) {
      if (snapshot.frame_b64) {
        const frameMime = snapshot.frame_mime || "image/jpeg";
        screen.src = "data:" + frameMime + ";base64," + snapshot.frame_b64;
      }
      updateCrosshair(snapshot);
      const step = snapshot.current_step_id === null ? "n/a" : snapshot.current_step_id;
      const status = snapshot.session_status || "idle";
      const updated = snapshot.updated_at ? new Date(snapshot.updated_at * 1000).toLocaleTimeString() : "n/a";
      meta.textContent = `session=${snapshot.session_id || "none"} | step=${step} | status=${status} | updated=${updated}`;

      fillList(plan, snapshot.plan_steps, (s) => {
        const stepId = s.id === null ? "?" : s.id;
        return `<div><span class="${statusClass(s.status)}">${s.status || "pending"}</span> step ${stepId}</div><div>${(s.description || "").replace(/</g, "&lt;")}</div>`;
      });
      fillList(thoughts, snapshot.thoughts, (t) => `<div>${String(t).replace(/</g, "&lt;")}</div>`);
      fillList(events, snapshot.events, (e) => `<div><span class="muted">${e.ts || ""}</span> ${String(e.text || "").replace(/</g, "&lt;")}</div>`);

      action.textContent = JSON.stringify(snapshot.last_action || {}, null, 2);
      verify.textContent = JSON.stringify(snapshot.last_verification || {}, null, 2);
    }

    async function tick() {
      try {
        const res = await fetch(stateUrl, { cache: "no-store" });
        if (!res.ok) return;
        const snapshot = await res.json();
        render(snapshot);
      } catch (err) {
        meta.textContent = "dashboard disconnected";
      }
    }
    setInterval(tick, 500);
    tick();
  </script>
</body>
</html>
"""


def _serialize_plan_steps(plan: Any) -> List[Dict[str, Any]]:
    if not plan:
        return []
    out: List[Dict[str, Any]] = []
    for step in getattr(plan, "steps", []) or []:
        out.append(
            {
                "id": getattr(step, "id", None),
                "description": getattr(step, "description", ""),
                "status": getattr(step, "status", ""),
                "success_criteria": getattr(step, "success_criteria", ""),
            }
        )
    return out


def _clip(value: str, limit: int = 600) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "... [truncated]"


class LiveDebugDashboard:
    """Thread-safe runtime state + optional FastAPI server."""

    def __init__(self, settings: Settings, logger: Any) -> None:
        self.settings = settings
        self.logger = logger
        self.enabled = False
        self.url = ""
        self._lock = threading.Lock()
        self._thoughts: Deque[str] = deque(maxlen=80)
        self._events: Deque[Dict[str, str]] = deque(maxlen=120)
        self._frame_mime = "image/png" if str(settings.encode_format or "").strip().lower() == "png" else "image/jpeg"
        self._state: Dict[str, Any] = {
            "session_id": None,
            "session_status": "idle",
            "prompt": "",
            "frame_b64": "",
            "frame_mime": self._frame_mime,
            "display": {"logical_width": 0, "logical_height": 0},
            "crosshair": None,
            "plan_steps": [],
            "current_step_id": None,
            "last_action": {},
            "last_verification": {},
            "loop_state": {},
            "thoughts": [],
            "events": [],
            "summary": {},
            "updated_at": time.time(),
        }
        self._server = None
        self._thread: Optional[threading.Thread] = None

        if not settings.enable_debug_dashboard:
            return
        self._boot_server()

    def _boot_server(self) -> None:
        try:
            from fastapi import FastAPI
            from fastapi.responses import HTMLResponse, JSONResponse
            import uvicorn
        except Exception as exc:
            self.logger.warning(
                "Debug dashboard disabled: FastAPI/Uvicorn unavailable (%s). Install fastapi+uvicorn.",
                exc,
            )
            return

        app = FastAPI(title="CUA Debug Dashboard")

        @app.get("/")
        def _index() -> HTMLResponse:
            return HTMLResponse(_DASHBOARD_HTML)

        @app.get("/api/state")
        def _api_state() -> JSONResponse:
            return JSONResponse(self.snapshot())

        @app.get("/health")
        def _health() -> Dict[str, str]:
            return {"status": "ok"}

        host = self.settings.debug_dashboard_host
        port = self.settings.debug_dashboard_port
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True, name="cua-debug-dashboard")
        self._thread.start()

        self.enabled = True
        self.url = f"http://{host}:{port}"
        self.logger.info("Debug dashboard started at %s", self.url)

    def _append_event(self, text: str) -> None:
        self._events.append(
            {
                "ts": time.strftime("%H:%M:%S"),
                "text": _clip(text, 500),
            }
        )

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            payload = copy.deepcopy(self._state)
            payload["thoughts"] = list(self._thoughts)
            payload["events"] = list(self._events)
            return payload

    def start_session(self, prompt: str, plan: Any, frame_b64: str, display: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._thoughts.clear()
            self._events.clear()
            self._state.update(
                {
                    "session_id": str(uuid.uuid4())[:8],
                    "session_status": "running",
                    "prompt": prompt,
                    "frame_b64": frame_b64 or "",
                    "frame_mime": self._frame_mime,
                    "display": {
                        "logical_width": int(getattr(display, "logical_width", 0) or 0),
                        "logical_height": int(getattr(display, "logical_height", 0) or 0),
                    },
                    "crosshair": None,
                    "plan_steps": _serialize_plan_steps(plan),
                    "current_step_id": getattr(getattr(plan, "current_step", lambda: None)(), "id", None) if plan else None,
                    "last_action": {},
                    "last_verification": {},
                    "loop_state": {},
                    "summary": {},
                    "updated_at": time.time(),
                }
            )
            self._append_event(f"session_started: {prompt}")

    def push_thought(self, text: str) -> None:
        if not self.enabled or not text:
            return
        with self._lock:
            self._thoughts.append(_clip(text, 900))
            self._state["updated_at"] = time.time()

    def push_action(
        self,
        action: Dict[str, Any],
        plan: Any,
        current_step_id: Any,
        loop_state: Optional[Dict[str, Any]],
        frame_b64: Optional[str],
        crosshair: Optional[Dict[str, Any]],
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            summarized = self._summarize_action(action)
            self._state["last_action"] = summarized
            self._state["current_step_id"] = current_step_id
            self._state["plan_steps"] = _serialize_plan_steps(plan)
            self._state["loop_state"] = dict(loop_state or {})
            if frame_b64:
                self._state["frame_b64"] = frame_b64
            self._state["crosshair"] = crosshair
            self._state["updated_at"] = time.time()
            self._append_event(f"proposed_action: {summarized.get('type')}")

    def push_verification(
        self,
        action: Dict[str, Any],
        changed: bool,
        hash_distance: Optional[int],
        ssim_score: Optional[float],
        ax_changed: bool,
        note: str = "",
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._state["last_verification"] = {
                "action_type": action.get("type"),
                "changed": bool(changed),
                "hash_distance": hash_distance,
                "ssim_score": ssim_score,
                "ax_changed": bool(ax_changed),
                "note": note,
            }
            self._state["updated_at"] = time.time()
            verdict = "changed" if changed else "no_change"
            self._append_event(f"verification:{verdict}:{action.get('type')}")

    def push_event(self, text: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._append_event(text)
            self._state["updated_at"] = time.time()

    def finish_session(self, summary: Dict[str, Any], plan: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._state["summary"] = copy.deepcopy(summary or {})
            self._state["plan_steps"] = _serialize_plan_steps(plan)
            self._state["session_status"] = "finished"
            self._state["updated_at"] = time.time()
            self._append_event("session_finished")

    def _summarize_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        keep = {
            "type",
            "x",
            "y",
            "target_x",
            "target_y",
            "keys",
            "text",
            "element_id",
            "app_name",
            "skill_id",
            "skill_name",
            "seconds",
            "clicks",
            "axis",
        }
        out: Dict[str, Any] = {}
        for key in keep:
            if key in action and action.get(key) is not None:
                out[key] = action.get(key)
        if action.get("type") == "macro_actions":
            out["actions"] = [
                self._summarize_action(item) for item in (action.get("actions") or []) if isinstance(item, dict)
            ][:6]
        return out
