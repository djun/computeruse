# cua_agent (Computer Use Agent)

This repo is organized as a small multi-package workspace:

- `cua_agent/`: OS-agnostic core (planner, orchestrator loop, memory, policies, prompts/tool mapping).
- `macos_cua_agent/`: macOS adapter implementing the "computer" capabilities (screen capture, HID, Accessibility, AppleScript browser, permission checks).
- `windows_cua_agent/`: Windows adapter (screen capture, SendInput HID, PowerShell shell sandbox, CDP browser; UIA semantic grounding + Phantom Mode via `comtypes`, with OCR/blob fallback when UIA is unavailable).

Architecture boundary: core business logic lives in `cua_agent/`; adapter packages keep OS-specific drivers/integration code only.

**What's inside (core + adapter)**
- Orchestrator loop with planner + structured reflection, stagnation detection, and auto-replanning; episodes are summarized and logged.
- Grounding from an accessibility/UI tree with numbered Set-of-Mark overlays; visual fallback uses optional detector backend (`ultralytics`), OCR, and blob proposals when semantic trees are unavailable; pHash/SSIM change detection on logical-resolution captures.
- Optional live debug dashboard (FastAPI) with real-time screenshot preview, next-click crosshair, cognitive trace, and plan status stream.
- Action execution via an adapter-provided computer implementation; adapters may offer semantic (Accessibility/UIA) paths, HID paths, browser ops, and sandboxed shell ops.
- Memory layer storing episodes/logs/semantic notes and procedural skills with semantic hints; skills are retrieved via embeddings/keywords, with optional local vector indexing via Chroma.
- Safety rules from `cua_agent/policies/safety_rules.yaml`.
- HITL terminal confirmations (`[y/N]`) for policy-marked high-risk actions when running in an interactive shell.
- Sensitive-screen redaction pass (OCR + blur) before screenshot frames are sent to model APIs.
- Execution profiles to constrain tooling by context: `local_gui`, `remote_cli`, or `hybrid`.

**Requirements**
- Python 3.11+; install base deps with `pip install -r requirements.txt`.
- Optional feature deps live in `requirements-optional.txt` (`pip install -r requirements-optional.txt`); each is gated by a feature flag and the core degrades gracefully without it:
  - `chromadb` — local skill vector indexing (`ENABLE_CHROMA_SKILLS=true`).
  - `fastapi` + `uvicorn` — live debug dashboard (`ENABLE_DEBUG_DASHBOARD=true`).
  - `ultralytics` — detector-assisted visual grounding (`ENABLE_VISUAL_DETECTOR=true`).
- Optional: `brew install tesseract` to improve OCR for the visual fallback path.
- OpenRouter account/key to drive the planner, cognitive core, and reflector models; without a key the agent runs in noop/stub mode.

**Run**
- Windows: `python -m windows_cua_agent.main` (if `ENABLE_HID=true` and `WINDOWS_AUTO_ELEVATE=true`, the agent may request elevation via UAC; `WINDOWS_CYBORG_MODE=true` avoids CDP and drives Chrome via UIA/HID; for full `browser` tool support, launch Chrome/Edge with `--remote-debugging-port=9222` on a non-default profile)
- macOS: `python -m macos_cua_agent.main`
- Core entrypoint (auto-selects adapter by OS, or override): `python -m cua_agent` (or `python -m cua_agent --adapter windows_cua_agent`)

**Setup**
- Create a `.env` with your keys and toggles (see `.env.example`).
- Install deps with `pip install -r requirements.txt`.
- Choose an execution profile:
  - `EXECUTION_PROFILE=local_gui`: computer/browser enabled, shell blocked.
  - `EXECUTION_PROFILE=remote_cli`: shell enabled, GUI/browser blocked.
  - `EXECUTION_PROFILE=hybrid`: both enabled.
- Optional debugging UX:
  - `ENABLE_DEBUG_DASHBOARD=true`
  - `DEBUG_DASHBOARD_HOST=127.0.0.1`
  - `DEBUG_DASHBOARD_PORT=8765`
- Strict post-action validation (recommended):
  - `STRICT_POST_ACTION_STATE_CHANGE=true`
- Procedural fast-path (the keyword path needs no extra dependency):
  - `ENABLE_FAST_PATH_SKILLS=true`
  - Optional vector retrieval on top (heavy dep, from `requirements-optional.txt`):
    `ENABLE_EMBEDDINGS=true`, `ENABLE_CHROMA_SKILLS=true`, `CHROMA_PERSIST_DIR=.agent_memory/chroma`

**Testing**
- `pytest`
