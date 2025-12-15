# cua_agent (Computer Use Agent)

This repo is organized as a small multi-package workspace:

- `cua_agent/`: OS-agnostic core (planner, orchestrator loop, memory, policies, prompts/tool mapping).
- `macos_cua_agent/`: macOS adapter implementing the "computer" capabilities (screen capture, HID, Accessibility, AppleScript browser, permission checks).
- `windows_cua_agent/`: Windows adapter (screen capture, SendInput HID, PowerShell shell sandbox, CDP browser; UIA semantic grounding + Phantom Mode via `comtypes`, with OCR/blob fallback when UIA is unavailable).

**What's inside (core + adapter)**
- Orchestrator loop with planner + structured reflection, stagnation detection, and auto-replanning; episodes are summarized and logged.
- Grounding from an accessibility/UI tree with numbered Set-of-Mark overlays; OCR/blob fallback when semantic trees are unavailable; pHash/SSIM change detection on logical-resolution captures.
- Action execution via an adapter-provided computer implementation; adapters may offer semantic (Accessibility/UIA) paths, HID paths, browser ops, and sandboxed shell ops.
- Memory layer storing episodes/logs/semantic notes and procedural skills with semantic hints; skills are retrieved via embeddings/keywords.
- Safety rules from `cua_agent/policies/safety_rules.yaml`.

**Requirements**
- Python 3.11+; install deps with `pip install -r requirements.txt`.
- Optional: `brew install tesseract` to improve OCR for the visual fallback path.
- OpenRouter account/key to drive the planner, cognitive core, and reflector models; without a key the agent runs in noop/stub mode.

**Run**
- Windows: `python -m windows_cua_agent.main` (if `ENABLE_HID=true` and `WINDOWS_AUTO_ELEVATE=true`, the agent may request elevation via UAC; for browser ops, launch Chrome/Edge with `--remote-debugging-port=9222`)
- macOS: `python -m macos_cua_agent.main`
- Core entrypoint (auto-selects adapter by OS, or override): `python -m cua_agent` (or `python -m cua_agent --adapter windows_cua_agent`)

**Setup**
- Create a `.env` with your keys and toggles (see `.env.example`).
- Install deps with `pip install -r requirements.txt`.

**Testing**
- `pytest`
