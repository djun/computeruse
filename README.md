# macos_cua_agent

Computer-use agent for macOS with planning, visual/semantic grounding, reflection, and reusable skills. It talks to OpenRouter using a custom tool schema and defaults to a safe dry-run if you do not provide keys or HID access.

**What’s inside**
- Orchestrator with planner + structured reflection, stagnation detection, and auto-replanning; episodes are summarized and logged.
- Grounding from the Accessibility tree with numbered Set-of-Mark overlays; OCR/blob fallback when AX is missing; pHash/SSIM change detection on logical-resolution captures.
- Action engine that prefers semantic AppleScript/AX focus and phantom typing/clicking, then HID; also exposes browser, notebook, and sandboxed shell tools.
- Memory layer storing episodes/logs/semantic notes and procedural skills with semantic hints; skills are retrieved via embeddings/keywords and re-targeted at runtime.
- Safety from `macos_cua_agent/policies/safety_rules.yaml`, including exclusion zones, JS guardrails, and permission health checks.
- Defaults keep HID off, shell off, and embeddings off; OpenRouter calls stub if `OPENROUTER_API_KEY` is absent.

**Requirements**
- macOS Monterey or newer; grant Screen Recording and Accessibility permissions to your terminal once you enable HID/semantic control.
- Python 3.11+; install deps with `pip install -r requirements.txt`.
- Optional: `brew install tesseract` to improve OCR for the visual fallback path.
- OpenRouter account/key to drive the planner, cognitive core, and reflector models; without a key the agent runs in noop/stub mode.

**Setup**
- `python -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`
- Create a `.env` with your keys and toggles (example below).
- Run `python -m macos_cua_agent.main` and enter a task when prompted.

Example `.env`:
```
OPENROUTER_API_KEY=sk-...
USE_OPENROUTER=true
PLANNER_MODEL=anthropic/claude-3.5-sonnet
REFLECTOR_MODEL=openai/gpt-5.1
ENABLE_HID=false
ENABLE_SEMANTIC=true
ENABLE_SHELL=false
ENABLE_EMBEDDINGS=false
```

**Key configuration**
- `OPENROUTER_*` sets the main cognitive core (Claude Opus 4.5 by default); `PLANNER_*` and `REFLECTOR_*` override planner/verifier models or base URLs.
- `ENABLE_HID` (false) sends real mouse/keyboard events; keep it off for dry-run. `ENABLE_SEMANTIC` (true) keeps the AppleScript/AX path on by default.
- `ENABLE_SHELL` (false) allows sandboxed commands under `SHELL_WORKSPACE_ROOT` (default `.agent_shell`); cap runtime/output via `SHELL_MAX_RUNTIME_S`/`SHELL_MAX_OUTPUT_BYTES`; extend the allowlist with `SHELL_ALLOWED_COMMANDS`.
- `ENABLE_EMBEDDINGS` (false) turns on vector search for semantic memory/skills using `EMBEDDING_API_KEY`/`OPENAI_API_KEY`, `EMBEDDING_BASE_URL`, and `EMBEDDING_MODEL`.
- Observation and loop tuning: `ENCODE_FORMAT`, `VERIFY_DELAY_MS`, `SETTLE_DELAY_MS`, `SSIM_CHANGE_THRESHOLD`, `MAX_STEPS`, `MAX_FAILURES`, `MAX_WALL_CLOCK_SECONDS`, `REASONING_EFFORT` or `REASONING_MAX_TOKENS`; browser timeouts via `BROWSER_SCRIPT_TIMEOUT_S` and `BROWSER_NAVIGATION_TIMEOUT_S`.
- `MEMORY_ROOT` sets persistence location (default `.agent_memory` for episodes/semantic/skills/logs); adjust safety rules in `macos_cua_agent/policies/safety_rules.yaml`.

**Data and safety**
- `.agent_memory` stores episodic logs, semantic notes, and procedural skills (used via `run_skill`); `.agent_shell` is the sandbox workspace for shell actions.
- Health checks fail fast if Screen Recording/Accessibility permissions are missing; grant them before enabling HID/semantic control.

**Testing**
- `pytest macos_cua_agent/tests`
