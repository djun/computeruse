# macos_cua_agent

Prototype implementation of the Computer Use Agent described in `projecto.md`. The agent now runs through an Orchestrator with planning (planner model via OpenRouter), episodic memory, and the existing See-Think-Decide-Act-Verify loop on macOS Monterey. By default it operates in a dry-run mode (no HID injection and no OpenRouter calls) so it can be inspected safely.
The cognitive core calls Claude Opus 4.5 via OpenRouter using a custom `computer` tool (OpenAI-style function calling), not Anthropic's official `computer_20xx` beta.

## Quickstart
- Python 3.11+ on macOS Monterey; grant Screen Recording and Accessibility permissions to your terminal once you enable HID control.
- Create a virtualenv and install dependencies: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Add a `.env` with your OpenRouter key and any overrides you need (see below).
- Run the loop: `python -m macos_cua_agent.main` (first run uses placeholder frames and a stubbed cognitive core).
  - The CLI will prompt you for a task description. The agent runs until it halts (noop/limits) or you hit `Ctrl+C`, then you can enter another prompt or press Enter to quit.

## Configuration
Environment variables (place in `.env`):
- Computer control uses Claude Opus 4.5 via OpenRouter: set `USE_OPENROUTER=true`, `OPENROUTER_API_KEY`, and optionally `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`) and `OPENROUTER_MODEL` (default `anthropic/claude-opus-4.5`).
- Planner runs through an LLM client (default OpenRouter); configure with `PLANNER_API_KEY`, `PLANNER_BASE_URL`, and `PLANNER_MODEL` (default `anthropic/claude-3.5-sonnet`).
- Reflection/verifier model is separate but also uses OpenRouter; set `REFLECTOR_API_KEY` (defaults to `OPENROUTER_API_KEY`), `REFLECTOR_BASE_URL` (defaults to `OPENROUTER_BASE_URL`), and `REFLECTOR_MODEL` (default `openai/gpt-5.1`). Toggle via `ENABLE_REFLECTION` (default true).
- Semantic memory embeddings are optional: `ENABLE_EMBEDDINGS=true`, `EMBEDDING_API_KEY`/`OPENAI_API_KEY`, `EMBEDDING_BASE_URL` (default OpenAI), and `EMBEDDING_MODEL` (default `text-embedding-3-small`).
- `ENABLE_HID`=true to send real mouse/keyboard events via `pyautogui`.
- `ENABLE_SEMANTIC`=true to route actions through the semantic driver (AppleScript-based focus/insert/save).
- `ENABLE_SHELL`=true to enable sandboxed shell execution (default false). Configure with `SHELL_WORKSPACE_ROOT` (default `.agent_shell`), `SHELL_MAX_RUNTIME_S` (default 10), and `SHELL_MAX_OUTPUT_BYTES` (default 65536).
- `STRICT_STEP_COMPLETION`=true to enforce strict step verification (default true).
- `REASONING_EFFORT` (high/medium/low) and `REASONING_MAX_TOKENS` for reasoning models.
- `BROWSER_SCRIPT_TIMEOUT_S` and `BROWSER_NAVIGATION_TIMEOUT_S` for browser interaction safety.
- `MAX_STEPS`, `MAX_FAILURES`, `MAX_WALL_CLOCK_SECONDS`, `VERIFY_DELAY_MS`, `LOG_LEVEL`, `ENCODE_FORMAT`.
- `MEMORY_ROOT` to change where episodic/semantic logs are persisted (default `.agent_memory`).

Policy configuration lives in `macos_cua_agent/policies/safety_rules.yaml`. Extend block/allow/HITL lists to reflect your risk posture.

## Safety and Permissions
- HID control stays disabled until `ENABLE_HID=true`; actions are logged instead.
- OpenRouter calls stay disabled until `USE_OPENROUTER=true` and `OPENROUTER_API_KEY` is set.
- macOS TCC permissions (Screen Recording, Accessibility) must be granted to your terminal when you turn on HID control or screenshots.

## Testing
- Run regression tests: `pytest macos_cua_agent/tests`
- `test_coordinates.py` checks Retina conversion helpers. Benchmark tests are skipped until a macOS interactive harness is available.
