from __future__ import annotations

import json
import subprocess
import ast
import time
from typing import Optional

from macos_cua_agent.agent.state_manager import ActionResult
from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger


class BrowserDriver:
    """
    Advanced semantic browser interaction driver.
    Supports React/Angular form filling, DOM tree extraction, and arbitrary JS execution.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)

    def execute_browser_action(self, action: dict) -> ActionResult:
        cmd = action.get("command")
        app = action.get("app_name", "Safari")
        
        # Dispatch table for new and legacy commands
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

    def _get_page_content(self, app_name: str) -> ActionResult:
        js_payload = "document.body.innerText"
        # Wrap simple return in the expected JSON structure for consistency
        wrapped_js = f"""
        (function() {{
            try {{
                var content = {js_payload};
                return JSON.stringify({{status: "success", result: content}}); 
            }} catch (e) {{
                return JSON.stringify({{status: "error", message: e.toString()}}); 
            }}
        }})();
        """
        return self._run_js_with_result(app_name, wrapped_js, "get_page_content")

    def _get_links(self, app_name: str) -> ActionResult:
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

    def _navigate(self, app_name: str, url: Optional[str]) -> ActionResult:
        if not url:
            return ActionResult(success=False, reason="url required")

        # Secure navigation using parameterized AppleScript (avoids injection)
        # Navigation is special as it doesn't return a JS result in the same way
        if "Safari" in app_name:
            script = """
            on run argv
                set targetUrl to item 1 of argv
                tell application "Safari"
                    if (count of documents) = 0 then make new document
                    set URL of front document to targetUrl
                end tell
            end run
            """
        else:
            # Chrome/Brave/Edge/etc
            script = f"""
            on run argv
                set targetUrl to item 1 of argv
                tell application "{app_name}"
                    if (count of windows) = 0 then make new window
                    set URL of active tab of front window to targetUrl
                end tell
            end run
            """
        result = self._run_arg_applescript(
            app_name, script, [url], "navigate", timeout=self.settings.browser_navigation_timeout_s
        )
        if result.success:
            self._wait_for_dom_ready(app_name)
        return result

    def _fill_form(self, app_name: str, selector: str, value: str) -> ActionResult:
        """
        Fills a form input using a React-compatible native setter hack.
        Dispatches 'input' and 'change' events to ensure state tracking updates.
        """
        if not selector or value is None:
            return ActionResult(success=False, reason="fill_form requires selector and value")

        selector_js = json.dumps(selector)
        value_js = json.dumps(value)

        # JS Payload: Handles Native Setter + Event Dispatching + Error Handling
        js_payload = f"""
        (function() {{
            function setNativeValue(el, val) {{
                // Prefer element-specific setter to avoid React/Angular proxy pitfalls
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

                // React 15/16 Tracker Hack
                const tracker = el._valueTracker;
                if (tracker) tracker.setValue(val);
            }}

            try {{
                var el = document.querySelector({selector_js});
                if (!el) return JSON.stringify({{status: "error", message: "Element not found: " + {selector_js}}});

                setNativeValue(el, {value_js});

                // 3. Dispatch Bubbling Events
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

    def _click_element(self, app_name: str, selector: str) -> ActionResult:
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

    def _get_dom_tree(self, app_name: str) -> ActionResult:
        """
        Returns a distilled JSON representation of the DOM, including shadow roots.
        Strips scripts, styles, and non-interactive nodes to save context tokens.
        """
        js_payload = """
        (function() {{
            const INTERACTIVE = new Set(["a","button","input","select","textarea","option","label"]);
            const CONTAINERS = new Set(["div","span","ul","li","ol","section","header","footer","main","nav","form"]);
            const PRUNE = new Set(["script","style","noscript","meta","link","svg","path"]);

            function distill(node, depth) {{
                if (depth > 6) return null; // Hard depth limit

                // Text Nodes: keep if not empty
                if (node.nodeType === 3) {{
                    var txt = node.textContent.trim();
                    return (txt.length > 0)? txt.substring(0, 200) : null;
                }}
                
                // Element or shadow/document fragments only
                if (node.nodeType !== 1 && node.nodeType !== 11) return null;
                
                var tagName = node.tagName ? node.tagName.toLowerCase() : "#fragment";
                if (PRUNE.has(tagName)) return null;

                var isInteractive = INTERACTIVE.has(tagName);
                var isContainer = CONTAINERS.has(tagName) || tagName === "#fragment" || tagName === "#shadow-root";
                if (!isInteractive && !isContainer) return null;
                
                var rect = node.getBoundingClientRect ? node.getBoundingClientRect() : {{width:1,height:1,x:0,y:0}};
                if (rect.width === 0 || rect.height === 0) return null; // Skip invisible

                var obj = {{ tag: tagName }};
                
                // Extract Semantic Attributes
                if (node.id) obj.id = node.id;
                if (node.className) obj.class = node.className; // informative but maybe verbose
                if (tagName === 'a' && node.href) obj.href = node.href;
                if (tagName === 'input') {{
                    obj.type = node.type;
                    obj.value = node.value;
                    obj.name = node.name;
                    obj.placeholder = node.placeholder;
                }}
                if (node.getAttribute && node.getAttribute('role')) obj.role = node.getAttribute('role');
                if (node.getAttribute && node.getAttribute('aria-label')) obj.ariaLabel = node.getAttribute('aria-label');

                // Visible geometry
                obj.rect = {{
                    x: rect.x,
                    y: rect.y,
                    width: rect.width,
                    height: rect.height
                }};

                // Include concise text for better grounding
                var text = (node.innerText || "").trim();
                if (text && text.length > 0) {{
                    obj.text = text.substring(0, 200);
                }}
                
                // Recursion, including shadow DOM
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
                else if (!isInteractive) {{
                    // Prune empty structural containers without content
                    return null;
                }}
                
                return obj;
            }
            return JSON.stringify(distill(document.body, 0));
        })();
        """
        return self._run_js_with_result(app_name, js_payload, "get_dom_tree")

    def _run_arbitrary_js(self, app_name: str, js_code: str) -> ActionResult:
        """Executes agent-provided JS wrapped in a safety harness (bounded wait for Promises)."""
        if not js_code:
            return ActionResult(success=False, reason="no JS code provided")

        promise_wait_ms = max(500, min(int(self.settings.browser_script_timeout_s * 1000) - 500, 4000))
             
        # Wrap in IIFE with try/catch, Promise support, and JSON return
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

    def _handle_nav_command(self, app_name: str, cmd: str) -> ActionResult:
        js = ""
        if cmd == "go_back":
            js = "history.back()"
        elif cmd == "go_forward":
            js = "history.forward()"
        elif cmd == "reload":
            js = "location.reload()"
        
        # These commands typically don't return a value we care about, 
        # but we wrap them to match the _run_js_with_result pattern for consistency
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
        result = self._run_js_with_result(app_name, wrapped, cmd)
        if result.success:
            self._wait_for_dom_ready(app_name)
        return result

    def _run_js_with_result(
        self, app_name: str, js_code: str, label: str, timeout: float | None = None
    ) -> ActionResult:
        """
        Executes the JS payload via AppleScript and parses the JSON string result.
        Handles the browser-specific 'do JavaScript' vs 'execute' syntax.
        """
        script = self._build_applescript(app_name, js_code)
        # Note: Re-using the existing _run_arg_applescript helper from the base implementation
        # We pass empty args because the JS code is embedded in the script by _build_applescript
        effective_timeout = timeout if timeout is not None else self.settings.browser_script_timeout_s
        result = self._run_arg_applescript(app_name, script, [], label, timeout=effective_timeout)
        
        if not result.success:
            return result
            
        raw_output = result.metadata.get("output", "").strip()
        
        # OSA quirk: sometimes returns the string wrapped in quotes, sometimes not.
        if raw_output.startswith('"') and raw_output.endswith('"'):
            # Basic unescaping for OSA string return
            try:
                raw_output = ast.literal_eval(raw_output)
            except:
                pass

        try:
            data = json.loads(raw_output)
            if isinstance(data, dict) and data.get("status") == "error":
                return ActionResult(success=False, reason=f"JS Error: {data.get('message')}", metadata=data)
            return ActionResult(success=True, reason=label, metadata={"data": data})
        except json.JSONDecodeError:
            # Fallback if the JS didn't return valid JSON
            return ActionResult(success=True, reason=f"{label} (raw return)", metadata={"raw": raw_output})

    def _build_applescript(self, app_name: str, js_code: str) -> str:
        """Generates the correct AppleScript for the specific browser."""
        # Escape backslashes and double quotes for AppleScript string literal
        safe_js = js_code.replace('\\', '\\\\').replace('"', '\\"')
        
        if "Chrome" in app_name or "Brave" in app_name or "Edge" in app_name:
            # Chrome uses 'execute active tab javascript'
            return f"""
            tell application "{app_name}"
                if (count of windows) = 0 then return "no_window"
                tell active tab of front window
                    execute javascript "{safe_js}"
                end tell
            end tell
            """
        elif "Safari" in app_name:
            # Safari uses 'do JavaScript' in document
            return f"""
            tell application "Safari"
                if (count of documents) = 0 then return "no_document"
                do JavaScript "{safe_js}" in document 1
            end tell
            """
        else:
            # Default fallback
            return f"""
            tell application "{app_name}"
                execute front window's active tab javascript "{safe_js}"
            end tell
            """

    def get_current_url(self, app_name: str) -> Optional[str]:
        """
        Returns the current tab/document URL for the given browser, or None on failure.
        Used to enforce policy checks before executing JS.
        """
        if "Safari" in app_name:
            script = """
            tell application "Safari"
                if (count of documents) = 0 then return ""
                return URL of front document
            end tell
            """
        else:
            script = f"""
            tell application "{app_name}"
                if (count of windows) = 0 then return ""
                return URL of active tab of front window
            end tell
            """

        result = self._run_arg_applescript(
            app_name, script, [], "get_current_url", timeout=self.settings.browser_script_timeout_s
        )
        if result.success:
            output = (result.metadata or {}).get("output", "").strip()
            return output or None
        return None

    def _run_arg_applescript(
        self, app_name: str, script: str, args: list[str], label: str, timeout: float | None = None
    ) -> ActionResult:
        try:
            # self.logger.debug("Running Arg AppleScript for %s", label)
            cmd = ["osascript", "-e", script] + args
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout
            )
            if result.returncode == 0:
                return ActionResult(
                    success=True, 
                    reason=f"{label} success", 
                    metadata={"output": result.stdout.strip()}
                )
            return ActionResult(
                success=False, 
                reason=f"{label} failed: {result.stderr.strip()}"
            )
        except subprocess.TimeoutExpired:
            return ActionResult(
                success=False,
                reason=f"{label} timed out after {timeout}s" if timeout else f"{label} timed out",
            )
        except Exception as e:
            return ActionResult(success=False, reason=f"{label} exception: {str(e)}")

    def _clean_osascript_output(self, output: str) -> str:
        output = output.strip()
        if output.startswith('"') and output.endswith('"'):
            # Remove wrapping quotes often added by OSA
            return output[1:-1].replace('\\"', '"')
        return output

    def _wait_for_dom_ready(self, app_name: str, timeout: float = 5.0, interval: float = 0.25) -> None:
        """
        Best-effort polling for DOM readiness after navigation to reduce race conditions.
        Does not raise on timeout; avoids blocking the agent indefinitely.
        """
        deadline = time.time() + timeout
        readiness_probe = """
        (function() {
            try {
                return JSON.stringify({status: "success", result: document.readyState});
            } catch (e) {
                return JSON.stringify({status: "error", message: e.toString()});
            }
        })();
        """
        while time.time() < deadline:
            probe = self._run_js_with_result(app_name, readiness_probe, "ready_state")
            if probe.success:
                data = probe.metadata.get("data") if probe.metadata else None
                if isinstance(data, dict):
                    state = data.get("result") or data.get("data", {}).get("result")
                    if state in ("complete", "interactive"):
                        return
            time.sleep(interval)
