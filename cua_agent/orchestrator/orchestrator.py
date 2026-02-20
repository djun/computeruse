"""Session orchestration logic (OS-agnostic)."""

from __future__ import annotations

import time
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from cua_agent.agent.cognitive_core import CognitiveCore
from cua_agent.agent.state_manager import ActionResult, StateManager, VerificationContract
from cua_agent.computer.adapter import ComputerAdapter
from cua_agent.memory.memory_manager import Episode, MemoryManager
from cua_agent.observability import LiveDebugDashboard
from cua_agent.orchestrator.planner_client import PlannerClient
from cua_agent.orchestrator.planning import Plan, Step
from cua_agent.orchestrator.reflection import Reflector
from cua_agent.utils.config import Settings
from cua_agent.utils.logger import get_logger
from cua_agent.utils.ax_pruning import prune_ax_tree_for_prompt
from cua_agent.utils.ax_utils import draw_som_overlay, flatten_nodes_with_frames

SKILL_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class Orchestrator:
    """Coordinates planning, execution, and memory."""

    def __init__(self, settings: Settings, computer: ComputerAdapter) -> None:
        self.settings = settings
        self.computer = computer
        self.logger = get_logger(__name__, level=settings.log_level)
        self.cognitive_core = CognitiveCore(settings, computer)
        self.memory = MemoryManager(settings)
        self.planner = PlannerClient(settings)
        self.reflector = Reflector(settings)
        self.dashboard = LiveDebugDashboard(settings, self.logger)
        self.display = computer.display
        self.global_hotkeys = getattr(computer, "global_hotkeys", set())

        if not settings.enable_hid:
            self.logger.warning("ENABLE_HID is false; actions will run in dry-run mode (no real input).")

    def run_task(self, user_prompt: str) -> dict:
        self.computer.run_health_checks(self.settings, logger=self.logger)
        # Capture context first for grounded planning
        initial_frame, initial_hash = self.computer.capture_with_hash()

        bootstrap_history: List[str] = []
        if self.settings.enable_fast_path_skills:
            fast_path = self._attempt_fast_path(user_prompt, initial_frame, initial_hash)
            if fast_path.get("success") and fast_path.get("summary"):
                return fast_path["summary"]
            if fast_path.get("attempted"):
                bootstrap_history.extend(fast_path.get("history") or [])
                initial_frame = fast_path.get("frame") or initial_frame
                initial_hash = fast_path.get("hash") or initial_hash

        prior_episodes = self.memory.list_episodes()
        prior_semantic = self.memory.search_semantic(user_prompt, top_k=5)

        # Plan with full context
        plan = self.planner.make_plan(user_prompt, prior_episodes, prior_semantic, screenshot_b64=initial_frame)
        self.logger.info("Plan %s created with %s steps", plan.id, len(plan.steps))

        summary = self._run_session(
            user_prompt=user_prompt,
            plan=plan,
            initial_frame=initial_frame,
            initial_hash=initial_hash,
            bootstrap_history=bootstrap_history,
        )
        return summary

    def _run_session(
        self,
        user_prompt: str,
        plan: Optional[Plan] = None,
        initial_frame: str | None = None,
        initial_hash: str | None = None,
        bootstrap_history: Optional[List[str]] = None,
    ) -> dict:
        state = StateManager(
            max_steps=self.settings.max_steps,
            max_failures=self.settings.max_failures,
            max_wall_clock_seconds=self.settings.max_wall_clock_seconds,
        )

        if plan:
            serialized_plan = "; ".join([f"{s.id}:{s.description}({s.status})" for s in plan.steps])
            state.history.append(f"plan_init:{serialized_plan}")

        state.history.append(f"user_prompt:{user_prompt}")
        if bootstrap_history:
            state.history.extend([f"bootstrap:{line}" for line in bootstrap_history[-30:]])

        if initial_frame:
            current_frame = initial_frame
            current_hash = initial_hash or self.computer.hash_base64(initial_frame)
        else:
            current_frame, current_hash = self.computer.capture_with_hash()

        state.record_observation(
            current_frame, changed=True, note=f"initial capture for: {user_prompt}", phash=current_hash
        )
        self.dashboard.start_session(user_prompt, plan, current_frame, self.display)

        last_action_sig: str | None = None
        action_sig_history: List[str] = []
        repeat_without_change = 0
        repeat_same_action = 0
        repeat_info_for_model: dict | None = None
        hotkey_counts: dict[tuple[str, ...], int] = {}
        hint_count = 0
        max_hints = 3
        plan_revision_count = 0
        max_plan_revisions = 3
        global_hotkeys = self.global_hotkeys
        low_change_streak = 0
        PHASH_STATIC_THRESHOLD = 4  # Hamming distance; increased to ignore noise
        STAGNATION_LIMIT = 5        # consecutive frames with minimal change
        current_tags: List[Dict[str, Any]] = []
        step_trace: List[Dict[str, Any]] = []
        active_step_id = plan.current_step().id if plan and plan.current_step() else None
        force_vision_next_turn = True

        try:
            while not state.should_halt():
                if plan and plan_revision_count < max_plan_revisions and self._should_replan(
                    plan, state, repeat_same_action, repeat_without_change
                ):
                    revised_plan = self.planner.revise_plan(plan, state.history, current_frame)
                    if revised_plan:
                        if revised_plan.to_dict() != plan.to_dict():
                            plan_revision_count += 1
                            self.logger.info("Plan revised (auto); new current step index %s", revised_plan.current_step_index)
                            state.history.append(
                                f"plan_revised:auto:step_index={revised_plan.current_step_index}"
                            )
                            self.dashboard.push_event(
                                f"plan_revised:auto:step_index={revised_plan.current_step_index}"
                            )
                        plan = revised_plan
                        repeat_same_action = 0
                        repeat_without_change = 0
                        last_action_sig = None
                        repeat_info_for_model = None

                current_step = plan.current_step() if plan else None
                current_step_id = current_step.id if current_step else None
                if current_step_id != active_step_id:
                    step_trace = []
                    active_step_id = current_step_id
                include_visual_context = force_vision_next_turn
                
                # Context Compression
                if len(state.history) > 60:
                    self._compress_history(state)

                # Semantic Grounding: Fetch accessibility tree
                ax_tree = None
                if self.settings.enable_semantic:
                    ax_res = self.computer.get_active_window_tree(max_depth=4)
                    if ax_res.success:
                        raw_tree = ax_res.metadata.get("tree")
                        ax_tree = prune_ax_tree_for_prompt(raw_tree) if raw_tree else None
                        # self.logger.debug("Fetched AX tree for grounding")

                # Visual Fallback if AX failed or is empty
                if not ax_tree:
                    vision_elements = self.computer.detect_ui_elements(current_frame)
                    if vision_elements:
                        self.logger.info("Using visual grounding fallback, found %d elements", len(vision_elements))
                        ax_tree = {
                            "role": "AXWindow",
                            "title": "Visual Fallback",
                            "children": vision_elements,
                            "frame": {"x": 0, "y": 0, "w": self.display.logical_width, "h": self.display.logical_height}
                        }

                # Overlay Set-of-Mark tags onto the screenshot for grounding
                overlay_frame = current_frame
                current_tags = []
                if ax_tree and include_visual_context:
                    nodes = flatten_nodes_with_frames(ax_tree, max_nodes=40)
                    overlay_frame, current_tags = draw_som_overlay(current_frame, nodes, self.display)
                elif not include_visual_context:
                    overlay_frame = ""

                loop_state = self._format_loop_state(plan, state, repeat_same_action, repeat_without_change)
                loop_state["visual_context"] = "image" if include_visual_context else "text_only"
                
                # Retrieve relevant skills
                relevant_skills = []
                query_text = (current_step.description if current_step else "") or user_prompt or ""
                if query_text:
                    relevant_skills = self.memory.search_skills(query_text)

                action = self.cognitive_core.propose_action(
                    overlay_frame,
                    state.history,
                    include_visual_context=include_visual_context,
                    user_prompt=user_prompt,
                    repeat_info=repeat_info_for_model,
                    plan=plan,
                    current_step=current_step,
                    loop_state=loop_state,
                    ax_tree=ax_tree,
                    som_tags=current_tags,
                    relevant_skills=relevant_skills,
                )
                cognitive_trace = self._extract_cognitive_trace(action)
                if cognitive_trace:
                    self.dashboard.push_thought(cognitive_trace)

                if action.get("type") == "noop":
                    self.logger.info("Noop action requested; stopping loop. Reason: %s", action.get("reason"))
                    break
                
                if action.get("type") == "run_skill":
                    skill_ref = action.get("skill_id") or action.get("skill_name")
                    skill = self.memory.get_skill(skill_ref) if skill_ref else None
                    if not skill:
                        result = ActionResult(success=False, reason="skill not found")
                        state.record_action(action, result)
                        repeat_info_for_model = {
                            "count": repeat_same_action,
                            "action": repr(action),
                            "hint": "Requested skill not found; try a different approach or rebuild it.",
                        }
                        continue
                    requested_contract = (
                        action.get("verification") if isinstance(action.get("verification"), dict) else None
                    )
                    raw_skill_args = action.get("skill_args")
                    runtime_args = raw_skill_args if isinstance(raw_skill_args, dict) else {}
                    rendered_actions, resolved_args, missing_params = self._materialize_skill_actions(skill, runtime_args)
                    if missing_params:
                        result = ActionResult(
                            success=False,
                            reason="missing required skill args: " + ", ".join(sorted(missing_params)),
                        )
                        state.record_action(action, result)
                        repeat_info_for_model = {
                            "count": repeat_same_action,
                            "action": repr(action),
                            "hint": (
                                "Skill invocation missing required parameters: "
                                + ", ".join(sorted(missing_params))
                            ),
                        }
                        continue
                    self.memory.record_skill_usage(skill.id)
                    action = {
                        "type": "macro_actions",
                        "actions": rendered_actions,
                        "skill_id": skill.id,
                        "skill_name": skill.name,
                        "skill_args": resolved_args,
                    }
                    skill_contract = self._render_skill_verification_contract(skill, resolved_args)
                    if requested_contract:
                        action["verification"] = requested_contract
                    elif skill_contract:
                        action["verification"] = skill_contract
                action = self._strip_debug_fields(action)
                verification_contract = self._resolve_verification_contract(state, action, current_step)
                action["verification"] = verification_contract.to_dict()
                telemetry_before = self._collect_os_telemetry_snapshot(verification_contract)

                # Resolve overlay element references to coordinates
                resolved_ok = self._resolve_element_references(action, current_tags)
                if not resolved_ok:
                    result = ActionResult(success=False, reason="element_id not found")
                    state.record_action(action, result)
                    repeat_info_for_model = {"count": repeat_same_action, "action": repr(action), "hint": "element_id not found; request new inspect_ui. Grounding refreshed."}
                    self.dashboard.push_event("element_id resolution failed; forcing grounding refresh")
                    # Immediately refresh grounding so the next loop has fresh SoM/AX context
                    current_frame, current_hash, current_tags, ax_tree = self._refresh_grounding(state)
                    force_vision_next_turn = True
                    continue
                current_crosshair = self._extract_crosshair(action)
                self.dashboard.push_action(
                    action=action,
                    plan=plan,
                    current_step_id=current_step_id,
                    loop_state=loop_state,
                    frame_b64=overlay_frame,
                    crosshair=current_crosshair,
                )

                # Handle Notebook Operations (Internal State)
                if action.get("type") == "notebook_op":
                    op_action = action.get("action")
                    content = action.get("content", "")
                    source = action.get("source", "")
                    if op_action == "add_note":
                        state.add_note(content, source)
                        result = ActionResult(success=True, reason="note added")
                    elif op_action == "clear_notes":
                        state.clear_notebook()
                        result = ActionResult(success=True, reason="notes cleared")
                    else:
                        result = ActionResult(success=False, reason=f"unknown notebook op {op_action}")
                    
                    state.record_action(action, result)
                    # Notebook ops are fast internal updates; no need to wait or sleep significantly.
                    # But we should capture a frame if we want to verify something, though usually not needed.
                    # We continue to the next step immediately.
                    continue

                if action.get("type") == "key":
                    combo = tuple(sorted([k.lower() for k in action.get("keys") or []]))
                    count = hotkey_counts.get(combo, 0)
                    if count >= 2:
                        self.logger.info("Skipping hotkey %s; already executed %s times", "+".join(combo), count)
                        result = ActionResult(success=False, reason="hotkey deduped")
                        state.record_action(action, result)
                        repeat_info_for_model = {"count": repeat_same_action, "action": repr(action)}
                        continue
                    hotkey_counts[combo] = count + 1
                
                if action.get("type") == "open_app":
                    app_key = ("open_app", action.get("app_name", "").lower())
                    count = hotkey_counts.get(app_key, 0)
                    if count >= 1:  # Strict limit: don't open the same app twice in a short loop
                         self.logger.info("Skipping open_app %s; already executed", app_key[1])
                         result = ActionResult(success=False, reason="app open deduped")
                         state.record_action(action, result)
                         repeat_info_for_model = {"count": repeat_same_action, "action": repr(action)}
                         continue
                    hotkey_counts[app_key] = count + 1

                result = self.computer.execute(action)
                state.record_action(action, result)
                if action.get("execution") == "shell" and result.metadata:
                    stdout = (result.metadata.get("stdout") or "").strip()
                    stderr = (result.metadata.get("stderr") or "").strip()
                    if stdout:
                        state.history.append(f"shell_stdout:{stdout[:500]}")
                    if stderr:
                        state.history.append(f"shell_stderr:{stderr[:500]}")

                verification_outcome = self._run_verification_contract(
                    action=action,
                    contract=verification_contract,
                    current_frame=current_frame,
                    current_hash=current_hash,
                    ax_tree_before=ax_tree,
                    telemetry_before=telemetry_before,
                    global_hotkeys=global_hotkeys,
                    phash_static_threshold=PHASH_STATIC_THRESHOLD,
                )

                next_frame = verification_outcome["next_frame"]
                next_hash = verification_outcome["next_hash"]
                hash_distance = int(verification_outcome.get("hash_distance") or 0)
                ssim_score = verification_outcome.get("ssim_score")
                ax_tree_after = verification_outcome.get("ax_tree_after")
                ax_changed = bool(verification_outcome.get("ax_changed"))
                changed = bool(verification_outcome.get("changed"))
                obs_note = str(verification_outcome.get("note") or "")
                verification_passed = bool(verification_outcome.get("passed"))
                verification_sensor = str(verification_outcome.get("sensor") or verification_contract.sensor)
                verification_reason = str(verification_outcome.get("reason") or "")
                force_vision_next_turn = bool(verification_outcome.get("force_vision_next_turn", True))

                state.record_observation(
                    next_frame, changed, phash=next_hash, hash_distance=hash_distance, note=obs_note
                )
                self.dashboard.push_verification(
                    action=action,
                    changed=changed,
                    hash_distance=hash_distance,
                    ssim_score=ssim_score,
                    ax_changed=ax_changed,
                    note=obs_note,
                )

                if not verification_passed:
                    expected = verification_contract.expected_state or "state_change"
                    timeout = verification_contract.timeout_seconds
                    failure_reason = (
                        f"expected='{expected}' sensor={verification_sensor} timeout={timeout}s detail={verification_reason}"
                    )
                    state.record_verification_failure(failure_reason, action=action)
                    state.history.append(f"verification_contract_failed:{failure_reason}")
                    repeat_info_for_model = {
                        "count": repeat_same_action,
                        "action": repr(action),
                        "hint": (
                            "Validation failed. "
                            f"Expected '{expected}' via {verification_sensor} in {timeout}s. "
                            "Analyze the latest screen and replan around blockers (popup, error, captcha, wrong focus)."
                        ),
                    }
                    self.dashboard.push_event(f"verification_contract_failed:{verification_sensor}")
                    if plan and current_step:
                        plan.fail_current(f"Verification failed: {failure_reason}")

                    repeat_same_action = 0
                    repeat_without_change = 0
                    low_change_streak = 0
                    last_action_sig = None
                    current_frame = next_frame
                    current_hash = next_hash
                    continue

                state.history.append(
                    "verification_contract_ok:"
                    f"sensor={verification_sensor}:"
                    f"expected={verification_contract.expected_state or '-'}:"
                    f"{verification_reason or 'ok'}"
                )

                if plan and current_step:
                    self._append_step_trace(step_trace, action, result, changed)

                if action.get("type") == "macro_actions":
                    self._maybe_save_skill(action, result, current_step, user_prompt, changed)
                    if action.get("skill_id") and not result.success and plan and current_step:
                        skill_name = action.get("skill_name") or action.get("skill_id")
                        fail_msg = f"skill_macro_failed:{skill_name}:{result.reason}"
                        state.history.append(fail_msg)
                        plan.fail_current(f"Skill macro failed: {result.reason}")

                if not changed:
                    self.logger.info("No UI change detected after action: %s", action)
                else:
                    # UI changed; allow future hotkeys to be considered fresh.
                    hotkey_counts.clear()

                is_action_interactive = action.get("type") not in {"wait", "capture_only", "noop"}
                contract_requires_check = verification_contract.sensor != "none"
                critical_no_change = False
                if (
                    self.settings.strict_post_action_state_change
                    and contract_requires_check
                    and is_action_interactive
                    and result.success
                    and not changed
                    and self._requires_state_change(action, current_step)
                ):
                    critical_no_change = True
                    no_change_reason = self._format_no_change_reason(
                        action=action,
                        current_step=current_step,
                        hash_distance=hash_distance,
                        ssim_score=ssim_score,
                        ax_changed=ax_changed,
                    )
                    state.record_verification_failure(no_change_reason, action=action)
                    state.history.append(f"critical_no_change:{no_change_reason}")
                    repeat_info_for_model = {
                        "count": repeat_same_action,
                        "action": repr(action),
                        "hint": "Action expected a UI transition but state did not change. Re-inspect UI before retry.",
                    }
                    if plan and current_step:
                        plan.fail_current(f"Critical no-change: {no_change_reason}")
                    self.dashboard.push_event(f"critical_no_change:{action.get('type')}")
                    self.logger.warning("Critical no-change detected: %s", no_change_reason)

                if contract_requires_check:
                    if hash_distance <= PHASH_STATIC_THRESHOLD and is_action_interactive and not ax_changed:
                        low_change_streak += 1
                    else:
                        low_change_streak = 0
                else:
                    low_change_streak = 0

                # Signature used for repeat detection and hinting; set early so failure handling can reference it.
                action_sig = repr(action)
                action_sig_history.append(action_sig)

                # N-Gram Cycle Detection
                cycle_detected = False
                for k in range(2, 6):
                    if len(action_sig_history) >= 2 * k:
                        if action_sig_history[-k:] == action_sig_history[-2*k:-k]:
                            cycle_detected = True
                            self.logger.warning("Oscillatory loop detected (length %d)", k)
                            break

                if plan and current_step:
                    step_completed = False
                    reflection_result = None
                    deterministic_complete = self._deterministic_step_complete(current_step, ax_tree_after, changed)
                    if deterministic_complete:
                        step_completed = True
                    elif self.reflector.available:
                        reflection_result = self.reflector.evaluate_step(
                            current_step, state.history, next_frame, changed
                        )
                        step_completed = reflection_result.is_complete
                    elif not self.settings.strict_step_completion:
                        step_completed = self._heuristic_step_complete(current_step, action, result, changed)
                    
                    if reflection_result and reflection_result.status == "failed":
                        self.logger.warning("Step %s failed verification: %s (%s)", current_step.id, reflection_result.reason, reflection_result.failure_type)
                        state.history.append(f"reflector_fail:{reflection_result.failure_type}:{reflection_result.reason}")
                        self.dashboard.push_event(
                            f"reflector_fail:{reflection_result.failure_type}:{reflection_result.reason}"
                        )
                        
                        # TRIGGER DYNAMIC REPLANNING:
                        # Explicitly mark the step as failed so _should_replan catches it next loop.
                        plan.fail_current(f"Reflector blocked: {reflection_result.failure_type} - {reflection_result.reason}")

                        # Dedicated Recovery for Popups
                        failure_type_norm = (reflection_result.failure_type or "").lower()
                        reason_lower = (reflection_result.reason or "").lower()
                        if failure_type_norm in {"popup_blocking", "blocked_by_popup", "popup", "wrong_app"} or "popup" in reason_lower:
                            self._run_recovery(failure_type_norm or reflection_result.failure_type)

                        # Contingency: If we have recovery steps, suggest them to the agent via history
                        if current_step.recovery_steps:
                            recovery_msg = f"recovery_suggestion: Step failed. Try: {', '.join(current_step.recovery_steps)}"
                            state.history.append(recovery_msg)
                            self.logger.info(recovery_msg)
                            repeat_info_for_model = {
                                "count": repeat_same_action, 
                                "action": action_sig,
                                "hint": f"Verification failed. Try: {', '.join(current_step.recovery_steps)}"
                            }

                    if step_completed:
                        finished_id = current_step.id if current_step else None
                        self._maybe_synthesize_skill_from_trace(step_trace, current_step, user_prompt)
                        
                        # Visual Memory: Record the state at step completion
                        if self.reflector.available:
                            description = self.reflector.describe_image(next_frame)
                            if description:
                                self.memory.add_semantic_item(
                                    text=f"Visual state after step {finished_id}: {description}",
                                    metadata={"step_id": finished_id, "plan_id": plan.id if plan else ""}
                                )
                                self.logger.info("Saved visual memory for step %s", finished_id)

                        plan.advance()
                        step_trace = []
                        next_step = plan.current_step()
                        active_step_id = next_step.id if next_step else None
                        hotkey_counts.clear()
                        state.history.append(
                            f"plan_step_completed:{finished_id if finished_id is not None else 'unknown'}"
                        )
                        self.dashboard.push_event(
                            f"plan_step_completed:{finished_id if finished_id is not None else 'unknown'}"
                        )
                        self.logger.info("Advanced plan to step index %s", plan.current_step_index)
                        if not plan.current_step():
                            self.logger.info("Plan completed; stopping loop.")
                            break

                is_wait = action.get("type") == "wait"
                pending_break = False
                break_reason = ""
                if critical_no_change:
                    pending_break = True
                    break_reason = "critical_no_change"
                
                if cycle_detected and not pending_break:
                    pending_break = True
                    break_reason = "oscillatory_loop"

                if not is_wait and not pending_break:
                    if action_sig == last_action_sig:
                        repeat_same_action += 1
                        if repeat_same_action >= 3:  # Stricter limit (was 5)
                            pending_break = True
                            break_reason = f"repeat_same_action:{repeat_same_action}"
                    else:
                        repeat_same_action = 0
                    if not changed and action_sig == last_action_sig:
                        repeat_without_change += 1
                        if repeat_without_change >= 2:  # Stricter limit (was 3)
                            pending_break = True
                            break_reason = break_reason or "repeat_without_change"
                    else:
                        repeat_without_change = 0
                else:
                    repeat_same_action = 0
                    repeat_without_change = 0

                if not pending_break and low_change_streak >= STAGNATION_LIMIT:
                    pending_break = True
                    break_reason = "visual_stagnation"
                    state.history.append(f"visual_stagnation:hash_dist={hash_distance}")

                if action.get("type") == "key":
                    combo = tuple(sorted([k.lower() for k in action.get("keys") or []]))
                    if combo in global_hotkeys and not changed:
                        state.history.append(f"global_hotkey_no_effect:{'+'.join(combo)}")
                        repeat_info_for_model = {
                            "count": repeat_same_action,
                            "action": action_sig,
                            "hint": "Global hotkey had no visible effect; prefer clicking the visible app or window.",
                        }

                if pending_break:
                    if plan and current_step:
                        plan.fail_current(break_reason or "stuck")
                        state.history.append(f"plan_step_failed:{current_step.id}:{break_reason or 'stuck'}")
                    self.dashboard.push_event(f"loop_break:{break_reason or 'stuck'}")
                    state.record_stuck(break_reason or "stuck")

                    hint = ""
                    plan_changed = False
                    if self.reflector.available and hint_count < max_hints:
                        hint = self.reflector.suggest_hint(plan.current_step() if plan else None, state.history, next_frame)
                        if hint:
                            hint_count += 1
                            state.history.append(f"reflector_hint:{hint}")
                            repeat_info_for_model = {
                                "count": repeat_same_action,
                                "action": action_sig,
                                "hint": hint,
                            }
                            self.logger.info("Injected reflector hint to unblock: %s", hint)
                            pending_break = False

                    if plan and plan_revision_count < max_plan_revisions:
                        revised_plan = self.planner.revise_plan(plan, state.history, next_frame)
                        if revised_plan.to_dict() != plan.to_dict():
                            plan_revision_count += 1
                            state.history.append(
                                f"plan_revised:stuck:step_index={revised_plan.current_step_index}"
                            )
                            self.dashboard.push_event(
                                f"plan_revised:stuck:step_index={revised_plan.current_step_index}"
                            )
                            self.logger.info(
                                "Plan revised after stuck; new current step index %s", revised_plan.current_step_index
                            )
                            plan_changed = True
                        plan = revised_plan
                        if plan_changed:
                            pending_break = False

                    repeat_same_action = 0
                    repeat_without_change = 0
                    last_action_sig = None
                    low_change_streak = 0

                    if pending_break:
                        self.logger.info("Breaking loop: %s", break_reason or "stuck")
                        self.dashboard.push_event(f"session_break:{break_reason or 'stuck'}")
                        break

                    repeat_info_for_model = repeat_info_for_model or {"count": 0, "action": action_sig}
                    current_frame = next_frame
                    current_hash = next_hash
                    continue

                last_action_sig = action_sig
                if not repeat_info_for_model or "hint" not in repeat_info_for_model:
                    repeat_info_for_model = {"count": repeat_same_action, "action": last_action_sig}

                current_frame = next_frame
                current_hash = next_hash
        except KeyboardInterrupt:
            self.logger.info("Session cancelled by user.")

        summary = state.summary()
        if plan:
            summary["plan"] = plan.to_dict()
        self.logger.info("Session finished: %s", summary)
        self.dashboard.finish_session(summary, plan)
        self._persist_episode(user_prompt, state, plan)
        return summary

    def _should_replan(
        self, plan: Optional[Plan], state: StateManager, repeat_same_action: int, repeat_without_change: int
    ) -> bool:
        if not plan or not plan.current_step():
            return False
        if repeat_same_action >= 3 or repeat_without_change >= 2:
            return True
        if state.failure_count >= 3:
            return True
        if plan.current_step().status == "failed":
            return True
        return False

    def _attempt_fast_path(self, user_prompt: str, current_frame: str, current_hash: str) -> dict:
        """
        Try a high-confidence procedural skill before invoking planner/model.
        Falls back to normal planning when the cached macro does not complete the task.
        """
        match = self.memory.select_fast_path_skill(user_prompt, top_k=3)
        if not match:
            return {"attempted": False, "success": False}

        skill = match.skill
        fast_actions, resolved_args, missing_params = self._materialize_skill_actions(skill, {})
        if missing_params:
            self.logger.info(
                "Fast-path skipped for skill %s due to missing required params: %s",
                skill.id,
                ", ".join(sorted(missing_params)),
            )
            return {"attempted": False, "success": False}

        history: List[str] = [
            f"fast_path_candidate:skill={skill.id}:strategy={match.strategy}:score={match.score:.3f}"
        ]
        action = {
            "type": "macro_actions",
            "actions": fast_actions,
            "skill_id": skill.id,
            "skill_name": skill.name,
            "skill_args": resolved_args,
            "skill_fast_path": True,
        }
        skill_contract = self._render_skill_verification_contract(skill, resolved_args)
        if skill_contract:
            action["verification"] = skill_contract

        self.logger.info(
            "Fast-path attempting skill %s (%s) score=%.3f via %s",
            skill.name,
            skill.id,
            match.score,
            match.strategy,
        )
        self.memory.record_skill_usage(skill.id)
        result = self.computer.execute(action)
        history.append(f"fast_path_execute:{'success' if result.success else 'failed'}:{result.reason}")

        next_frame, next_hash = self.computer.capture_with_hash()
        changed = self.computer.has_changed(current_frame, next_frame)

        verified_success = False
        if result.success and self.reflector.available:
            probe_step = Step(
                id=0,
                description=f"Execute cached skill {skill.name}",
                success_criteria=user_prompt,
                status="in_progress",
            )
            reflection = self.reflector.evaluate_step(probe_step, history, next_frame, changed)
            history.append(
                f"fast_path_reflection:{reflection.status}:{reflection.failure_type}:{reflection.reason}"
            )
            verified_success = reflection.is_complete
        elif result.success:
            verified_success = bool(changed)
            history.append(f"fast_path_change_check:{'changed' if changed else 'unchanged'}")

        if verified_success:
            summary = {
                "steps": 1,
                "failures": 0,
                "history": list(history),
                "actions": [action],
                "observations": 1,
                "runtime_seconds": 0.0,
                "stuck_reasons": [],
                "fast_path": True,
                "fast_path_skill_id": skill.id,
                "fast_path_skill_name": skill.name,
            }
            self._persist_fast_path_episode(
                user_prompt=user_prompt,
                history=history,
                action=action,
                result=result,
                skill_id=skill.id,
                skill_name=skill.name,
            )
            self.logger.info("Fast-path succeeded with skill %s", skill.id)
            return {
                "attempted": True,
                "success": True,
                "summary": summary,
                "frame": next_frame,
                "hash": next_hash,
                "history": history,
            }

        failure_note = result.reason or "fast-path verification failed"
        history.append(f"fast_path_fallback:{failure_note}")
        try:
            self.memory.add_semantic_item(
                text=(
                    f"Fast-path skill failure for prompt '{user_prompt}': "
                    f"skill={skill.name} ({skill.id}), reason={failure_note}"
                ),
                metadata={"kind": "fast_path_failure", "skill_id": skill.id, "strategy": match.strategy},
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("Failed to persist fast-path failure memory: %s", exc)
        self.logger.info("Fast-path failed for skill %s, falling back to planner", skill.id)
        return {
            "attempted": True,
            "success": False,
            "summary": None,
            "frame": next_frame,
            "hash": next_hash,
            "history": history,
        }

    def _persist_fast_path_episode(
        self,
        user_prompt: str,
        history: List[str],
        action: Dict[str, Any],
        result: ActionResult,
        skill_id: str,
        skill_name: str,
    ) -> None:
        episode_id = f"fast-{int(time.time())}-{skill_id[:8]}"
        log_path = self.memory.logs_dir / f"{episode_id}.log"
        try:
            with log_path.open("w", encoding="utf-8") as handle:
                for line in history:
                    handle.write(f"{line}\n")
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Failed to write fast-path episode log: %s", exc)
            log_path = None

        episode = Episode(
            id=episode_id,
            created_at=time.time(),
            user_prompt=user_prompt,
            plan={
                "fast_path": True,
                "skill_id": skill_id,
                "skill_name": skill_name,
                "action": action,
            },
            outcome="success" if result.success else "mixed",
            summary=f"Fast-path skill {skill_name} ({skill_id}) executed: {result.reason}",
            tags=["desktop", "cua", "fast_path", "macro"],
            raw_log_path=str(log_path) if log_path else None,
        )
        try:
            self.memory.save_episode(episode)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Failed to persist fast-path episode: %s", exc)

    def _append_step_trace(self, step_trace: List[Dict[str, Any]], action: dict, result: ActionResult, changed: bool) -> None:
        action_type = action.get("type")
        if action_type in {"notebook_op", "inspect_ui", "probe_ui", "capture_only", "noop"}:
            return
        record: Dict[str, Any] = {
            "success": bool(result.success),
            "changed": bool(changed),
            "reason": result.reason,
        }
        if action_type == "macro_actions":
            record["action"] = {"type": "macro_actions", "actions": [dict(a) for a in action.get("actions") or []]}
        else:
            record["action"] = dict(action)
        step_trace.append(record)
        max_items = max(5, int(self.settings.dynamic_skill_capture_window))
        if len(step_trace) > max_items:
            del step_trace[:-max_items]

    def _maybe_synthesize_skill_from_trace(
        self,
        step_trace: List[Dict[str, Any]],
        current_step: Optional[Step],
        user_prompt: str,
    ) -> None:
        if not step_trace:
            return
        actions = self._extract_recovered_actions(step_trace)
        actions, parameters = self._build_composable_skill_payload(actions)
        min_actions = max(1, int(self.settings.dynamic_skill_min_actions))
        if len(actions) < min_actions:
            return

        had_failures = any(not item.get("success", True) for item in step_trace)
        if not had_failures and len(actions) < max(min_actions, 4):
            return

        try:
            name_seed = (current_step.description if current_step else "") or user_prompt or "task"
            name = (self._slugify(name_seed)[:40] + "-auto") if name_seed else f"macro-auto-{int(time.time())}"
            description = ""
            if current_step:
                description = current_step.success_criteria or current_step.description or ""
            if not description:
                description = user_prompt or "Auto synthesized procedural skill."
            if had_failures:
                description = f"{description} (Synthesized after recovery/self-healing.)"
            tags = ["macro", "synthesized"]
            if had_failures:
                tags.append("self_healed")
            if current_step and current_step.id:
                tags.append(f"step:{current_step.id}")
            verification_contract = self._derive_skill_verification_contract(current_step)
            self.memory.save_skill(
                name=name,
                description=description,
                actions=actions,
                tags=tags,
                source_prompt=user_prompt,
                plan_step_id=current_step.id if current_step else None,
                parameters=parameters,
                verification_contract=verification_contract,
            )
            self.logger.info(
                "Synthesized procedural skill '%s' with %d actions (had_failures=%s)",
                name,
                len(actions),
                had_failures,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("Failed to synthesize dynamic skill: %s", exc)

    def _extract_recovered_actions(self, step_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        last_failure_idx = -1
        for idx, event in enumerate(step_trace):
            if not event.get("success", True):
                last_failure_idx = idx
        candidate_window = step_trace[last_failure_idx + 1 :] if step_trace else []

        extracted: List[Dict[str, Any]] = []
        for event in candidate_window:
            if not event.get("success", True):
                continue
            raw_action = event.get("action") or {}
            if raw_action.get("type") == "macro_actions":
                for sub in raw_action.get("actions") or []:
                    cleaned_sub = self._sanitize_action_for_skill(sub)
                    if cleaned_sub:
                        extracted.append(cleaned_sub)
                continue
            cleaned = self._sanitize_action_for_skill(raw_action)
            if cleaned:
                extracted.append(cleaned)

        deduped: List[Dict[str, Any]] = []
        for act in extracted:
            if deduped and act == deduped[-1]:
                continue
            deduped.append(act)
        return deduped

    def _sanitize_action_for_skill(self, action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        allowed_types = {
            "left_click",
            "right_click",
            "double_click",
            "click_element",
            "mouse_move",
            "hover",
            "drag_and_drop",
            "select_area",
            "scroll",
            "scroll_to_element",
            "type",
            "fill_field",
            "click_and_type",
            "key",
            "open_app",
            "focus_window",
            "wait",
            "wait_for_element",
            "wait_for_idle",
        }
        action_type = action.get("type")
        if action_type not in allowed_types:
            return None

        keep_keys = {
            "type",
            "x",
            "y",
            "target_x",
            "target_y",
            "scroll_y",
            "clicks",
            "axis",
            "text",
            "keys",
            "seconds",
            "duration",
            "hold_delay",
            "app_name",
            "phantom_mode",
            "semantic_role",
            "semantic_label",
            "semantic_path",
            "role",
            "label",
            "path",
            "element_ref",
            "window_title",
            "submit",
            "clear",
            "timeout",
            "max_scrolls",
            "paste",
            "capture_selection",
            "click_type",
        }
        cleaned: Dict[str, Any] = {}
        for key in keep_keys:
            if key in action and action.get(key) is not None:
                cleaned[key] = action.get(key)
        if cleaned.get("type") == "scroll" and "clicks" not in cleaned:
            if action.get("scroll_y") is not None:
                cleaned["clicks"] = int(action.get("scroll_y"))
            else:
                cleaned["clicks"] = 0
        if "type" not in cleaned:
            cleaned["type"] = action_type
        return cleaned

    def _materialize_skill_actions(
        self,
        skill: Any,
        runtime_args: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
        parameters = skill.parameters if isinstance(getattr(skill, "parameters", None), dict) else {}
        resolved_args: Dict[str, Any] = {}

        for key, value in (runtime_args or {}).items():
            if self._is_skill_arg_scalar(value):
                resolved_args[str(key)] = value

        missing: List[str] = []
        for param_name, spec in parameters.items():
            token = str(param_name)
            if token in resolved_args and resolved_args[token] not in (None, ""):
                continue

            default_value = self._skill_param_default(spec)
            if default_value is not None:
                resolved_args[token] = default_value
                continue

            if self._skill_param_required(spec):
                missing.append(token)

        rendered = self._render_template_value(skill.actions, resolved_args)
        if not isinstance(rendered, list):
            return [], resolved_args, sorted(set(missing))

        rendered_actions = [dict(item) for item in rendered if isinstance(item, dict)]
        unresolved = self._extract_template_placeholders(rendered_actions)
        unresolved_missing = sorted(name for name in unresolved if name not in resolved_args)
        if unresolved_missing:
            missing.extend(unresolved_missing)

        return rendered_actions, resolved_args, sorted(set(missing))

    def _render_skill_verification_contract(
        self,
        skill: Any,
        resolved_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw_contract = (
            skill.verification_contract
            if isinstance(getattr(skill, "verification_contract", None), dict)
            else {}
        )
        if not raw_contract:
            return {}
        rendered = self._render_template_value(raw_contract, resolved_args)
        if not isinstance(rendered, dict):
            return {}
        contract: Dict[str, Any] = {}
        if rendered.get("sensor") is not None:
            contract["sensor"] = str(rendered.get("sensor")).strip().lower()
        if rendered.get("expected_state") is not None:
            expected = str(rendered.get("expected_state")).strip()
            if expected:
                contract["expected_state"] = expected[:500]
        if rendered.get("timeout_seconds") is not None:
            try:
                timeout = int(rendered.get("timeout_seconds"))
                contract["timeout_seconds"] = max(1, min(timeout, 30))
            except (TypeError, ValueError):
                pass
        return contract

    def _build_composable_skill_payload(
        self,
        actions: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        templated_actions = [dict(action) for action in actions if isinstance(action, dict)]
        parameters: Dict[str, Any] = {}
        counters: Dict[str, int] = {}
        param_fields = ("text", "content", "app_name", "url", "value")

        for action in templated_actions:
            for field in param_fields:
                raw_value = action.get(field)
                if not isinstance(raw_value, str):
                    continue
                value = raw_value.strip()
                if not value:
                    continue
                # Keep explicit templates intact.
                if SKILL_PLACEHOLDER_RE.search(value):
                    continue
                if not any(char.isalnum() for char in value):
                    continue

                counters[field] = counters.get(field, 0) + 1
                param_name = f"{field}_{counters[field]}"
                action[field] = "{" + param_name + "}"
                parameters[param_name] = {
                    "description": f"Runtime value for '{field}' in the skill action sequence.",
                    "required": False,
                    "default": raw_value,
                }

        return templated_actions, parameters

    def _derive_skill_verification_contract(
        self,
        current_step: Optional[Step],
        source_action: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        source_contract = (
            source_action.get("verification")
            if isinstance(source_action, dict) and isinstance(source_action.get("verification"), dict)
            else {}
        )
        if source_contract:
            contract: Dict[str, Any] = {}
            if source_contract.get("sensor") is not None:
                contract["sensor"] = str(source_contract.get("sensor")).strip().lower()
            if source_contract.get("expected_state") is not None:
                expected = str(source_contract.get("expected_state")).strip()
                if expected:
                    contract["expected_state"] = expected[:500]
            if source_contract.get("timeout_seconds") is not None:
                try:
                    timeout = int(source_contract.get("timeout_seconds"))
                    contract["timeout_seconds"] = max(1, min(timeout, 30))
                except (TypeError, ValueError):
                    pass
            if contract:
                return contract

        if current_step and getattr(current_step, "expected_state", ""):
            expected = str(current_step.expected_state).strip()
            if expected:
                return {
                    "sensor": "a11y_tree",
                    "expected_state": expected[:500],
                    "timeout_seconds": 5,
                }
        return {}

    def _render_template_value(self, value: Any, args: Dict[str, Any]) -> Any:
        if isinstance(value, dict):
            return {key: self._render_template_value(val, args) for key, val in value.items()}
        if isinstance(value, list):
            return [self._render_template_value(item, args) for item in value]
        if not isinstance(value, str):
            return value

        exact = SKILL_PLACEHOLDER_RE.fullmatch(value.strip())
        if exact:
            token = exact.group(1)
            if token in args:
                return args[token]

        def _replace(match: re.Match[str]) -> str:
            token = match.group(1)
            replacement = args.get(token)
            if replacement is None:
                return match.group(0)
            return str(replacement)

        return SKILL_PLACEHOLDER_RE.sub(_replace, value)

    def _extract_template_placeholders(self, value: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(value, dict):
            for child in value.values():
                found.update(self._extract_template_placeholders(child))
            return found
        if isinstance(value, list):
            for child in value:
                found.update(self._extract_template_placeholders(child))
            return found
        if isinstance(value, str):
            for match in SKILL_PLACEHOLDER_RE.finditer(value):
                found.add(match.group(1))
        return found

    def _skill_param_default(self, spec: Any) -> Any:
        if isinstance(spec, dict):
            default = spec.get("default")
            if self._is_skill_arg_scalar(default):
                return default
            return None
        if isinstance(spec, (int, float, bool)):
            return spec
        return None

    def _skill_param_required(self, spec: Any) -> bool:
        if isinstance(spec, dict):
            return bool(spec.get("required", False))
        return False

    def _is_skill_arg_scalar(self, value: Any) -> bool:
        return isinstance(value, (str, int, float, bool))

    def _run_recovery(self, failure_type: str) -> None:
        """Quick, deterministic recovery steps for common failure types."""
        # Press Escape twice to clear popups/dialogs
        if failure_type in {"popup_blocking", "blocked_by_popup", "popup"}:
            self.logger.info("Recovery: attempting ESC to clear popup")
            self.computer.execute({"type": "key", "keys": ["escape"]})
            time.sleep(0.3)
            self.computer.execute({"type": "key", "keys": ["escape"]})
            time.sleep(0.3)
        # Click desktop background to reset focus if wrong app
        if failure_type in {"wrong_app"}:
            width, height = self.display.logical_width, self.display.logical_height
            self.logger.info("Recovery: clicking desktop to reset focus")
            self.computer.execute(
                {"type": "left_click", "x": width * 0.05, "y": height * 0.95, "phantom_mode": False}
            )

    def _resolve_verification_contract(
        self,
        state: StateManager,
        action: Dict[str, Any],
        current_step: Optional[Step],
    ) -> VerificationContract:
        fallback_expected = None
        if current_step and getattr(current_step, "expected_state", ""):
            fallback_expected = str(current_step.expected_state).strip() or None
        if fallback_expected is None:
            fallback_expected = self._default_expected_state_for_action(action)
        return state.normalize_verification_contract(
            action.get("verification") if isinstance(action.get("verification"), dict) else None,
            fallback_sensor=self._default_sensor_for_action(action),
            fallback_expected_state=fallback_expected,
            verify_after=action.get("verify_after"),
        )

    def _default_sensor_for_action(self, action: Dict[str, Any]) -> str:
        action_type = str(action.get("type") or "").strip().lower()
        if action_type in {
            "wait",
            "wait_for_element",
            "wait_for_idle",
            "capture_only",
            "noop",
            "inspect_ui",
            "probe_ui",
            "notebook_op",
            "mouse_move",
            "hover",
            "scroll",
            "scroll_to_element",
        }:
            return "none"
        if action_type in {"sandbox_shell", "script_op"}:
            # Shell/script actions often have no reliable UI/telemetry delta; default to no-op verification
            # unless the action explicitly provides a stronger contract.
            return "none"
        if action_type == "clipboard_op":
            sub_action = str(action.get("sub_action") or "").strip().lower()
            # Clipboard reads already return direct tool success/failure; generic state-change
            # verification is often inconclusive and causes false failures.
            if sub_action == "read":
                return "none"
            return "os_telemetry"
        if action_type in {"open_app", "focus_window"}:
            return "os_telemetry"
        if action_type in {"browser_op"}:
            return "pixel_diff"
        return "a11y_tree"

    def _default_expected_state_for_action(self, action: Dict[str, Any]) -> str | None:
        action_type = str(action.get("type") or "").strip().lower()
        if action_type != "clipboard_op":
            return None

        sub_action = str(action.get("sub_action") or "").strip().lower()
        if sub_action == "write":
            content = str(action.get("content") or "")
            if not content:
                return "clipboard_equals:"
            if len(content) > 220:
                return f"clipboard_contains:{content[:220]}"
            return f"clipboard_equals:{content}"
        if sub_action == "clear":
            return "clipboard_equals:"
        return None

    def _collect_os_telemetry_snapshot(self, contract: VerificationContract) -> Dict[str, Any]:
        if contract.sensor != "os_telemetry":
            return {}
        snapshot: Dict[str, Any] = {}
        clipboard = self._read_clipboard_snapshot()
        if clipboard is not None:
            snapshot["clipboard"] = clipboard
        return snapshot

    def _read_clipboard_snapshot(self) -> str | None:
        try:
            res = self.computer.execute({"type": "clipboard_op", "sub_action": "read"})
        except Exception:
            return None
        if not getattr(res, "success", False):
            return None
        metadata = getattr(res, "metadata", {}) or {}
        value = metadata.get("content")
        if value is None:
            return ""
        return str(value)

    def _run_verification_contract(
        self,
        action: Dict[str, Any],
        contract: VerificationContract,
        current_frame: str,
        current_hash: str | None,
        ax_tree_before: Dict[str, Any] | None,
        telemetry_before: Dict[str, Any],
        global_hotkeys: set[tuple[str, ...]],
        phash_static_threshold: int,
    ) -> Dict[str, Any]:
        sensor = contract.sensor

        if sensor == "none":
            next_frame, next_hash = self.computer.capture_with_hash()
            hash_distance = self.computer.hash_distance(current_hash, next_hash)
            return {
                "passed": True,
                "reason": "verification bypassed by contract",
                "sensor": sensor,
                "changed": True,
                "next_frame": next_frame,
                "next_hash": next_hash,
                "hash_distance": hash_distance,
                "ssim_score": None,
                "ax_tree_after": None,
                "ax_changed": False,
                "note": "verification:none",
                "force_vision_next_turn": False,
            }

        if sensor == "os_telemetry":
            passed, reason = self._verify_os_telemetry(contract, telemetry_before)
            if passed:
                next_frame, next_hash = self.computer.capture_with_hash()
                hash_distance = self.computer.hash_distance(current_hash, next_hash)
                return {
                    "passed": True,
                    "reason": reason,
                    "sensor": sensor,
                    "changed": True,
                    "next_frame": next_frame,
                    "next_hash": next_hash,
                    "hash_distance": hash_distance,
                    "ssim_score": None,
                    "ax_tree_after": None,
                    "ax_changed": False,
                    "note": f"verification:{sensor}:ok",
                    "force_vision_next_turn": False,
                }

            visual = self._run_visual_verification(
                action=action,
                current_frame=current_frame,
                current_hash=current_hash,
                ax_tree_before=ax_tree_before,
                global_hotkeys=global_hotkeys,
                phash_static_threshold=phash_static_threshold,
            )
            if self._is_os_telemetry_inconclusive_reason(reason):
                visual_changed = bool(visual.get("changed"))
                visual["passed"] = visual_changed
                visual["reason"] = (
                    f"{reason}; visual fallback detected change"
                    if visual_changed
                    else f"{reason}; visual fallback found no change"
                )
                visual["note"] = f"verification:{sensor}:fallback"
            else:
                visual["passed"] = False
                visual["reason"] = reason
                visual["note"] = f"verification:{sensor}:timeout"
            visual["sensor"] = sensor
            visual["force_vision_next_turn"] = True
            return visual

        if sensor == "a11y_tree":
            passed, reason, ax_tree_after = self._verify_a11y_tree(contract, ax_tree_before)
            if passed:
                next_frame, next_hash = self.computer.capture_with_hash()
                hash_distance = self.computer.hash_distance(current_hash, next_hash)
                ax_changed = self._ax_changed(ax_tree_before, ax_tree_after) if ax_tree_after else False
                return {
                    "passed": True,
                    "reason": reason,
                    "sensor": sensor,
                    "changed": True,
                    "next_frame": next_frame,
                    "next_hash": next_hash,
                    "hash_distance": hash_distance,
                    "ssim_score": None,
                    "ax_tree_after": ax_tree_after,
                    "ax_changed": ax_changed,
                    "note": f"verification:{sensor}:ok",
                    "force_vision_next_turn": False,
                }

            visual = self._run_visual_verification(
                action=action,
                current_frame=current_frame,
                current_hash=current_hash,
                ax_tree_before=ax_tree_before,
                global_hotkeys=global_hotkeys,
                phash_static_threshold=phash_static_threshold,
            )
            if self._is_a11y_unavailable_reason(reason):
                visual_changed = bool(visual.get("changed"))
                visual["passed"] = visual_changed
                visual["reason"] = (
                    f"{reason}; visual fallback detected change"
                    if visual_changed
                    else f"{reason}; visual fallback found no change"
                )
                visual["note"] = f"verification:{sensor}:fallback"
            else:
                visual["passed"] = False
                visual["reason"] = reason
                visual["note"] = f"verification:{sensor}:timeout"
            visual["sensor"] = sensor
            visual["force_vision_next_turn"] = True
            return visual

        if sensor == "pixel_diff":
            passed, reason, frame_after = self._verify_pixel_diff(contract, current_frame)
            if passed:
                next_frame = frame_after
                next_hash = self.computer.hash_base64(next_frame)
                hash_distance = self.computer.hash_distance(current_hash, next_hash)
                return {
                    "passed": True,
                    "reason": reason,
                    "sensor": sensor,
                    "changed": True,
                    "next_frame": next_frame,
                    "next_hash": next_hash,
                    "hash_distance": hash_distance,
                    "ssim_score": None,
                    "ax_tree_after": None,
                    "ax_changed": False,
                    "note": f"verification:{sensor}:ok",
                    "force_vision_next_turn": False,
                }

            visual = self._run_visual_verification(
                action=action,
                current_frame=current_frame,
                current_hash=current_hash,
                ax_tree_before=ax_tree_before,
                global_hotkeys=global_hotkeys,
                phash_static_threshold=phash_static_threshold,
            )
            visual["passed"] = False
            visual["reason"] = reason
            visual["sensor"] = sensor
            visual["note"] = f"verification:{sensor}:timeout"
            visual["force_vision_next_turn"] = True
            return visual

        visual = self._run_visual_verification(
            action=action,
            current_frame=current_frame,
            current_hash=current_hash,
            ax_tree_before=ax_tree_before,
            global_hotkeys=global_hotkeys,
            phash_static_threshold=phash_static_threshold,
        )
        visual["note"] = f"verification:{sensor}"
        if contract.expected_state:
            expected_key, _ = self._parse_expected_state(contract.expected_state)
            if expected_key in {"any", "state_change", "changed"}:
                visual_changed = bool(visual.get("changed"))
                visual["passed"] = visual_changed
                visual["reason"] = "vision_full detected change" if visual_changed else "vision_full found no change"
            else:
                matched, expected_reason = self._evaluate_a11y_state(contract.expected_state, ax_tree_before, visual.get("ax_tree_after"))
                if matched:
                    visual["passed"] = True
                    visual["reason"] = expected_reason
                elif self._is_a11y_unavailable_reason(expected_reason):
                    visual_changed = bool(visual.get("changed"))
                    visual["passed"] = visual_changed
                    visual["reason"] = (
                        f"{expected_reason}; visual fallback detected change"
                        if visual_changed
                        else f"{expected_reason}; visual fallback found no change"
                    )
                    visual["note"] = f"verification:{sensor}:fallback"
                else:
                    visual["passed"] = False
                    visual["reason"] = expected_reason
        else:
            visual["passed"] = bool(visual.get("changed"))
            visual["reason"] = "vision_full detected change" if visual.get("changed") else "vision_full found no change"
        visual["sensor"] = sensor
        visual["force_vision_next_turn"] = True
        return visual

    def _run_visual_verification(
        self,
        action: Dict[str, Any],
        current_frame: str,
        current_hash: str | None,
        ax_tree_before: Dict[str, Any] | None,
        global_hotkeys: set[tuple[str, ...]],
        phash_static_threshold: int,
    ) -> Dict[str, Any]:
        verify_after = bool(action.get("verify_after", True))
        is_interactive = action.get("type") not in {"wait", "capture_only", "noop"}

        extra_delay = 0.0
        if verify_after and action.get("type") == "key":
            combo = tuple(sorted([str(k).lower() for k in action.get("keys") or []]))
            if combo in global_hotkeys:
                extra_delay = 0.5

        if verify_after and is_interactive:
            time.sleep(0.2 + extra_delay)
            stabilize_timeout = max(2.0, self.settings.settle_delay_ms / 1000.0)
            start_time = time.time()
            last_poll_frame = self.computer.capture_base64()
            stable_frames = 0
            while (time.time() - start_time) < stabilize_timeout:
                time.sleep(0.15)
                current_poll_frame = self.computer.capture_base64()
                if not self.computer.has_changed(last_poll_frame, current_poll_frame, threshold=0.002):
                    stable_frames += 1
                else:
                    stable_frames = 0
                last_poll_frame = current_poll_frame
                if stable_frames >= 2:
                    break
            next_frame = last_poll_frame
            next_hash = self.computer.hash_base64(next_frame)
        else:
            next_frame, next_hash = self.computer.capture_with_hash()

        hash_distance = self.computer.hash_distance(current_hash, next_hash)
        ssim_score = None
        ax_tree_after = None
        ax_changed = False

        if verify_after:
            ssim_score = self.computer.structural_similarity(current_frame, next_frame)
            if self.settings.enable_semantic:
                ax_after_res = self.computer.get_active_window_tree(max_depth=4)
                if ax_after_res.success:
                    ax_tree_after = ax_after_res.metadata.get("tree")
                    ax_changed = self._ax_changed(ax_tree_before, ax_tree_after)
            changed = self._compute_changed(
                current_frame,
                next_frame,
                hash_distance,
                ssim_score,
                ax_changed,
                phash_static_threshold=phash_static_threshold,
            )
            note = "verification:visual"
        else:
            changed = True
            note = "verify_skipped"

        return {
            "passed": bool(changed),
            "reason": "visual changed" if changed else "visual unchanged",
            "sensor": "vision_full",
            "changed": changed,
            "next_frame": next_frame,
            "next_hash": next_hash,
            "hash_distance": hash_distance,
            "ssim_score": ssim_score,
            "ax_tree_after": ax_tree_after,
            "ax_changed": ax_changed,
            "note": note,
            "force_vision_next_turn": True,
        }

    def _verify_os_telemetry(self, contract: VerificationContract, before_snapshot: Dict[str, Any]) -> tuple[bool, str]:
        deadline = time.time() + max(1, int(contract.timeout_seconds))
        last_reason = "telemetry condition unmet"
        while time.time() <= deadline:
            now_snapshot = self._collect_os_telemetry_snapshot(contract)
            ok, reason = self._evaluate_os_telemetry_state(contract.expected_state, before_snapshot, now_snapshot)
            if ok:
                return True, reason
            last_reason = reason
            if self._is_os_telemetry_inconclusive_reason(reason):
                return False, reason
            time.sleep(0.25)
        return False, last_reason

    def _has_non_clipboard_os_signal(
        self,
        before_snapshot: Dict[str, Any],
        after_snapshot: Dict[str, Any],
    ) -> bool:
        keys = set(before_snapshot.keys()) | set(after_snapshot.keys())
        return any(str(key).strip().lower() != "clipboard" for key in keys)

    def _has_non_clipboard_os_delta(
        self,
        before_snapshot: Dict[str, Any],
        after_snapshot: Dict[str, Any],
    ) -> bool:
        keys = set(before_snapshot.keys()) | set(after_snapshot.keys())
        for key in keys:
            if str(key).strip().lower() == "clipboard":
                continue
            if before_snapshot.get(key) != after_snapshot.get(key):
                return True
        return False

    def _is_os_telemetry_inconclusive_reason(self, reason: str) -> bool:
        token = str(reason or "").strip().lower()
        return token.startswith("os telemetry inconclusive")

    def _verify_a11y_tree(
        self,
        contract: VerificationContract,
        before_tree: Dict[str, Any] | None,
    ) -> tuple[bool, str, Dict[str, Any] | None]:
        deadline = time.time() + max(1, int(contract.timeout_seconds))
        last_reason = "a11y condition unmet"
        last_tree: Dict[str, Any] | None = None

        while time.time() <= deadline:
            ax_res = self.computer.get_active_window_tree(max_depth=4)
            if ax_res.success:
                last_tree = ax_res.metadata.get("tree")
                ok, reason = self._evaluate_a11y_state(contract.expected_state, before_tree, last_tree)
                if ok:
                    return True, reason, last_tree
                last_reason = reason
            else:
                last_reason = ax_res.reason or "a11y capture failed"
            time.sleep(0.35)

        return False, last_reason, last_tree

    def _is_a11y_unavailable_reason(self, reason: str) -> bool:
        token = str(reason or "").strip().lower()
        if not token:
            return True
        unavailable_markers = (
            "unavailable",
            "capture failed",
            "permission denied",
            "missing accessibility permission",
            "missing accessibility permissions",
            "accessibility permission",
            "accessibility permissions",
            "ax permission",
            "ax permissions",
            "ax api disabled",
            "kaxerrorapidisabled",
            "not trusted for accessibility",
            "not authorized for accessibility",
            "not supported",
            "blocked",
            "denied",
        )
        return any(marker in token for marker in unavailable_markers)

    def _verify_pixel_diff(
        self,
        contract: VerificationContract,
        base_frame: str,
    ) -> tuple[bool, str, str]:
        key, value = self._parse_expected_state(contract.expected_state)
        threshold = 0.01
        if key in {"pixel_change_gt", "pixel_diff_gt", "change_gt"} and value:
            try:
                threshold = max(0.0005, min(float(value), 0.2))
            except ValueError:
                threshold = 0.01
        if key in {"pixel_change_pct_gt"} and value:
            try:
                threshold = max(0.0005, min(float(value) / 100.0, 0.2))
            except ValueError:
                threshold = 0.01

        deadline = time.time() + max(1, int(contract.timeout_seconds))
        last_frame = base_frame
        while time.time() <= deadline:
            frame = self.computer.capture_base64()
            last_frame = frame
            if self.computer.has_changed(base_frame, frame, threshold=threshold):
                return True, f"pixel delta exceeded threshold={threshold:.4f}", frame
            time.sleep(0.2)
        return False, f"pixel delta stayed below threshold={threshold:.4f}", last_frame

    def _parse_expected_state(self, expected_state: Optional[str]) -> tuple[str, str]:
        raw = str(expected_state or "").strip()
        if not raw:
            return "any", ""
        if ":" in raw:
            key, value = raw.split(":", 1)
            return key.strip().lower(), value.strip()
        return raw.strip().lower(), ""

    def _evaluate_os_telemetry_state(
        self,
        expected_state: Optional[str],
        before_snapshot: Dict[str, Any],
        after_snapshot: Dict[str, Any],
    ) -> tuple[bool, str]:
        key, value = self._parse_expected_state(expected_state)
        clipboard_before = str(before_snapshot.get("clipboard") or "")
        clipboard_after = str(after_snapshot.get("clipboard") or "")
        value_l = value.lower()

        if key in {"any", "state_change", "changed"}:
            changed = before_snapshot != after_snapshot
            if changed:
                if not self._has_non_clipboard_os_signal(before_snapshot, after_snapshot):
                    return False, "os telemetry inconclusive (no non-clipboard signal)"
                if not self._has_non_clipboard_os_delta(before_snapshot, after_snapshot):
                    return False, "os telemetry inconclusive (clipboard-only change)"
                return True, "os telemetry changed"
            if not self._has_non_clipboard_os_signal(before_snapshot, after_snapshot):
                return False, "os telemetry inconclusive (no non-clipboard signal)"
            return False, "os telemetry unchanged"
        if key == "clipboard_changed":
            changed = clipboard_before != clipboard_after
            return changed, "clipboard changed" if changed else "clipboard unchanged"
        if key == "clipboard_contains":
            matched = value_l in clipboard_after.lower()
            return matched, "clipboard contains expected text" if matched else "clipboard missing expected text"
        if key == "clipboard_equals":
            matched = clipboard_after == value
            return matched, "clipboard equals expected text" if matched else "clipboard value mismatch"
        if key == "file_exists":
            exists = Path(value).expanduser().exists() if value else False
            return exists, "file exists" if exists else "file not found"
        if key in {"file_not_exists", "file_missing"}:
            missing = not Path(value).expanduser().exists() if value else True
            return missing, "file absent" if missing else "file still exists"
        if key in {"process_exists", "app_open"}:
            running = self._process_exists(value)
            return running, "process found" if running else "process not found"
        if key == "process_not_exists":
            stopped = not self._process_exists(value)
            return stopped, "process absent" if stopped else "process still running"
        if key == "app_focused":
            focused = self._process_exists(value) or (value_l in json.dumps(after_snapshot, ensure_ascii=False).lower())
            return focused, "app appears focused/open" if focused else "app focus not detected"

        if not self._has_non_clipboard_os_signal(before_snapshot, after_snapshot):
            return False, "os telemetry inconclusive (no non-clipboard signal)"
        blob = json.dumps(after_snapshot, ensure_ascii=False).lower()
        matched = key in blob if value == "" else value_l in blob
        return matched, "telemetry token found" if matched else "telemetry token missing"

    def _evaluate_a11y_state(
        self,
        expected_state: Optional[str],
        before_tree: Dict[str, Any] | None,
        after_tree: Dict[str, Any] | None,
    ) -> tuple[bool, str]:
        if after_tree is None:
            return False, "a11y tree unavailable"

        key, value = self._parse_expected_state(expected_state)
        payload = json.dumps(after_tree, ensure_ascii=False).lower()
        value_l = value.lower()

        if key in {"any", "state_change", "changed"}:
            if before_tree is None:
                return True, "a11y tree captured"
            changed = self._ax_changed(before_tree, after_tree)
            return changed, "a11y tree changed" if changed else "a11y tree unchanged"
        if key in {"text_exists", "contains", "title_contains", "url_contains"}:
            matched = value_l in payload
            return matched, "a11y text found" if matched else "a11y text not found"
        if key in {"text_not_exists", "not_contains"}:
            matched = value_l not in payload
            return matched, "a11y text absent" if matched else "a11y text still present"
        if key == "role_exists":
            matched = f"\"role\": \"{value_l}\"" in payload or f"\"role\":\"{value_l}\"" in payload
            return matched, "a11y role found" if matched else "a11y role not found"

        token = key if value == "" else value_l
        matched = token in payload
        return matched, "a11y token found" if matched else "a11y token missing"

    def _process_exists(self, name: str) -> bool:
        query = str(name or "").strip().lower()
        if not query:
            return False
        try:
            if self.computer.platform_name.lower().startswith("windows"):
                completed = subprocess.run(
                    ["tasklist", "/fo", "csv", "/nh"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
            else:
                completed = subprocess.run(
                    ["ps", "-A", "-o", "comm="],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
        except Exception:
            return False
        haystack = (completed.stdout or "").lower()
        return query in haystack

    def _format_loop_state(
        self, plan: Optional[Plan], state: StateManager, repeat_same_action: int, repeat_without_change: int
    ) -> dict:
        current_step = plan.current_step() if plan else None
        return {
            "current_step_id": current_step.id if current_step else None,
            "current_step_status": current_step.status if current_step else None,
            "failure_count": state.failure_count,
            "steps_taken": state.steps,
            "repeat_same_action": repeat_same_action,
            "repeat_without_change": repeat_without_change,
            "notebook_summary": state.get_notebook_summary() if hasattr(state, "get_notebook_summary") else ""
        }

    def _extract_cognitive_trace(self, action: Dict[str, Any]) -> str:
        parts: List[str] = []
        if action.get("_debug_trace"):
            parts.append(str(action.get("_debug_trace")))
        if action.get("_debug_rationale"):
            parts.append(f"rationale: {action.get('_debug_rationale')}")

        if action.get("type") == "macro_actions":
            for sub in action.get("actions") or []:
                if not isinstance(sub, dict):
                    continue
                if sub.get("_debug_rationale"):
                    parts.append(f"sub_action_rationale: {sub.get('_debug_rationale')}")
                    break

        return "\n".join(parts).strip()[:1400]

    def _strip_debug_fields(self, action: Dict[str, Any]) -> Dict[str, Any]:
        clean: Dict[str, Any] = {}
        for key, value in action.items():
            if str(key).startswith("_debug"):
                continue
            if key == "actions" and isinstance(value, list):
                sub_actions = []
                for item in value:
                    if isinstance(item, dict):
                        sub_actions.append(self._strip_debug_fields(item))
                    else:
                        sub_actions.append(item)
                clean[key] = sub_actions
                continue
            clean[key] = value
        return clean

    def _extract_crosshair(self, action: Dict[str, Any]) -> Dict[str, Any] | None:
        def _from_single(single: Dict[str, Any]) -> Dict[str, Any] | None:
            if single.get("x") is None or single.get("y") is None:
                return None
            return {
                "x": float(single.get("x")),
                "y": float(single.get("y")),
                "action_type": single.get("type"),
            }

        if action.get("type") == "macro_actions":
            for sub in action.get("actions") or []:
                if not isinstance(sub, dict):
                    continue
                hit = _from_single(sub)
                if hit:
                    return hit
            return None
        return _from_single(action)

    def _requires_state_change(self, action: Dict[str, Any], current_step: Optional[Step]) -> bool:
        action_type = action.get("type")
        if action_type == "macro_actions":
            return any(
                self._requires_state_change(item, current_step)
                for item in (action.get("actions") or [])
                if isinstance(item, dict)
            )

        non_transition_types = {
            "wait",
            "wait_for_element",
            "wait_for_idle",
            "capture_only",
            "noop",
            "inspect_ui",
            "probe_ui",
            "notebook_op",
            "clipboard_op",
            "sandbox_shell",
            "script_op",
            "browser_op",
            "mouse_move",
            "hover",
            "scroll",
            "scroll_to_element",
            "type",
            "fill_field",
        }
        if action_type in non_transition_types:
            return False

        if action_type in {"drag_and_drop", "select_area", "open_app", "focus_window"}:
            return True

        if action_type == "click_and_type":
            return bool(action.get("submit", False))

        if action_type in {"left_click", "right_click", "double_click", "click_element", "key"}:
            if action_type == "key":
                combo = {str(k).lower() for k in action.get("keys") or []}
                if combo.intersection({"enter", "return"}):
                    return True
                if combo.intersection({"space"}) and len(combo) == 1:
                    return self._action_suggests_progress(action, current_step)
                return False
            return self._action_suggests_progress(action, current_step)

        return False

    def _action_suggests_progress(self, action: Dict[str, Any], current_step: Optional[Step]) -> bool:
        keywords = {
            "avancar",
            "next",
            "continue",
            "proceed",
            "submit",
            "confirm",
            "save",
            "salvar",
            "finish",
            "done",
            "ok",
            "apply",
        }

        chunks: List[str] = []
        for key in ("semantic_label", "label", "semantic_path", "app_name", "text"):
            value = action.get(key)
            if value:
                chunks.append(str(value).lower())

        if current_step:
            chunks.append(str(current_step.description or "").lower())
            chunks.append(str(current_step.success_criteria or "").lower())
            chunks.append(str(getattr(current_step, "expected_state", "") or "").lower())

        blob = " ".join(chunks)
        return any(token in blob for token in keywords)

    def _format_no_change_reason(
        self,
        action: Dict[str, Any],
        current_step: Optional[Step],
        hash_distance: int,
        ssim_score: Optional[float],
        ax_changed: bool,
    ) -> str:
        step_id = current_step.id if current_step else "none"
        ssim_text = "none" if ssim_score is None else f"{ssim_score:.4f}"
        return (
            f"step={step_id} action={action.get('type')} hash_distance={hash_distance} "
            f"ssim={ssim_text} ax_changed={ax_changed}"
        )

    def _heuristic_step_complete(self, step: Step, action: dict, result: ActionResult, changed: bool) -> bool:
        """Conservative fallback when reflection is disabled."""
        if not result.success or not changed:
            return False
        if action.get("type") in {"wait", "noop", "capture_only"}:
            return False
        if step.status == "failed":
            return False
        # Require a direct UI interaction before auto-completing.
        if action.get("type") in {"left_click", "double_click", "right_click", "type", "scroll", "key", "mouse_move", "open_app"}:
            return True
        return False

    def _deterministic_step_complete(self, step: Step, ax_tree_after: dict | None, changed: bool) -> bool:
        """
        Fast programmatic verification using AX tree when possible before invoking the reflector.
        Examples:
        - success_criteria contains a window title and the title appears
        - success_criteria mentions a button/text disappearing and it is gone
        """
        if not changed or not ax_tree_after or not step.success_criteria:
            return False

        criteria = step.success_criteria.lower()

        def _walk(node: dict) -> list[dict]:
            nodes = [node]
            for child in node.get("children") or []:
                nodes.extend(_walk(child))
            return nodes

        nodes = _walk(ax_tree_after)
        titles = [str(n.get("title") or "").lower() for n in nodes if n.get("title")]
        roles = [str(n.get("role") or "").lower() for n in nodes if n.get("role")]

        # If success criteria mentions a specific title/keyword, check presence
        for token in re.findall(r"[a-z0-9]{3,}", criteria):
            if token in titles:
                return True

        # If criteria suggests a window/dialog is closed, verify by absence
        if "dialog" in criteria or "popup" in criteria:
            if not any("dialog" in r for r in roles):
                return True

        return False

    def _compute_changed(
        self,
        prev_frame: str,
        next_frame: str,
        hash_distance: int,
        ssim_score: float | None,
        ax_changed: bool,
        phash_static_threshold: int,
    ) -> bool:
        """Blend visual hash, SSIM-like score, and accessibility diffs to reduce false stagnation."""
        if ax_changed:
            return True

        if ssim_score is not None and ssim_score < self.settings.ssim_change_threshold:
            return True

        if hash_distance > phash_static_threshold:
            return True

        return self.computer.has_changed(prev_frame, next_frame)

    def _ax_changed(self, before: dict | None, after: dict | None) -> bool:
        if before is None or after is None:
            return False
        try:
            before_str = json.dumps(before, sort_keys=True)
            after_str = json.dumps(after, sort_keys=True)
            return before_str != after_str
        except Exception:
            return False

    def _maybe_save_skill(
        self,
        action: dict,
        result: ActionResult,
        current_step: Optional[Step],
        user_prompt: str,
        changed: bool,
    ) -> None:
        """Persist successful macro_actions as reusable procedural skills."""
        if action.get("type") != "macro_actions" or not result.success or not changed:
            return
        try:
            name_seed = ""
            if current_step and getattr(current_step, "description", ""):
                name_seed = current_step.description
            elif user_prompt:
                name_seed = user_prompt
            name = self._slugify(name_seed)[:50] or f"macro-{int(time.time())}"
            description = ""
            if current_step:
                description = current_step.success_criteria or current_step.description or ""
            if not description:
                description = user_prompt or ""
            tags = ["macro"]
            if current_step and current_step.id:
                tags.append(f"step:{current_step.id}")
            raw_actions = [dict(a) for a in action.get("actions") or []]
            composable_actions, parameters = self._build_composable_skill_payload(raw_actions)
            verification_contract = self._derive_skill_verification_contract(current_step, source_action=action)
            self.memory.save_skill(
                name=name,
                description=description,
                actions=composable_actions,
                tags=tags,
                source_prompt=user_prompt,
                plan_step_id=current_step.id if current_step else None,
                parameters=parameters,
                verification_contract=verification_contract,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("Failed to save procedural skill: %s", exc)

    def _slugify(self, text: str) -> str:
        """Lightweight slug for skill names."""
        if not text:
            return ""
        lowered = text.strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
        return slug or "macro"

    def _persist_episode(self, user_prompt: str, state: StateManager, plan: Optional[Plan]) -> None:
        episode_id = plan.id if plan else f"session-{int(state.started_at)}"
        outcome = "success"
        if state.failure_count > 0:
            outcome = "mixed"
        if plan and plan.current_step():
            outcome = "incomplete"
        summary = self.planner.summarize_episode(user_prompt, state.history, plan)

        log_path = self.memory.logs_dir / f"{episode_id}.log"
        try:
            with log_path.open("w", encoding="utf-8") as handle:
                for line in state.history:
                    handle.write(f"{line}\n")
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Failed to write episode log: %s", exc)
            log_path = None

        episode = Episode(
            id=episode_id,
            created_at=state.started_at,
            user_prompt=user_prompt,
            plan=plan.to_dict() if plan else {},
            outcome=outcome,
            summary=summary,
            tags=["desktop", "cua"],
            raw_log_path=str(log_path) if log_path else None,
        )
        try:
            self.memory.save_episode(episode)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Failed to persist episode: %s", exc)

    def _compress_history(self, state: StateManager) -> None:
        """Summarize the oldest part of the history to keep the context manageable."""
        # Keep index 0 (usually plan_init or user_prompt)
        # Summarize index 1 to 20
        # Keep rest
        if len(state.history) < 60:
            return
        
        chunk_to_summarize = state.history[1:21]
        summary = self.planner.summarize_history_chunk(chunk_to_summarize)
        if summary:
            self.logger.info("Compressing history: %s items -> 1 summary", len(chunk_to_summarize))
            new_history = [state.history[0]] + [f"history_summary:{summary}"] + state.history[21:]
            state.history = new_history

    def _resolve_element_references(self, action: dict, tags: List[Dict[str, Any]]) -> bool:
        """Translate element_id to x/y (physical px) using the most recent overlay tags."""
        lookup: Dict[str, Dict[str, Any]] = {str(tag["id"]): tag for tag in tags}

        def _annotate_from_tag(act: dict, tag: Dict[str, Any]) -> bool:
            frame = tag.get("frame") or {}
            try:
                cx, cy = self._frame_center_px(frame)
                act["x"] = cx
                act["y"] = cy
                if "semantic_role" not in act and tag.get("role"):
                    act["semantic_role"] = tag.get("role")
                if "semantic_label" not in act and tag.get("label"):
                    act["semantic_label"] = tag.get("label")
                if "semantic_path" not in act and tag.get("path"):
                    act["semantic_path"] = tag.get("path")
                return True
            except Exception:
                return False

        def _apply(act: dict) -> bool:
            if act.get("x") is not None and act.get("y") is not None:
                return True

            raw_element_id = act.get("element_id")
            if raw_element_id is not None:
                token = str(raw_element_id).strip()
                if not token:
                    return True
                tag = lookup.get(token)
                if tag:
                    return _annotate_from_tag(act, tag)
                if token.isdigit():
                    # Numeric IDs are strict overlay references; treat miss as stale grounding.
                    return False
                fallback = self._match_tag_reference(token, tags)
                if fallback:
                    return _annotate_from_tag(act, fallback)
                # Non-numeric IDs are semantic references; keep execution moving.
                act.setdefault("element_ref", token)
                return True

            raw_ref = act.get("element_ref")
            ref = str(raw_ref).strip() if raw_ref is not None else ""
            if not ref:
                return True
            if ref.isdigit():
                tag = lookup.get(ref)
                if tag:
                    return _annotate_from_tag(act, tag)
                return False
            matched = self._match_tag_reference(ref, tags)
            if matched:
                return _annotate_from_tag(act, matched)
            return True

        if action.get("type") == "macro_actions":
            for sub in action.get("actions") or []:
                if not _apply(sub):
                    return False
            return True
        return _apply(action)

    def _frame_center_px(self, frame: Dict[str, Any]) -> tuple[float, float]:
        """Return logical point center for a frame."""
        cx = float(frame.get("x", 0)) + float(frame.get("w", 0)) / 2.0
        cy = float(frame.get("y", 0)) + float(frame.get("h", 0)) / 2.0
        return cx, cy

    def _match_tag_reference(self, reference: str, tags: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        token = str(reference or "").strip().lower()
        if not token or not tags:
            return None

        normalized = self._normalize_reference_token(token)
        parts = [
            self._normalize_reference_token(part)
            for part in re.split(r"[^a-z0-9]+", token)
            if part and len(part) > 1
        ]
        generic_parts = {
            "btn",
            "button",
            "input",
            "field",
            "text",
            "textbox",
            "link",
            "label",
            "item",
            "icon",
            "window",
            "pane",
            "tab",
            "menu",
            "msg",
            "message",
            "element",
        }
        parts = [part for part in parts if part not in generic_parts]

        best_tag: Dict[str, Any] | None = None
        best_score = 0.0
        for tag in tags:
            role = str(tag.get("role") or "").lower()
            label = str(tag.get("label") or "").lower()
            path = str(tag.get("path") or "").lower()
            blob = f"{role} {label} {path}".strip()
            if not blob:
                continue

            score = 0.0
            norm_blob = self._normalize_reference_token(blob)
            if normalized and normalized in norm_blob:
                score += 4.0
            if token == label or token == path or token == role:
                score += 6.0
            if token in label:
                score += 3.0
            if token in path:
                score += 2.0
            if token in role:
                score += 1.5

            if parts:
                matched_parts = 0
                for part in parts:
                    if part and part in norm_blob:
                        score += 1.2
                        matched_parts += 1
                if matched_parts == len(parts):
                    score += 2.0

            if score > best_score:
                best_score = score
                best_tag = tag

        return best_tag if best_score > 0.0 else None

    def _normalize_reference_token(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())

    def _refresh_grounding(self, state: StateManager) -> tuple[str, str, List[Dict[str, Any]], dict | None]:
        """
        Capture a new frame, hash, tags, and AX tree immediately.
        Used when element references fail so the next action proposal has fresh context.
        """
        frame, phash = self.computer.capture_with_hash()
        ax_tree = None
        if self.settings.enable_semantic:
            ax_res = self.computer.get_active_window_tree(max_depth=4)
            if ax_res.success:
                raw_tree = ax_res.metadata.get("tree")
                ax_tree = prune_ax_tree_for_prompt(raw_tree) if raw_tree else None
        if not ax_tree:
            vision_elements = self.computer.detect_ui_elements(frame)
            if vision_elements:
                ax_tree = {
                    "role": "AXWindow",
                    "title": "Visual Fallback",
                    "children": vision_elements,
                    "frame": {"x": 0, "y": 0, "w": self.display.logical_width, "h": self.display.logical_height},
                }
        tags: List[Dict[str, Any]] = []
        overlay_frame = frame
        if ax_tree:
            nodes = flatten_nodes_with_frames(ax_tree, max_nodes=40)
            overlay_frame, tags = draw_som_overlay(frame, nodes, self.display)
        state.record_observation(overlay_frame, changed=True, phash=phash, note="grounding_refresh")
        return overlay_frame, phash, tags, ax_tree
