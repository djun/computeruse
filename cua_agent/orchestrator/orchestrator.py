"""Session orchestration logic (OS-agnostic)."""

from __future__ import annotations

import sys
import time
import re
from typing import Any, Dict, List, Optional

from cua_agent.agent.cognitive_core import CognitiveCore
from cua_agent.agent.state_manager import ActionResult, StateManager
from cua_agent.computer.adapter import ComputerAdapter
from cua_agent.grounding.grounder import Grounder
from cua_agent.memory.memory_manager import Episode, MemoryManager
from cua_agent.memory.skill_composer import SkillComposer
from cua_agent.observability import LiveDebugDashboard
from cua_agent.orchestrator.action_policy import ActionPolicy
from cua_agent.orchestrator.planner_client import PlannerClient
from cua_agent.orchestrator.planning import Plan, Step
from cua_agent.orchestrator.react_controller import ReactController
from cua_agent.orchestrator.recovery_manager import RecoveryManager
from cua_agent.orchestrator.reflection import Reflector
from cua_agent.orchestrator.verification_manager import VerificationManager
from cua_agent.utils.config import Settings
from cua_agent.utils.logger import get_logger
from cua_agent.utils.ax_pruning import prune_ax_tree_for_prompt
from cua_agent.utils.ax_utils import draw_som_overlay, flatten_nodes_with_frames


class Orchestrator:
    """Coordinates planning, execution, and memory."""

    def __init__(self, settings: Settings, computer: ComputerAdapter) -> None:
        self.settings = settings
        self.computer = computer
        self.logger = get_logger(__name__, level=settings.log_level)
        self.cognitive_core = CognitiveCore(settings, computer)
        self.memory = MemoryManager(settings)
        self.planner = PlannerClient(settings, platform_name=computer.platform_name)
        self.reflector = Reflector(settings)
        self.grounder = Grounder(settings, computer)
        self.verifier = VerificationManager(settings, computer)
        self.action_policy = ActionPolicy(settings)
        self.react_controller = ReactController(settings)
        self.recovery_manager = RecoveryManager(settings)
        self.skill_composer = SkillComposer(platform_name=computer.platform_name)
        self.trajectory = None
        if settings.enable_trajectory_recording:
            from cua_agent.observability.trajectory import TrajectoryRecorder

            self.trajectory = TrajectoryRecorder(settings.trajectory_path)
        self.dashboard = LiveDebugDashboard(settings, self.logger)
        self.display = computer.display
        self.global_hotkeys = getattr(computer, "global_hotkeys", set())

        if not settings.sends_real_input():
            if settings.simulation_mode:
                self.logger.warning("SIMULATION_MODE is on; the loop runs but no real input is sent.")
            else:
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
        max_plan_revisions = self.settings.max_replans_per_task
        global_hotkeys = self.global_hotkeys
        low_change_streak = 0
        VISUAL_HASH_STATIC_THRESHOLD = 4  # Hamming distance; increased to ignore noise
        STAGNATION_LIMIT = 5        # consecutive frames with minimal change
        current_tags: List[Dict[str, Any]] = []
        pending_zoom_frame: str | None = None
        step_trace: List[Dict[str, Any]] = []
        active_step_id = plan.current_step().id if plan and plan.current_step() else None
        force_vision_next_turn = True
        recovery_pending_replan = False
        low_conf_refresh_streak = 0
        # noop ("no safe action now") and invalid_action (mapping/schema error)
        # keep the loop alive with feedback instead of ending the task; this
        # bounds how many turns in a row may stall without a real action.
        consecutive_stall_turns = 0
        MAX_CONSECUTIVE_STALL_TURNS = 3
        # `done` with an incomplete plan step and no evidence is challenged once
        # per step; a second insistence on the same step is accepted so the
        # challenge can never loop. Tracked by step id so a premature `done` on a
        # later, different incomplete step is still challenged.
        done_challenged_step_id: Any = None

        try:
            while not state.should_halt():
                budget = self.settings.max_total_tokens
                if budget and self._total_tokens_used() >= budget:
                    used = self._total_tokens_used()
                    self.logger.warning("Token budget reached (%s/%s); stopping.", used, budget)
                    state.history.append(f"token_budget_exceeded:{used}/{budget}")
                    self.dashboard.push_event(f"token budget exceeded ({used}/{budget}); stopping")
                    break

                wants_replan = recovery_pending_replan or self._should_replan(
                    plan, state, repeat_same_action, repeat_without_change
                )
                recovery_pending_replan = False  # consumed this turn to avoid replan loops
                if plan and plan_revision_count < max_plan_revisions and wants_replan:
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
                turn = self.react_controller.start_turn(state, plan)
                include_visual_context = force_vision_next_turn
                
                # Context Compression
                if len(state.history) > 60:
                    self._compress_history(state)

                force_grounding_vision = force_vision_next_turn or self.react_controller.should_force_vision(state)
                grounding = self.grounder.observe(
                    previous=state.last_grounding,
                    force_vision=force_grounding_vision,
                    include_semantic=self.settings.enable_semantic,
                    include_visual=True,
                )
                state.record_grounding(grounding)
                current_frame = grounding.screenshot_b64
                current_hash = grounding.frame_hash
                ax_tree = grounding.ax_tree
                current_tags = grounding.som_tags
                # In native-coordinate mode, send the raw screenshot (no numbered overlay)
                # and let the model return x/y directly; the SoM overlay is the fallback aid.
                if not include_visual_context:
                    overlay_frame = ""
                elif pending_zoom_frame:
                    # Show the zoomed region this turn; coordinates stay in original space.
                    overlay_frame = pending_zoom_frame
                elif self.settings.prefer_native_coordinates:
                    overlay_frame = grounding.screenshot_b64
                else:
                    overlay_frame = grounding.overlay_b64
                pending_zoom_frame = None

                loop_state = self._format_loop_state(plan, state, repeat_same_action, repeat_without_change)
                loop_state["visual_context"] = "image" if include_visual_context else "text_only"
                
                # Retrieve relevant skills; prompt inclusion is gated by score so
                # weak matches do not spend context every turn.
                relevant_skills = []
                query_text = (current_step.description if current_step else "") or user_prompt or ""
                if query_text:
                    relevant_skills = [
                        match.skill
                        for match in self.memory.search_skills_scored(query_text)
                        if match.score >= self._skill_prompt_threshold(match.strategy)
                    ][:3]

                envelope = self.cognitive_core.propose_react_action(
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
                    grounding=grounding,
                    state_view=state.to_react_view(),
                )
                action = envelope.action
                cognitive_trace = self._extract_cognitive_trace(action)
                if cognitive_trace:
                    self.dashboard.push_thought(cognitive_trace)

                if action.get("type") == "done":
                    reason = str(action.get("reason") or "task completed")
                    evidence = str(action.get("evidence") or "")
                    if (
                        current_step is not None
                        and not evidence
                        and done_challenged_step_id != current_step.id
                    ):
                        done_challenged_step_id = current_step.id
                        state.history.append(
                            f"done_challenged:step_{current_step.id}_incomplete_without_evidence"
                        )
                        repeat_info_for_model = {
                            "count": repeat_same_action,
                            "action": repr(action),
                            "hint": (
                                f"You declared done, but plan step {current_step.id} "
                                f"('{current_step.description}') is not complete and no evidence was given. "
                                "Either finish the step, or call done again with concrete `evidence` "
                                "(visible text, final state) that the task is complete."
                            ),
                        }
                        self.dashboard.push_event("done challenged: plan step incomplete, no evidence")
                        self.logger.info(
                            "Done challenged: step %s incomplete and no evidence provided.", current_step.id
                        )
                        continue
                    self.logger.info("Done action received; stopping loop. Reason: %s", reason)
                    state.history.append(
                        f"task_done:{reason}" + (f":evidence={evidence[:300]}" if evidence else "")
                    )
                    self.dashboard.push_event(f"task_done:{reason}")
                    break

                # Information HITL: the model needs a human answer (login, captcha,
                # ambiguity) — distinct from approval HITL handled by the engine.
                if action.get("type") == "ask_user":
                    question = str(action.get("question") or "").strip()
                    ask_kind = str(action.get("kind") or "other")
                    answer = self._request_user_input(question, ask_kind)
                    if answer:
                        consecutive_stall_turns = 0
                        state.history.append(f"user_answer:{ask_kind}:{answer[:400]}")
                        self.dashboard.push_event(f"user answered ({ask_kind})")
                        repeat_info_for_model = {
                            "count": 0,
                            "action": repr(action),
                            "hint": f"User answered your question ('{question[:120]}'): {answer[:400]}",
                        }
                        # The user may have acted on the screen (login, captcha,
                        # permission dialog); re-observe before the next action.
                        force_vision_next_turn = True
                    else:
                        consecutive_stall_turns += 1
                        state.history.append(f"user_input_unavailable:{ask_kind}:{question[:200]}")
                        self.dashboard.push_event("ask_user: no interactive user available")
                        repeat_info_for_model = {
                            "count": 0,
                            "action": repr(action),
                            "hint": (
                                "No interactive user is available to answer. Continue autonomously "
                                "if safe, or finish with done explaining the blocker."
                            ),
                        }
                        if consecutive_stall_turns >= MAX_CONSECUTIVE_STALL_TURNS:
                            self.logger.warning("Unanswerable ask_user repeated; stopping loop.")
                            state.history.append("stalled:ask_user_without_interactive_user")
                            self.dashboard.push_event("stalled: repeated ask_user without a user; stopping")
                            break
                    continue

                if action.get("type") in {"noop", "invalid_action"}:
                    reason = str(action.get("reason") or "")
                    consecutive_stall_turns += 1
                    if action.get("type") == "invalid_action":
                        state.record_action(
                            action,
                            ActionResult(
                                success=False,
                                reason=reason or "invalid action",
                                code="invalid_action",
                                category="schema",
                                retryable=True,
                            ),
                        )
                        repeat_info_for_model = {
                            "count": repeat_same_action,
                            "action": repr(action),
                            "hint": (
                                f"Previous tool call was invalid: {reason}. "
                                "Fix the arguments or choose a different action; the task is still in progress."
                            ),
                        }
                        self.dashboard.push_event(f"invalid_action:{reason}")
                        self.logger.warning("Invalid action from model; feeding back. Reason: %s", reason)
                    else:
                        state.history.append(f"noop:{reason}")
                        self.dashboard.push_event(f"noop:{reason}")
                        self.logger.info("Noop action; continuing loop. Reason: %s", reason)
                    if consecutive_stall_turns >= MAX_CONSECUTIVE_STALL_TURNS:
                        self.logger.warning(
                            "%s consecutive stall turns (noop/invalid_action); stopping loop.",
                            consecutive_stall_turns,
                        )
                        state.history.append(f"stalled:{consecutive_stall_turns}_consecutive_noop_or_invalid")
                        self.dashboard.push_event("stalled: too many consecutive noop/invalid actions; stopping")
                        break
                    continue
                consecutive_stall_turns = 0

                # Model-requested observation (screenshot/capture_only): end the
                # turn and guarantee the next prompt carries fresh visual context.
                # Without this the sensor-`none` verification path would leave
                # force_vision_next_turn=False and the requested screenshot would
                # never reach the model.
                if action.get("type") == "capture_only":
                    result = self.computer.execute(action)
                    state.record_action(action, result)
                    reason = str(action.get("reason") or "model requested screenshot")
                    state.history.append(f"screenshot_requested:{reason}")
                    self.dashboard.push_event("screenshot requested; forcing visual context next turn")
                    force_vision_next_turn = True
                    continue

                if action.get("type") == "run_skill":
                    skill_ref = action.get("skill_id") or action.get("skill_name")
                    skill = self.memory.get_skill(skill_ref) if skill_ref else None
                    if not skill:
                        result = ActionResult(
                            success=False,
                            reason="skill not found",
                            code="skill_not_found",
                            category="execution",
                            retryable=False,
                            suggested_next=["use a skill ID from the listed skills", "perform the steps manually"],
                        )
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
                    rendered_actions, resolved_args, missing_params = self.skill_composer._materialize_skill_actions(skill, runtime_args)
                    if missing_params:
                        result = ActionResult(
                            success=False,
                            reason="missing required skill args: " + ", ".join(sorted(missing_params)),
                            code="skill_args_missing",
                            category="schema",
                            retryable=True,
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
                    skill_contract = self.skill_composer._render_skill_verification_contract(skill, resolved_args)
                    if requested_contract:
                        action["verification"] = requested_contract
                    elif skill_contract:
                        action["verification"] = skill_contract
                action = self._strip_debug_fields(action)
                policy_decision = self.action_policy.normalize_and_guard(
                    action,
                    grounding=grounding,
                    state=state,
                    step_risk=getattr(current_step, "risk_level", None) if current_step else None,
                )
                action = policy_decision.action
                if not policy_decision.allowed:
                    result = self.action_policy.blocked_result(policy_decision)
                    state.record_action(action, result)
                    repeat_info_for_model = {
                        "count": repeat_same_action,
                        "action": repr(action),
                        "hint": policy_decision.reason,
                    }
                    state.record_turn(
                        self.react_controller.finalize_turn(
                            turn,
                            observation_summary=envelope.observation_summary,
                            grounding_quality=grounding.quality,
                            selected_target_gid=policy_decision.target_gid,
                            action=action,
                            result={"success": result.success, "reason": result.reason, "metadata": result.metadata},
                            recovery_decision={"reason": policy_decision.reason, "stop": False},
                        ),
                    )
                    continue

                # Honor the policy's low-confidence target signal: refresh grounding
                # before acting so MIN_GROUNDING_CONFIDENCE is enforced rather than
                # advisory. Bounded to one refresh per weak target so a persistently
                # weak target still gets a best-effort attempt instead of looping.
                if bool(action.pop("needs_fresh_grounding", False)) and low_conf_refresh_streak < 1:
                    low_conf_refresh_streak += 1
                    state.history.append("grounding_confidence_low:refresh_before_act")
                    self.dashboard.push_event(
                        "grounding confidence below MIN_GROUNDING_CONFIDENCE; refreshing before acting"
                    )
                    repeat_info_for_model = {
                        "count": repeat_same_action,
                        "action": repr(action),
                        "hint": (
                            "Target grounding confidence below MIN_GROUNDING_CONFIDENCE; grounding was "
                            "refreshed. Re-select the target on the new overlay or pick a higher-confidence element."
                        ),
                    }
                    current_frame, current_hash, current_tags, ax_tree = self._refresh_grounding(state)
                    force_vision_next_turn = True
                    continue
                low_conf_refresh_streak = 0

                verification_contract = self.verifier.resolve_contract(state, action, current_step)
                action["verification"] = verification_contract.to_dict()
                telemetry_before = self.verifier.collect_os_telemetry_snapshot(verification_contract)

                # Resolve overlay element references to coordinates
                resolved_ok = self._resolve_element_references(action, current_tags)
                if not resolved_ok:
                    result = ActionResult(
                        success=False,
                        reason="element_id not found",
                        code="target_not_found",
                        category="grounding",
                        retryable=True,
                        suggested_next=["observe:fused", "observe:zoom", "probe_ui"],
                    )
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

                # Handle Zoom: re-capture a region at higher detail as an observation.
                if action.get("type") == "zoom":
                    result = self.computer.execute(action)
                    state.record_action(action, result)
                    if result.success and result.metadata:
                        pending_zoom_frame = result.metadata.get("zoom_image") or None
                        region = result.metadata.get("region")
                        state.history.append(f"zoom_region:{region}")
                        self.dashboard.push_event(f"zoom region {region}")
                    else:
                        state.history.append(f"zoom_failed:{result.reason}")
                    force_vision_next_turn = True
                    continue

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
                        result = ActionResult(
                            success=False,
                            reason="hotkey deduped",
                            code="hotkey_deduped",
                            category="execution",
                            retryable=False,
                            suggested_next=["click the visible control instead", "observe:screenshot"],
                        )
                        state.record_action(action, result)
                        repeat_info_for_model = {"count": repeat_same_action, "action": repr(action)}
                        continue
                    hotkey_counts[combo] = count + 1
                
                if action.get("type") == "open_app":
                    app_key = ("open_app", action.get("app_name", "").lower())
                    count = hotkey_counts.get(app_key, 0)
                    if count >= 1:  # Strict limit: don't open the same app twice in a short loop
                         self.logger.info("Skipping open_app %s; already executed", app_key[1])
                         result = ActionResult(
                             success=False,
                             reason="app open deduped",
                             code="app_open_deduped",
                             category="execution",
                             retryable=False,
                             suggested_next=["focus the already-open window", "observe:screenshot"],
                         )
                         state.record_action(action, result)
                         repeat_info_for_model = {"count": repeat_same_action, "action": repr(action)}
                         continue
                    hotkey_counts[app_key] = count + 1

                result = self.computer.execute(action)
                state.record_action(action, result)
                if self.trajectory is not None:
                    self.trajectory.record(
                        action, success=result.success, frame_hash=current_hash, reason=result.reason
                    )
                if action.get("execution") == "shell" and result.metadata:
                    stdout = (result.metadata.get("stdout") or "").strip()
                    stderr = (result.metadata.get("stderr") or "").strip()
                    if stdout:
                        state.history.append(f"shell_stdout:{stdout[:500]}")
                    if stderr:
                        state.history.append(f"shell_stderr:{stderr[:500]}")

                verification_outcome = self.verifier.run_verification_contract(
                    action=action,
                    contract=verification_contract,
                    current_frame=current_frame,
                    current_hash=current_hash,
                    ax_tree_before=ax_tree,
                    telemetry_before=telemetry_before,
                    global_hotkeys=global_hotkeys,
                    visual_hash_static_threshold=VISUAL_HASH_STATIC_THRESHOLD,
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

                # Macros truncated at an observation/done sub-action: surface the
                # truncation to the model, and when the cut was an observation,
                # guarantee fresh visual context next turn regardless of the
                # verification sensor used for the executable prefix.
                if action.get("type") == "macro_actions" and action.get("truncation_note"):
                    state.history.append(f"macro_truncated:{action.get('truncation_note')}")
                    if action.get("observe_after"):
                        force_vision_next_turn = True

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

                recovery_decision = self.recovery_manager.decide(
                    state=state,
                    plan=plan,
                    verification=verification_outcome,
                    repeat_same_action=repeat_same_action,
                    repeat_without_change=repeat_without_change,
                    reason=verification_reason,
                )
                # Let the recovery decision drive the loop (not just be logged): it can
                # request a forced visual refresh and/or a replan on the next turn.
                if recovery_decision.force_vision_next_turn:
                    force_vision_next_turn = True
                if recovery_decision.replan:
                    recovery_pending_replan = True
                if recovery_decision.request_user_input:
                    guidance = self._request_user_input(
                        recovery_decision.user_prompt
                        or "The agent is stuck and needs guidance. How should it proceed?",
                        "recovery",
                    )
                    if guidance:
                        state.history.append(f"user_guidance:{guidance[:400]}")
                        self.dashboard.push_event("user provided recovery guidance")
                        repeat_info_for_model = {
                            "count": 0,
                            "action": repr(action),
                            "hint": f"User guidance: {guidance[:400]}",
                        }
                        force_vision_next_turn = True
                # Strip heavy payloads (base64 frame, full a11y subtree) from the
                # turn record. The turn is copied into the event log and fed back
                # through compact_view into the next prompt; keeping image/tree data
                # there bloats memory and displaces useful state when the prompt is
                # truncated. The full outcome is still used directly above/below.
                verification_record = {
                    key: value
                    for key, value in verification_outcome.items()
                    if key not in {"next_frame", "ax_tree_after"}
                }
                state.record_turn(
                    self.react_controller.finalize_turn(
                        turn,
                        observation_summary=envelope.observation_summary,
                        grounding_quality=grounding.quality,
                        selected_target_gid=action.get("target_gid") or policy_decision.target_gid,
                        action=action,
                        verification=verification_record,
                        result={"success": result.success, "reason": result.reason, "metadata": result.metadata},
                        recovery_decision=recovery_decision.to_dict(),
                    ),
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
                    if hash_distance <= VISUAL_HASH_STATIC_THRESHOLD and is_action_interactive and not ax_changed:
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

                repeat_same_action, repeat_without_change, pending_break, break_reason = (
                    self._apply_repeat_stagnation(
                        is_wait=is_wait,
                        pending_break=pending_break,
                        break_reason=break_reason,
                        action_sig=action_sig,
                        last_action_sig=last_action_sig,
                        changed=changed,
                        repeat_same_action=repeat_same_action,
                        repeat_without_change=repeat_without_change,
                    )
                )

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

    def _total_tokens_used(self) -> int:
        """Cumulative tokens across planner + cognitive core + reflector + grounder this run."""
        action_engine = getattr(getattr(self, "computer", None), "action_engine", None)
        grounder = getattr(action_engine, "grounding_model", None)
        return (
            int(getattr(self.planner, "tokens_used", 0) or 0)
            + int(getattr(self.cognitive_core, "tokens_used", 0) or 0)
            + int(getattr(self.reflector, "tokens_used", 0) or 0)
            + int(getattr(grounder, "tokens_used", 0) or 0)
        )

    @staticmethod
    def _apply_repeat_stagnation(
        *,
        is_wait: bool,
        pending_break: bool,
        break_reason: str,
        action_sig: str | None,
        last_action_sig: str | None,
        changed: bool,
        repeat_same_action: int,
        repeat_without_change: int,
    ) -> tuple[int, int, bool, str]:
        """Update repeat counters and the break decision from a turn's outcome.

        Pure: no side effects. Extracted from the session loop for testability.
        Counters reset when the action is a wait or a break is already pending.
        """
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
        return repeat_same_action, repeat_without_change, pending_break, break_reason

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

    def _current_skill_context(self) -> Dict[str, Any]:
        context: Dict[str, Any] = {"platform": self.computer.platform_name}
        try:
            ax_res = self.computer.get_active_window_tree(max_depth=1)
            if ax_res.success:
                tree = (ax_res.metadata or {}).get("tree") or {}
                title = str(tree.get("title") or tree.get("name") or "").strip()
                if title:
                    context["active_window_title"] = title
                app = str(tree.get("app") or tree.get("application") or "").strip()
                if app:
                    context["active_app"] = app
        except Exception:
            pass
        return context

    def _grounding_signature_from_frame(self, frame_b64: str) -> Dict[str, Any]:
        signature: Dict[str, Any] = {"labels": [], "roles": []}
        try:
            nodes = self.computer.detect_ui_elements(frame_b64)
        except Exception:
            return signature
        labels: set[str] = set()
        roles: set[str] = set()
        for node in nodes or []:
            label = str(node.get("label") or node.get("title") or node.get("text") or "").strip()
            role = str(node.get("role") or "").strip()
            if label:
                labels.add(label[:120])
            if role:
                roles.add(role[:80])
        signature["labels"] = sorted(labels)[:40]
        signature["roles"] = sorted(roles)[:40]
        return signature

    def _attempt_fast_path(self, user_prompt: str, current_frame: str, current_hash: str) -> dict:
        """
        Try a high-confidence procedural skill before invoking planner/model.
        Falls back to normal planning when the cached macro does not complete the task.
        """
        context = self._current_skill_context()
        grounding_signature = self._grounding_signature_from_frame(current_frame)
        match = self.memory.select_fast_path_skill(
            user_prompt,
            top_k=3,
            context=context,
            grounding_signature=grounding_signature,
        )
        if not match:
            return {"attempted": False, "success": False}

        skill = match.skill
        fast_actions, resolved_args, missing_params = self.skill_composer._materialize_skill_actions(skill, {})
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
        skill_contract = self.skill_composer._render_skill_verification_contract(skill, resolved_args)
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
            self.memory.record_skill_result(skill.id, success=True)
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
        self.memory.record_skill_result(
            skill.id,
            success=False,
            negative_example={
                "prompt": user_prompt,
                "reason": failure_note,
                "context": context,
                "grounding_signature": grounding_signature,
            },
        )
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
        if action_type in {"notebook_op", "inspect_ui", "probe_ui", "capture_only", "noop", "done", "invalid_action", "ask_user"}:
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
        actions = self.skill_composer._extract_recovered_actions(step_trace)
        actions, parameters = self.skill_composer._build_composable_skill_payload(actions)
        min_actions = max(1, int(self.settings.dynamic_skill_min_actions))
        if len(actions) < min_actions:
            return

        had_failures = any(not item.get("success", True) for item in step_trace)
        if not had_failures and len(actions) < max(min_actions, 4):
            return

        try:
            name_seed = (current_step.description if current_step else "") or user_prompt or "task"
            name = (self.skill_composer._slugify(name_seed)[:40] + "-auto") if name_seed else f"macro-auto-{int(time.time())}"
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
            verification_contract = self.skill_composer._derive_skill_verification_contract(current_step)
            preconditions, grounding_signature = self.skill_composer._skill_context_metadata(actions)
            self.memory.save_skill(
                name=name,
                description=description,
                actions=actions,
                tags=tags,
                source_prompt=user_prompt,
                plan_step_id=current_step.id if current_step else None,
                parameters=parameters,
                verification_contract=verification_contract,
                preconditions=preconditions,
                grounding_signature=grounding_signature,
            )
            self.logger.info(
                "Synthesized procedural skill '%s' with %d actions (had_failures=%s)",
                name,
                len(actions),
                had_failures,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("Failed to synthesize dynamic skill: %s", exc)












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
            name = self.skill_composer._slugify(name_seed)[:50] or f"macro-{int(time.time())}"
            description = ""
            if current_step:
                description = current_step.success_criteria or current_step.description or ""
            if not description:
                description = user_prompt or ""
            tags = ["macro"]
            if current_step and current_step.id:
                tags.append(f"step:{current_step.id}")
            raw_actions = [dict(a) for a in action.get("actions") or []]
            composable_actions, parameters = self.skill_composer._build_composable_skill_payload(raw_actions)
            verification_contract = self.skill_composer._derive_skill_verification_contract(current_step, source_action=action)
            preconditions, grounding_signature = self.skill_composer._skill_context_metadata(composable_actions)
            self.memory.save_skill(
                name=name,
                description=description,
                actions=composable_actions,
                tags=tags,
                source_prompt=user_prompt,
                plan_step_id=current_step.id if current_step else None,
                parameters=parameters,
                verification_contract=verification_contract,
                preconditions=preconditions,
                grounding_signature=grounding_signature,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("Failed to save procedural skill: %s", exc)



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
        """Compact the oldest part of the history to keep the context manageable.

        Deterministic (no LLM call): history lines are structured, so counting
        and keeping notable events summarizes them for free instead of paying
        tokens now to save tokens later.
        """
        # Keep index 0 (usually plan_init or user_prompt)
        # Compact index 1 to 20
        # Keep rest
        if len(state.history) < 60:
            return

        chunk_to_summarize = state.history[1:21]
        summary = self._compact_history_chunk(chunk_to_summarize)
        if summary:
            self.logger.info("Compressing history: %s items -> 1 summary", len(chunk_to_summarize))
            new_history = [state.history[0]] + [f"history_summary:{summary}"] + state.history[21:]
            state.history = new_history

    @staticmethod
    def _compact_history_chunk(chunk: List[str]) -> str:
        """Deterministic one-line summary of structured history lines."""
        action_counts: Dict[str, int] = {}
        action_failures = 0
        verification_failures = 0
        observations = 0
        notable: List[str] = []
        for line in chunk:
            if line.startswith("action:"):
                match = re.search(r"'type': '([^']+)'", line)
                action_type = match.group(1) if match else "unknown"
                action_counts[action_type] = action_counts.get(action_type, 0) + 1
                if "'success': False" in line:
                    action_failures += 1
            elif line.startswith(("verification_contract_failed", "verification_failure")):
                verification_failures += 1
            elif line.startswith("observation@"):
                observations += 1
            elif line.startswith(
                (
                    "stuck:",
                    "plan_revised",
                    "plan_step_completed",
                    "plan_step_failed",
                    "reflector_fail",
                    "user_answer",
                    "user_guidance",
                    "macro_truncated",
                    "done_challenged",
                )
            ):
                notable.append(line[:100])

        parts: List[str] = []
        if action_counts:
            rendered = ", ".join(
                f"{name}x{count}"
                for name, count in sorted(action_counts.items(), key=lambda kv: -kv[1])[:6]
            )
            failures_note = f", {action_failures} failed" if action_failures else ""
            parts.append(f"{sum(action_counts.values())} actions ({rendered}{failures_note})")
        if observations:
            parts.append(f"{observations} observations")
        if verification_failures:
            parts.append(f"{verification_failures} verification failures")
        if notable:
            parts.append("notable: " + " | ".join(notable[-4:]))
        return "; ".join(parts) or f"{len(chunk)} events"

    def _skill_prompt_threshold(self, strategy: str) -> float:
        """Prompt inclusion is advisory (the model still decides), so gate at a
        softer bar (0.75x) than the autonomous fast-path execution thresholds."""
        if strategy == "keyword":
            return float(self.settings.fast_path_min_keyword_score) * 0.75
        return float(self.settings.fast_path_min_vector_score) * 0.75

    def _resolve_element_references(self, action: dict, tags: List[Dict[str, Any]]) -> bool:
        """Translate element_id/target_gid to x/y (physical px) using the most recent overlay tags."""
        lookup: Dict[str, Dict[str, Any]] = {}
        gid_lookup: Dict[str, Dict[str, Any]] = {}
        for tag in tags:
            if tag.get("id") is not None:
                lookup[str(tag["id"])] = tag
            gid = tag.get("gid")
            if gid:
                gid_lookup[str(gid)] = tag

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
                if tag.get("gid"):
                    act["target_gid"] = tag.get("gid")
                if tag.get("source"):
                    act["grounding_source"] = tag.get("source")
                if tag.get("confidence") is not None:
                    act["grounding_confidence"] = tag.get("confidence")
                act["target_frame"] = dict(frame)
                return True
            except Exception:
                return False

        def _apply(act: dict) -> bool:
            if act.get("x") is not None and act.get("y") is not None:
                return True

            # The action policy may attach a fused-grounding target_gid; resolve it
            # directly to coordinates before falling back to numeric/semantic refs.
            target_gid = str(act.get("target_gid") or "").strip()
            if target_gid:
                tag = gid_lookup.get(target_gid)
                if tag:
                    return _annotate_from_tag(act, tag)

            raw_element_id = act.get("element_id")
            if raw_element_id is not None:
                token = str(raw_element_id).strip()
                if not token:
                    return True
                tag = lookup.get(token) or gid_lookup.get(token)
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
            tag = gid_lookup.get(ref)
            if tag:
                return _annotate_from_tag(act, tag)
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

    def _request_user_input(self, question: str, kind: str = "other") -> str | None:
        """Ask the human for information mid-task (not an approval prompt).

        Returns None when no interactive channel exists (HITL disabled or no
        TTY) so callers can degrade to autonomous behavior.
        """
        if not self.settings.enable_hitl_prompt:
            return None
        stdin = getattr(sys, "stdin", None)
        if not stdin or not hasattr(stdin, "isatty") or not stdin.isatty():
            self.logger.warning("ask_user requested but no interactive stdin is available.")
            return None
        try:
            print(f"\n[AGENT QUESTION - {kind}] {question}")
            answer = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        return answer or None

    def _refresh_grounding(self, state: StateManager) -> tuple[str, str, List[Dict[str, Any]], dict | None]:
        """
        Capture a new frame, hash, tags, and AX tree immediately.
        Used when element references fail so the next action proposal has fresh context.
        """
        if hasattr(self, "grounder"):
            grounding = self.grounder.observe(
                previous=state.last_grounding,
                force_vision=True,
                include_semantic=self.settings.enable_semantic,
                include_visual=True,
            )
            state.record_grounding(grounding)
            state.record_observation(
                grounding.overlay_b64 or grounding.screenshot_b64,
                changed=True,
                phash=grounding.frame_hash,
                note="grounding_refresh",
            )
            return grounding.screenshot_b64, grounding.frame_hash, grounding.som_tags, grounding.ax_tree

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
