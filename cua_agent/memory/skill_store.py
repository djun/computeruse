"""Persistent store for procedural skills (macros)."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

def _fingerprint_actions(
    actions: List[Dict[str, Any]],
    semantic_hints: Optional[Dict[str, Any]] = None,
    verification_contract: Optional[Dict[str, Any]] = None,
    preconditions: Optional[Dict[str, Any]] = None,
) -> str:
    """Stable contextual hash of a macro for deduplication."""
    payload = {
        "actions": actions,
        "semantic_hints": semantic_hints or {},
        "verification_contract": verification_contract or {},
        "preconditions": preconditions or {},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


@dataclass
class ProceduralSkill:
    id: str
    name: str
    description: str
    actions: List[Dict[str, Any]]
    created_at: float
    updated_at: float
    usage_count: int = 0
    last_used: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    fingerprint: str = ""
    source_prompt: Optional[str] = None
    plan_step_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    semantic_hints: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    verification_contract: Dict[str, Any] = field(default_factory=dict)
    preconditions: Dict[str, Any] = field(default_factory=dict)
    postconditions: Dict[str, Any] = field(default_factory=dict)
    negative_examples: List[Dict[str, Any]] = field(default_factory=list)
    grounding_signature: Dict[str, Any] = field(default_factory=dict)
    failure_count: int = 0
    success_count: int = 0

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ProceduralSkill":
        actions = raw.get("actions")
        if not actions and isinstance(raw.get("action_sequence"), list):
            actions = raw.get("action_sequence")
        parameters = raw.get("parameters", {}) or {}
        if not isinstance(parameters, dict):
            parameters = {}
        verification_contract = raw.get("verification_contract", {}) or raw.get("verification", {}) or {}
        if not isinstance(verification_contract, dict):
            verification_contract = {}
        preconditions = raw.get("preconditions", {}) or {}
        if not isinstance(preconditions, dict):
            preconditions = {}
        postconditions = raw.get("postconditions", {}) or {}
        if not isinstance(postconditions, dict):
            postconditions = {}
        grounding_signature = raw.get("grounding_signature", {}) or {}
        if not isinstance(grounding_signature, dict):
            grounding_signature = {}
        negative_examples = raw.get("negative_examples", []) or []
        if not isinstance(negative_examples, list):
            negative_examples = []

        return cls(
            id=raw.get("id", str(uuid.uuid4())),
            name=raw.get("name", "unnamed"),
            description=raw.get("description", ""),
            actions=actions or [],
            created_at=float(raw.get("created_at", time.time())),
            updated_at=float(raw.get("updated_at", time.time())),
            usage_count=int(raw.get("usage_count", 0)),
            last_used=raw.get("last_used"),
            tags=list(raw.get("tags", []) or []),
            fingerprint=raw.get("fingerprint", ""),
            source_prompt=raw.get("source_prompt"),
            plan_step_id=raw.get("plan_step_id"),
            embedding=raw.get("embedding"),
            semantic_hints=raw.get("semantic_hints", {}) or {},
            parameters=parameters,
            verification_contract=verification_contract,
            preconditions=preconditions,
            postconditions=postconditions,
            negative_examples=[item for item in negative_examples if isinstance(item, dict)],
            grounding_signature=grounding_signature,
            failure_count=int(raw.get("failure_count", 0) or 0),
            success_count=int(raw.get("success_count", 0) or 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        # External format compatibility: some tooling expects this key.
        payload["action_sequence"] = [dict(a) for a in self.actions]
        return payload


class SkillStore:
    """File-backed store of procedural skills/macros."""

    def __init__(self, root: Path, logger) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def save_skill(
        self,
        name: str,
        description: str,
        actions: List[Dict[str, Any]],
        tags: Optional[List[str]] = None,
        source_prompt: Optional[str] = None,
        plan_step_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        semantic_hints: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        verification_contract: Optional[Dict[str, Any]] = None,
        preconditions: Optional[Dict[str, Any]] = None,
        postconditions: Optional[Dict[str, Any]] = None,
        negative_examples: Optional[List[Dict[str, Any]]] = None,
        grounding_signature: Optional[Dict[str, Any]] = None,
    ) -> ProceduralSkill:
        """Persist a skill; deduplicate by action fingerprint."""
        cleaned_actions = [dict(a) for a in (actions or []) if isinstance(a, dict)]
        if not cleaned_actions:
            raise ValueError("skill actions cannot be empty")
        cleaned_parameters = dict(parameters) if isinstance(parameters, dict) else {}
        cleaned_verification = (
            dict(verification_contract) if isinstance(verification_contract, dict) else {}
        )
        cleaned_preconditions = dict(preconditions) if isinstance(preconditions, dict) else {}
        cleaned_postconditions = dict(postconditions) if isinstance(postconditions, dict) else {}
        cleaned_grounding_signature = (
            dict(grounding_signature) if isinstance(grounding_signature, dict) else {}
        )
        cleaned_negative_examples = [
            dict(item) for item in (negative_examples or []) if isinstance(item, dict)
        ]

        fingerprint = _fingerprint_actions(
            cleaned_actions,
            semantic_hints=semantic_hints or {},
            verification_contract=cleaned_verification,
            preconditions=cleaned_preconditions,
        )
        existing = self._find_by_fingerprint(fingerprint)
        now = time.time()
        if existing:
            existing.updated_at = now
            existing.usage_count += 1
            if tags:
                merged = set(existing.tags) | set(tags)
                existing.tags = sorted(merged)
            if description and not existing.description:
                existing.description = description
            if embedding:
                existing.embedding = embedding
            if semantic_hints:
                existing.semantic_hints = semantic_hints
            if cleaned_parameters:
                existing.parameters = cleaned_parameters
            if cleaned_verification:
                existing.verification_contract = cleaned_verification
            if cleaned_preconditions:
                existing.preconditions = cleaned_preconditions
            if cleaned_postconditions:
                existing.postconditions = cleaned_postconditions
            if cleaned_grounding_signature:
                existing.grounding_signature = cleaned_grounding_signature
            if cleaned_negative_examples:
                existing.negative_examples.extend(cleaned_negative_examples)
            self._write(existing)
            return existing

        skill_id = str(uuid.uuid4())
        skill = ProceduralSkill(
            id=skill_id,
            name=name or f"skill-{skill_id[:8]}",
            description=description or "",
            actions=cleaned_actions,
            created_at=now,
            updated_at=now,
            tags=tags or [],
            fingerprint=fingerprint,
            source_prompt=source_prompt,
            plan_step_id=plan_step_id,
            embedding=embedding,
            semantic_hints=semantic_hints or {},
            parameters=cleaned_parameters,
            verification_contract=cleaned_verification,
            preconditions=cleaned_preconditions,
            postconditions=cleaned_postconditions,
            negative_examples=cleaned_negative_examples,
            grounding_signature=cleaned_grounding_signature,
        )
        self._write(skill)
        return skill

    def list_skills(self) -> List[ProceduralSkill]:
        skills: List[ProceduralSkill] = []
        for path in self.root.glob("*.json"):
            loaded = self._read(path)
            if loaded:
                skills.append(loaded)
        skills.sort(key=lambda s: s.created_at)
        return skills

    def get_skill(self, skill_id_or_name: str) -> Optional[ProceduralSkill]:
        if not skill_id_or_name:
            return None
        # First try by id filename
        by_id = self._read(self.root / f"{skill_id_or_name}.json")
        if by_id:
            return by_id
        # Fallback: scan for matching name
        for skill in self.list_skills():
            if skill.name == skill_id_or_name:
                return skill
        return None

    def record_usage(self, skill_id: str) -> Optional[ProceduralSkill]:
        skill = self.get_skill(skill_id)
        if not skill:
            return None
        skill.usage_count += 1
        skill.last_used = time.time()
        skill.updated_at = skill.last_used
        self._write(skill)
        return skill

    def record_result(
        self,
        skill_id: str,
        *,
        success: bool,
        negative_example: Optional[Dict[str, Any]] = None,
    ) -> Optional[ProceduralSkill]:
        skill = self.get_skill(skill_id)
        if not skill:
            return None
        if success:
            skill.success_count += 1
        else:
            skill.failure_count += 1
            if negative_example:
                skill.negative_examples.append(dict(negative_example))
                skill.negative_examples = skill.negative_examples[-20:]
        skill.updated_at = time.time()
        self._write(skill)
        return skill

    def _find_by_fingerprint(self, fingerprint: str) -> Optional[ProceduralSkill]:
        for skill in self.list_skills():
            if skill.fingerprint == fingerprint:
                return skill
        return None

    def _write(self, skill: ProceduralSkill) -> None:
        path = self.root / f"{skill.id}.json"
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(skill.to_dict(), handle, ensure_ascii=False, indent=2)
        except Exception as exc:
            self.logger.warning("Failed to write skill %s: %s", skill.id, exc)

    def _read(self, path: Path) -> Optional[ProceduralSkill]:
        try:
            if not path.exists():
                return None
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            return ProceduralSkill.from_dict(raw)
        except Exception as exc:
            self.logger.warning("Failed to read skill %s: %s", path, exc)
            return None
