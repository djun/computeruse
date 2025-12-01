from __future__ import annotations

import json
import hashlib
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from macos_cua_agent.utils.text import tokenize_lower

def _fingerprint_actions(actions: List[Dict[str, Any]]) -> str:
    """Stable hash of a macro action list for deduplication."""
    canonical = json.dumps(actions, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
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

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ProceduralSkill":
        return cls(
            id=raw.get("id", str(uuid.uuid4())),
            name=raw.get("name", "unnamed"),
            description=raw.get("description", ""),
            actions=raw.get("actions", []),
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
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    ) -> ProceduralSkill:
        """Persist a skill; deduplicate by action fingerprint."""
        cleaned_actions = [dict(a) for a in (actions or []) if isinstance(a, dict)]
        if not cleaned_actions:
            raise ValueError("skill actions cannot be empty")

        fingerprint = _fingerprint_actions(cleaned_actions)
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
