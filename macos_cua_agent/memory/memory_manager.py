from __future__ import annotations

import json
import time
import uuid
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from macos_cua_agent.utils.config import Settings
from macos_cua_agent.utils.logger import get_logger
from macos_cua_agent.memory.skill_store import SkillStore, ProceduralSkill
from macos_cua_agent.utils.text import tokenize_lower


@dataclass
class Episode:
    id: str
    created_at: float
    user_prompt: str
    plan: Dict[str, Any]
    outcome: str
    summary: str
    tags: List[str]
    raw_log_path: str | None = None


@dataclass
class SemanticMemoryItem:
    id: str
    created_at: float
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class MemoryManager:
    """File-backed episodic and semantic memory."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger(__name__, level=settings.log_level)
        self.root = Path(settings.memory_root or ".agent_memory")
        self.episodes_dir = self.root / "episodes"
        self.semantic_dir = self.root / "semantic"
        self.logs_dir = self.root / "logs"
        self.skills_dir = self.root / "skills"
        self.embed_client = self._build_embed_client()
        for path in (self.root, self.episodes_dir, self.semantic_dir, self.logs_dir, self.skills_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.skill_store = SkillStore(self.skills_dir, self.logger)

    def _build_embed_client(self) -> Optional[Any]:
        if not self.settings.enable_embeddings:
            return None
        api_key = self.settings.embedding_api_key
        if not api_key:
            self.logger.info("Embedding disabled: EMBEDDING_API_KEY/OPENAI_API_KEY missing.")
            return None
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            self.logger.warning("openai package unavailable for embeddings: %s", exc)
            return None
        return OpenAI(base_url=self.settings.embedding_base_url, api_key=api_key)

    def save_episode(self, episode: Episode) -> Path:
        path = self.episodes_dir / f"{episode.id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(episode), handle, ensure_ascii=False, indent=2)
        return path

    def list_episodes(self) -> List[Episode]:
        episodes: List[Episode] = []
        for path in self.episodes_dir.glob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                episodes.append(Episode(**raw))
            except Exception as exc:
                self.logger.warning("Failed to load episode %s: %s", path, exc)
        episodes.sort(key=lambda ep: ep.created_at)
        return episodes

    def add_semantic_item(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> SemanticMemoryItem:
        embedding: Optional[List[float]] = None
        if self.embed_client:
            embedding = self._embed_text(text)
        item = SemanticMemoryItem(
            id=str(uuid.uuid4()),
            created_at=time.time(),
            text=text,
            metadata=metadata or {},
            embedding=embedding,
        )
        path = self.semantic_dir / f"{item.id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(item), handle, ensure_ascii=False, indent=2)
        return item

    def search_semantic(self, query: str, top_k: int = 5) -> List[SemanticMemoryItem]:
        # Prefer vector similarity when embeddings are available; otherwise fallback to keyword search.
        items: List[SemanticMemoryItem] = []
        for path in self.semantic_dir.glob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                if "embedding" not in raw:
                    raw["embedding"] = None
                items.append(SemanticMemoryItem(**raw))
            except Exception as exc:
                self.logger.warning("Failed to read semantic memory %s: %s", path, exc)

        if not items:
            return []

        if self.embed_client:
            query_embedding = self._embed_text(query)
            if query_embedding:
                scored = [
                    (self._cosine_similarity(query_embedding, item.embedding), item)
                    for item in items
                    if item.embedding
                ]
                scored.sort(key=lambda pair: pair[0], reverse=True)
                return [item for _, item in scored[:top_k] if item]

        lowered = query.lower()
        filtered = [item for item in items if lowered in item.text.lower()]
        return filtered[:top_k]

    def _embed_text(self, text: str) -> Optional[List[float]]:
        if not self.embed_client:
            return None
        try:
            response = self.embed_client.embeddings.create(
                model=self.settings.embedding_model,
                input=text,
            )
            vector = response.data[0].embedding if response and response.data else None
            if vector and isinstance(vector, list):
                return [float(v) for v in vector]
        except Exception as exc:
            self.logger.warning("Embedding request failed; continuing without vector search: %s", exc)
        return None

    def _cosine_similarity(self, a: List[float], b: Optional[List[float]]) -> float:
        if not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _extract_semantic_hints(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Capture lightweight semantic hints for later matching: roles, labels, action types.
        This enables future generalization beyond raw coordinates.
        """
        hints: Dict[str, Any] = {"roles": [], "labels": [], "types": [], "paths": []}
        for act in actions or []:
            act_type = act.get("type")
            if act_type:
                hints["types"].append(act_type)
            role = act.get("role") or act.get("semantic_role")
            label = act.get("label") or act.get("semantic_label")
            path = act.get("semantic_path") or act.get("path")
            if role:
                hints["roles"].append(role)
            if label:
                hints["labels"].append(label)
            if path:
                hints["paths"].append(path)
                # Also break path into crumbs so we can match partial hierarchy later.
                for crumb in str(path).split(">"):
                    crumb_clean = crumb.strip()
                    if crumb_clean:
                        hints["labels"].append(crumb_clean)
        hints["roles"] = sorted(set(hints["roles"]))
        hints["labels"] = sorted(set(hints["labels"]))
        hints["types"] = sorted(set(hints["types"]))
        hints["paths"] = sorted(set(hints["paths"]))
        return hints

    # Procedural skill helpers
    def save_skill(
        self,
        name: str,
        description: str,
        actions: List[Dict[str, Any]],
        tags: Optional[List[str]] = None,
        source_prompt: Optional[str] = None,
        plan_step_id: Optional[str] = None,
    ) -> ProceduralSkill:
        embedding = None
        semantic_hints = self._extract_semantic_hints(actions)
        if self.embed_client:
            text_for_embed = f"{name}\n{description}\n{semantic_hints}"
            embedding = self._embed_text(text_for_embed)
        return self.skill_store.save_skill(
            name=name,
            description=description,
            actions=actions,
            tags=tags,
            source_prompt=source_prompt,
            plan_step_id=plan_step_id,
            embedding=embedding,
            semantic_hints=semantic_hints,
        )

    def list_skills(self) -> List[ProceduralSkill]:
        return self.skill_store.list_skills()

    def get_skill(self, skill_id_or_name: str) -> Optional[ProceduralSkill]:
        return self.skill_store.get_skill(skill_id_or_name)

    def record_skill_usage(self, skill_id: str) -> Optional[ProceduralSkill]:
        return self.skill_store.record_usage(skill_id)

    def search_skills(self, query: str, top_k: int = 5) -> List[ProceduralSkill]:
        skills = self.list_skills()
        if not skills:
            return []

        # If embeddings are available, use vector search; otherwise use improved keyword scoring.
        if self.embed_client:
            query_embedding = self._embed_text(query)
            if query_embedding:
                scored = []
                for skill in skills:
                    if not skill.embedding:
                        continue
                    sim = self._cosine_similarity(query_embedding, skill.embedding)
                    scored.append((sim, skill))
                scored.sort(key=lambda x: x[0], reverse=True)
                return [s for _, s in scored[:top_k]]

        # Keyword fallback with token overlap
        query_tokens = tokenize_lower(query)
        scored = []
        for skill in skills:
            hint_text = ""
            if getattr(skill, "semantic_hints", None):
                hints = skill.semantic_hints
                hint_text = " ".join(
                    (hints.get("roles") or []) + (hints.get("labels") or []) + (hints.get("types") or [])
                )
            text = (skill.name + " " + skill.description + " " + " ".join(skill.tags) + " " + hint_text).lower()
            skill_tokens = tokenize_lower(text)
            overlap = len(query_tokens & skill_tokens)
            exact = 5 if query.lower() in text else 0
            score = exact + overlap
            if score > 0:
                scored.append((score, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]
