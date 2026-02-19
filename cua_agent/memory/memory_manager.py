"""File-backed episodic, semantic, and procedural-skill memory."""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from cua_agent.memory.skill_store import ProceduralSkill, SkillStore
from cua_agent.utils.config import Settings
from cua_agent.utils.logger import get_logger
from cua_agent.utils.text import tokenize_lower


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


@dataclass
class SkillSearchResult:
    skill: ProceduralSkill
    score: float
    strategy: str  # chroma|vector|keyword


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
        self.skill_vector_collection = self._build_skill_vector_collection()
        if self.skill_vector_collection:
            self._sync_skill_vector_collection()

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

    def _build_skill_vector_collection(self) -> Optional[Any]:
        """Optional local vector index for skills (ChromaDB-backed)."""
        if not self.settings.enable_chroma_skills:
            return None
        if not self.settings.enable_embeddings:
            self.logger.info("Chroma skill index disabled: ENABLE_EMBEDDINGS is false.")
            return None
        try:
            import chromadb  # type: ignore
        except Exception as exc:
            self.logger.warning("Chroma skill index unavailable: %s", exc)
            return None

        persist_dir = Path(self.settings.chroma_persist_dir or (self.root / "chroma"))
        try:
            persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(persist_dir))
            return client.get_or_create_collection(
                name=self.settings.chroma_skills_collection,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            self.logger.warning("Failed to initialize Chroma skill index: %s", exc)
            return None

    def _sync_skill_vector_collection(self) -> None:
        if not self.skill_vector_collection:
            return
        for skill in self.skill_store.list_skills():
            self._upsert_skill_vector(skill)

    def _upsert_skill_vector(self, skill: ProceduralSkill) -> None:
        if not self.skill_vector_collection:
            return
        if not skill.embedding:
            return
        document = (
            f"{skill.name}\n"
            f"{skill.description}\n"
            f"{skill.semantic_hints or {}}\n"
            f"{skill.parameters or {}}\n"
            f"{skill.verification_contract or {}}"
        )
        metadata = {
            "name": str(skill.name or "")[:200],
            "plan_step_id": str(skill.plan_step_id or ""),
            "source_prompt": str(skill.source_prompt or "")[:500],
            "usage_count": int(skill.usage_count),
        }
        try:
            self.skill_vector_collection.upsert(
                ids=[skill.id],
                embeddings=[skill.embedding],
                documents=[document],
                metadatas=[metadata],
            )
        except Exception as exc:
            self.logger.warning("Failed to upsert skill %s into vector index: %s", skill.id, exc)

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
        parameters: Optional[Dict[str, Any]] = None,
        verification_contract: Optional[Dict[str, Any]] = None,
    ) -> ProceduralSkill:
        embedding = None
        semantic_hints = self._extract_semantic_hints(actions)
        if self.embed_client:
            text_for_embed = (
                f"{name}\n"
                f"{description}\n"
                f"{semantic_hints}\n"
                f"{parameters or {}}\n"
                f"{verification_contract or {}}"
            )
            embedding = self._embed_text(text_for_embed)
        skill = self.skill_store.save_skill(
            name=name,
            description=description,
            actions=actions,
            tags=tags,
            source_prompt=source_prompt,
            plan_step_id=plan_step_id,
            embedding=embedding,
            semantic_hints=semantic_hints,
            parameters=parameters,
            verification_contract=verification_contract,
        )
        self._upsert_skill_vector(skill)
        return skill

    def list_skills(self) -> List[ProceduralSkill]:
        return self.skill_store.list_skills()

    def get_skill(self, skill_id_or_name: str) -> Optional[ProceduralSkill]:
        return self.skill_store.get_skill(skill_id_or_name)

    def record_skill_usage(self, skill_id: str) -> Optional[ProceduralSkill]:
        skill = self.skill_store.record_usage(skill_id)
        if skill:
            self._upsert_skill_vector(skill)
        return skill

    def search_skills(self, query: str, top_k: int = 5) -> List[ProceduralSkill]:
        return [res.skill for res in self.search_skills_scored(query=query, top_k=top_k)]

    def search_skills_scored(self, query: str, top_k: int = 5) -> List[SkillSearchResult]:
        if not query:
            return []
        skills = self.list_skills()
        if not skills:
            return []

        # Prefer Chroma local vector index when available.
        if self.skill_vector_collection and self.embed_client:
            query_embedding = self._embed_text(query)
            if query_embedding:
                chroma_results = self._search_skills_chroma(query_embedding, top_k=top_k)
                if chroma_results:
                    return chroma_results

        # If embeddings are available, use direct vector search fallback.
        if self.embed_client:
            query_embedding = self._embed_text(query)
            if query_embedding:
                scored = []
                for skill in skills:
                    if not skill.embedding:
                        continue
                    sim = self._cosine_similarity(query_embedding, skill.embedding)
                    scored.append(SkillSearchResult(skill=skill, score=sim, strategy="vector"))
                scored.sort(key=lambda x: x.score, reverse=True)
                if scored:
                    return scored[:top_k]

        # Keyword fallback with token overlap.
        query_tokens = tokenize_lower(query)
        query_lower = query.lower()
        scored_keywords: List[SkillSearchResult] = []
        for skill in skills:
            score = self._keyword_skill_score(skill, query_tokens, query_lower)
            if score > 0:
                scored_keywords.append(SkillSearchResult(skill=skill, score=float(score), strategy="keyword"))

        scored_keywords.sort(key=lambda x: x.score, reverse=True)
        return scored_keywords[:top_k]

    def select_fast_path_skill(
        self,
        query: str,
        top_k: int = 3,
        min_vector_score: Optional[float] = None,
        min_keyword_score: Optional[float] = None,
    ) -> Optional[SkillSearchResult]:
        min_vector = (
            float(min_vector_score)
            if min_vector_score is not None
            else float(self.settings.fast_path_min_vector_score)
        )
        min_keyword = (
            float(min_keyword_score)
            if min_keyword_score is not None
            else float(self.settings.fast_path_min_keyword_score)
        )
        results = self.search_skills_scored(query=query, top_k=top_k)
        if not results:
            return None

        best = results[0]
        if best.strategy in {"chroma", "vector"} and best.score >= min_vector:
            return best
        if best.strategy == "keyword" and best.score >= min_keyword:
            return best
        return None

    def _search_skills_chroma(self, query_embedding: List[float], top_k: int) -> List[SkillSearchResult]:
        if not self.skill_vector_collection:
            return []
        try:
            payload = self.skill_vector_collection.query(
                query_embeddings=[query_embedding],
                n_results=max(1, int(top_k)),
                include=["distances"],
            )
            ids = (payload.get("ids") or [[]])[0]
            distances = (payload.get("distances") or [[]])[0]
            results: List[SkillSearchResult] = []
            for skill_id, dist in zip(ids, distances):
                skill = self.get_skill(str(skill_id))
                if not skill:
                    continue
                try:
                    score = max(0.0, 1.0 - float(dist))
                except Exception:
                    score = 0.0
                results.append(SkillSearchResult(skill=skill, score=score, strategy="chroma"))
            results.sort(key=lambda x: x.score, reverse=True)
            return results
        except Exception as exc:
            self.logger.warning("Chroma skill query failed, falling back: %s", exc)
            return []

    def _keyword_skill_score(self, skill: ProceduralSkill, query_tokens: set[str], query_lower: str) -> int:
        hint_text = ""
        if getattr(skill, "semantic_hints", None):
            hints = skill.semantic_hints
            hint_text = " ".join(
                (hints.get("roles") or []) + (hints.get("labels") or []) + (hints.get("types") or [])
            )
        param_names = []
        if getattr(skill, "parameters", None):
            param_names = [str(key) for key in skill.parameters.keys()]
        verification_blob = ""
        if getattr(skill, "verification_contract", None):
            verification_blob = json.dumps(skill.verification_contract, ensure_ascii=False)
        text = (
            skill.name
            + " "
            + skill.description
            + " "
            + " ".join(skill.tags)
            + " "
            + hint_text
            + " "
            + " ".join(param_names)
            + " "
            + verification_blob
        ).lower()
        skill_tokens = tokenize_lower(text)
        overlap = len(query_tokens & skill_tokens)
        exact = 5 if query_lower in text else 0
        return exact + overlap
