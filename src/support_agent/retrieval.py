"""Hybrid semantic + lexical retrieval for support documentation."""

import re
from typing import Protocol

from pydantic import BaseModel

from .db import DocumentChunkRepository
from .embeddings import EmbeddingProvider
from .models import Ticket


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_path: str
    title: str
    category: str
    text: str
    similarity: float


class Retriever(Protocol):
    def retrieve(self, ticket: Ticket) -> list[RetrievedChunk]:
        ...


class SemanticRetriever:
    """Retrieve semantic candidates and rerank them using support-specific signals."""

    DEFAULT_CANDIDATE_K = 30
    SEMANTIC_WEIGHT = 0.55
    LEXICAL_WEIGHT = 0.10
    TITLE_WEIGHT = 0.12
    CONCEPT_WEIGHT = 0.08
    PHRASE_WEIGHT = 0.07
    INTENT_WEIGHT = 0.08

    STOPWORDS = {
        "a","an","and","are","as","at","be","been","but","by","can","could",
        "did","do","does","for","from","get","getting","give","given","has",
        "have","how","i","if","in","into","is","it","me","my","of","on","or",
        "our","please","should","that","the","their","them","there","this",
        "to","was","we","what","when","where","which","who","why","will",
        "with","would","you","your",
    }

    CONCEPTS = {
        "expiration": {"expiration","expire","expires","expired","expiry","expiring","expiration_date","expiration_time","expiry_date","expiry_time"},
        "active": {"active","inactive","activation","deactivate","deactivated","remain","remains","stays","stay","available","availability","valid","validity","lifespan","duration","enabled","disabled"},
        "test": {"test","tests","assessment","assessments"},
        "invitation": {"invite","invites","invitation","invitations","invited"},
        "variant": {"variant","variants"},
        "account": {"account","accounts","profile","user","users"},
        "community": {"community"},
        "delete": {"delete","deletion","remove","removal","erase","close","deactivate"},
        "login": {"login","log","signin","sign","authenticate","authentication","sso"},
        "google": {"google","workspace"},
        "candidate": {"candidate","candidates","applicant","applicants"},
        "invite": {"invite","invites","invitation","invitations","reinvite","reinvitation"},
        "time_accommodation": {"accommodation","accommodations","extra_time","additional_time","extended_time","minutes","percentage","percent"},
        "reinvite": {"reinvite","reinvitation","resend"},
    }

    PHRASES = {
        "stay active": {"expiration","active","test"},
        "remain active": {"expiration","active","test"},
        "how long": {"expiration","duration"},
        "active in the system": {"expiration","active","test"},
        "test active": {"expiration","active","test"},
        "test expiration": {"expiration","test"},
        "test expiry": {"expiration","test"},
        "expiration time": {"expiration","duration"},
        "expiry time": {"expiration","duration"},
        "test lifespan": {"expiration","active","test"},
        "test duration": {"expiration","duration","test"},
        "invitation expiry": {"expiration","invitation"},
        "invite expiry": {"expiration","invitation"},
        "test invitation": {"invitation","test"},
        "delete account": {"delete","account"},
        "deleting account": {"delete","account"},
        "remove account": {"delete","account"},
        "close account": {"delete","account"},
        "community account": {"account","community"},
        "google login": {"login","google","account"},
        "google workspace": {"login","google","account"},
        "test variant": {"test","variant"},
        "test variants": {"test","variant"},
        "time accommodation": {"time_accommodation"},
        "extra time": {"time_accommodation"},
        "additional time": {"time_accommodation"},
        "reinvite candidate": {"candidate","reinvite"},
        "reinvite the candidate": {"candidate","reinvite"},
        "send new invitation": {"reinvite","invite"},
    }

    INTENTS = {
        "test_expiration": (
            {"how long do tests stay active","how long does a test stay active","how long do tests remain active",
             "how long does the test remain active","test active in the system","test expiration","test expiry","test expiration time","test lifespan"},
            {"test","expiration","active"},
            {"test","expiration","expire","expiry"},
            {"invite","invitation"},
        ),
        "invitation_expiration": (
            {"invitation expiry","invitation expiration","invite expiry","invite expiration","how long invitations remain valid"},
            {"invitation","expiration"},
            {"invite","invitation","expiry","expiration"},
            set(),
        ),
        "test_variant": (
            {"test variant","test variants","new test versus variant","new test vs variant","test vs variant","difference between test and variant","difference between creating a new test and creating a test variant"},
            {"test","variant"},
            {"test","variant"},
            set(),
        ),
        "account_deletion": (
            {"delete account","deleting account","delete my account","remove account","close account","community account"},
            {"account","delete"},
            {"account","delete","deletion"},
            set(),
        ),
        "community_account": (
            {"community account","hackerrank community account","personal hackerrank account"},
            {"account","community"},
            {"account","community"},
            set(),
        ),
        "google_account": (
            {"google login","google account","google workspace","created using google","created with google"},
            {"account","login","google"},
            {"google","login","account"},
            set(),
        ),
        "time_accommodation": (
            {"extra time","additional time","time accommodation","50% extra time","give candidate more time","extend test time"},
            {"candidate","time_accommodation"},
            {"time","accommodation","candidate"},
            set(),
        ),
        "candidate_reinvite": (
            {"reinvite candidate","reinvite the candidate","send new invitation","invite candidate again"},
            {"candidate","invite","reinvite"},
            {"invite","candidate","reinvite"},
            set(),
        ),
    }

    def __init__(self, embedding_provider: EmbeddingProvider, repository: DocumentChunkRepository,
                 top_k: int = 5, similarity_threshold: float = 0.0,
                 candidate_k: int = DEFAULT_CANDIDATE_K):
        if top_k < 1 or candidate_k < 1:
            raise ValueError("top_k and candidate_k must be positive")
        if candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        self.embedding_provider = embedding_provider
        self.repository = repository
        self.top_k = top_k
        self.threshold = similarity_threshold
        self.candidate_k = candidate_k

    def retrieve(self, ticket: Ticket) -> list[RetrievedChunk]:
        query = self._query(ticket)
        if not query:
            return []
        embedding = self.embedding_provider.embed_query(query)
        if not embedding:
            return []

        rows = self.repository.similarity_search(embedding, match_count=self.candidate_k)
        candidates = [
            RetrievedChunk(
                chunk_id=row.get("id", ""),
                source_path=row.get("source_path", ""),
                title=row.get("title", ""),
                category=row.get("category", ""),
                text=row.get("chunk_text", ""),
                similarity=self._float(row.get("similarity")),
            )
            for row in rows
            if row.get("source_path") and self._float(row.get("similarity")) >= self.threshold
        ]
        if not candidates:
            return []

        q_tokens = self._tokens(query)
        q_concepts = self._concepts(query)
        q_phrases = self._phrases(query)
        q_intents = self._intents(query)

        ranked = []
        for chunk in candidates:
            score = (
                self.SEMANTIC_WEIGHT * chunk.similarity
                + self.LEXICAL_WEIGHT * self._lexical(q_tokens, chunk)
                + self.TITLE_WEIGHT * self._title(q_tokens, chunk.title)
                + self.CONCEPT_WEIGHT * self._concept(q_concepts, chunk)
                + self.PHRASE_WEIGHT * self._phrase(q_phrases, chunk)
                + self.INTENT_WEIGHT * self._intent(q_intents, chunk)
            )
            ranked.append((score, chunk))

        ranked.sort(key=lambda x: x[0], reverse=True)

        selected, seen = [], set()
        for _, chunk in ranked:
            key = chunk.source_path.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            selected.append(chunk)
            if len(selected) == self.top_k:
                break
        return selected

    @staticmethod
    def _query(ticket: Ticket) -> str:
        subject = (ticket.subject or "").strip()
        issue = (ticket.issue or "").strip()
        return f"{subject}. {issue}".strip(". ") if subject and issue else subject or issue

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        return {
            w for w in re.findall(r"[a-zA-Z0-9]+", text.lower())
            if len(w) > 1 and w not in cls.STOPWORDS
        }

    @classmethod
    def _concepts(cls, text: str) -> set[str]:
        text = re.sub(r"\s+", " ", text.lower())
        found = set()
        tokens = cls._tokens(text)
        for phrase, concepts in cls.PHRASES.items():
            if phrase in text:
                found |= concepts
        for name, terms in cls.CONCEPTS.items():
            if tokens & terms:
                found.add(name)
        return found

    @classmethod
    def _phrases(cls, text: str) -> set[str]:
        text = re.sub(r"\s+", " ", text.lower())
        return {p for p in cls.PHRASES if p in text}

    @classmethod
    def _intents(cls, text: str) -> set[str]:
        normalized = re.sub(r"\s+", " ", text.lower())
        concepts = cls._concepts(text)
        found = set()
        for name, (phrases, required, _, _) in cls.INTENTS.items():
            if any(p in normalized for p in phrases) or (
                required and len(concepts & required) / len(required) >= 0.67
            ):
                found.add(name)
        if {"delete", "account"} <= concepts:
            found.add("account_deletion")
        if {"community", "account"} <= concepts:
            found.add("community_account")
        if {"google", "login"} <= concepts:
            found.add("google_account")
        if "time_accommodation" in concepts and "candidate" in concepts:
            found.add("time_accommodation")
        if "reinvite" in concepts and ({"candidate","invite"} & concepts):
            found.add("candidate_reinvite")
        return found

    @classmethod
    def _lexical(cls, query: set[str], chunk: RetrievedChunk) -> float:
        if not query:
            return 0.0
        title = cls._tokens(chunk.title)
        category = cls._tokens(chunk.category)
        body = cls._tokens(chunk.text)
        score = 3 * len(query & title) + 1.5 * len(query & category) + len(query & body)
        return min(score / (3 * len(query)), 1.0)

    @classmethod
    def _title(cls, query: set[str], title: str) -> float:
        if not query or not title:
            return 0.0
        overlap = len(query & cls._tokens(title)) / len(query)
        title_l = title.lower()
        phrase_bonus = 0.3 if any(p in title_l for p in cls.PHRASES) else 0
        return min(overlap + phrase_bonus, 1.0)

    @classmethod
    def _concept(cls, query: set[str], chunk: RetrievedChunk) -> float:
        if not query:
            return 0.0
        title = cls._concepts(chunk.title)
        category = cls._concepts(chunk.category)
        body = cls._concepts(chunk.text)
        matched = sum(3 if c in title else 2 if c in category else 1 if c in body else 0 for c in query)
        return min(matched / (3 * len(query)), 1.0)

    @classmethod
    def _phrase(cls, phrases: set[str], chunk: RetrievedChunk) -> float:
        if not phrases:
            return 0.0
        title, category, body = chunk.title.lower(), chunk.category.lower(), chunk.text.lower()
        score = sum(1 if p in title else 0.6 if p in category else 0.3 if p in body else 0 for p in phrases)
        return min(score / len(phrases), 1.0)

    @classmethod
    def _intent(cls, intents: set[str], chunk: RetrievedChunk) -> float:
        if not intents:
            return 0.0
        title_tokens = cls._tokens(chunk.title)
        doc = f"{chunk.title.lower()} {chunk.category.lower()} {chunk.text.lower()}"
        scores = []
        for name in intents:
            definition = cls.INTENTS.get(name)
            if not definition:
                continue
            phrases, concepts, preferred, avoid = definition
            score = 0.5 * (len(title_tokens & preferred) / len(preferred) if preferred else 0)
            score += 0.35 * (len(cls._concepts(doc) & concepts) / len(concepts) if concepts else 0)
            score += 0.2 if any(p in doc for p in phrases) else 0
            if avoid & title_tokens:
                score -= 0.3
            scores.append(max(0.0, min(score, 1.0)))
        return min(sum(scores) / len(intents), 1.0) if scores else 0.0

    @staticmethod
    def _float(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0