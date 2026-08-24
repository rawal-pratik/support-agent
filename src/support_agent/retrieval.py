"""Hybrid semantic + lexical retrieval for support tickets."""

import re
from typing import Protocol

from pydantic import BaseModel

from .db import DocumentChunkRepository
from .embeddings import EmbeddingProvider
from .models import Ticket


class RetrievedChunk(BaseModel):
    """A document chunk retrieved for a ticket with similarity score."""

    chunk_id: str
    source_path: str
    title: str
    category: str
    text: str
    similarity: float


class Retriever(Protocol):
    """Protocol for ticket retrieval."""

    def retrieve(self, ticket: Ticket) -> list[RetrievedChunk]:
        """Return relevant chunks for a ticket."""
        ...


class SemanticRetriever:
    """
    Hybrid semantic + lexical retrieval.

    Retrieval pipeline:

        Ticket
          ↓
        Query embedding
          ↓
        Broad semantic retrieval
          ↓
        Query concept expansion
          ↓
        Hybrid reranking
          ↓
        Source/article deduplication
          ↓
        Final top-K results
    """

    DEFAULT_CANDIDATE_K = 20

    # Keep semantic similarity as the strongest signal, but give
    # lexical/conceptual evidence enough influence to correct cases
    # where the embedding model ranks generic articles too highly.
    SEMANTIC_WEIGHT = 0.65
    LEXICAL_WEIGHT = 0.15
    TITLE_WEIGHT = 0.10
    CONCEPT_WEIGHT = 0.10

    STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "get",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "please",
        "should",
        "that",
        "the",
        "their",
        "them",
        "there",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }

    # Support-domain terminology mappings.
    #
    # These are intentionally small and conservative. They are not
    # intended to replace semantic search. They help bridge common
    # differences between the words a user uses and the terminology
    # used by HackerRank documentation.
    CONCEPT_GROUPS = {
        "expiration": {
            "expiration",
            "expire",
            "expires",
            "expiry",
            "expiring",
            "expiration_date",
            "expiration_time",
            "expiry_date",
            "expiry_time",
            "active_duration",
            "test_lifespan",
        },
        "active": {
            "active",
            "inactive",
            "activation",
            "deactivate",
            "deactivated",
            "remain",
            "stays",
            "stay",
            "available",
            "availability",
            "valid",
            "validity",
            "lifespan",
            "duration",
        },
        "variant": {
            "variant",
            "variants",
        },
        "test": {
            "test",
            "tests",
            "assessment",
            "assessments",
        },
        "account": {
            "account",
            "accounts",
            "profile",
            "user",
            "users",
        },
        "delete": {
            "delete",
            "deletion",
            "remove",
            "removal",
            "erase",
            "close",
            "deactivate",
        },
        "login": {
            "login",
            "log",
            "signin",
            "sign",
            "authenticate",
            "authentication",
            "sso",
        },
        "payment": {
            "payment",
            "payments",
            "billing",
            "charge",
            "charges",
            "invoice",
            "invoices",
        },
        "refund": {
            "refund",
            "refunds",
            "reimbursement",
            "reimburse",
        },
        "subscription": {
            "subscription",
            "subscriptions",
            "plan",
            "plans",
            "cancel",
            "pause",
        },
        "candidate": {
            "candidate",
            "candidates",
            "applicant",
            "applicants",
        },
        "invite": {
            "invite",
            "invites",
            "invitation",
            "invitations",
            "reinvite",
            "reinvitation",
        },
        "time_accommodation": {
            "accommodation",
            "accommodations",
            "extra_time",
            "additional_time",
            "extended_time",
            "duration",
            "minutes",
        },
        "interview": {
            "interview",
            "interviews",
            "interviewer",
            "interviewers",
        },
        "security": {
            "security",
            "privacy",
            "gdpr",
            "infosec",
            "compliance",
        },
    }

    # Multi-word phrases are especially useful because support users
    # often describe an intent with a phrase rather than an exact
    # documentation keyword.
    PHRASE_CONCEPTS = {
        "stay active": {"expiration", "active"},
        "remain active": {"expiration", "active"},
        "how long": {"expiration", "duration"},
        "active in the system": {"expiration", "active"},
        "expire": {"expiration"},
        "expiry": {"expiration"},
        "test expiration": {"expiration", "test"},
        "expiration time": {"expiration", "duration"},
        "delete account": {"delete", "account"},
        "remove account": {"delete", "account"},
        "close account": {"delete", "account"},
        "community account": {"account"},
        "google login": {"login", "account"},
        "google workspace": {"login", "account"},
        "test variant": {"test", "variant"},
        "test variants": {"test", "variant"},
        "extra time": {"time_accommodation"},
        "more time": {"time_accommodation"},
        "time accommodation": {"time_accommodation"},
        "cancel subscription": {"subscription"},
        "pause subscription": {"subscription"},
        "payment issue": {"payment"},
        "payment problem": {"payment"},
        "request refund": {"refund"},
        "get refund": {"refund"},
    }

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        repository: DocumentChunkRepository,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        candidate_k: int = DEFAULT_CANDIDATE_K,
    ):
        """
        Args:
            embedding_provider:
                Provider used to create the retrieval query embedding.

            repository:
                Repository containing document chunks and vector search.

            top_k:
                Number of final chunks returned.

            similarity_threshold:
                Minimum semantic similarity required for a candidate.

            candidate_k:
                Number of semantic candidates retrieved before reranking.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be between 0.0 and 1.0"
            )

        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k"
            )

        self._embedding_provider = embedding_provider
        self._repository = repository
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold
        self._candidate_k = candidate_k

    def retrieve(self, ticket: Ticket) -> list[RetrievedChunk]:
        """
        Retrieve relevant documentation for a support ticket.

        The vector database first provides a broad candidate pool.
        Candidates are then reranked using:

        - semantic similarity
        - lexical overlap
        - title relevance
        - support-domain concept relevance

        Source-level deduplication happens AFTER reranking so that
        the strongest chunk from an article is selected first.
        """

        ticket_text = self._build_query_text(ticket)

        if not ticket_text:
            return []

        # --------------------------------------------------------------
        # 1. Create query embedding
        # --------------------------------------------------------------

        query_embedding = self._embedding_provider.embed_query(
            ticket_text
        )

        if not query_embedding:
            return []

        # --------------------------------------------------------------
        # 2. Broad semantic retrieval
        # --------------------------------------------------------------

        results = self._repository.similarity_search(
            query_embedding,
            match_count=self._candidate_k,
        )

        candidates: list[RetrievedChunk] = []

        for result in results:
            similarity = self._safe_float(
                result.get("similarity", 0.0)
            )

            if similarity < self._similarity_threshold:
                continue

            source_path = result.get("source_path", "")

            if not source_path:
                continue

            candidates.append(
                RetrievedChunk(
                    chunk_id=result.get("id", ""),
                    source_path=source_path,
                    title=result.get("title", ""),
                    category=result.get("category", ""),
                    text=result.get("chunk_text", ""),
                    similarity=similarity,
                )
            )

        if not candidates:
            return []

        # --------------------------------------------------------------
        # 3. Build lexical and conceptual query representation
        # --------------------------------------------------------------

        query_tokens = self._tokenize(ticket_text)
        query_concepts = self._expand_concepts(ticket_text)

        # --------------------------------------------------------------
        # 4. Hybrid reranking BEFORE deduplication
        # --------------------------------------------------------------

        scored_candidates: list[
            tuple[float, RetrievedChunk]
        ] = []

        for chunk in candidates:
            lexical_score = self._lexical_score(
                query_tokens=query_tokens,
                chunk=chunk,
            )

            title_score = self._title_score(
                query_tokens=query_tokens,
                title=chunk.title,
            )

            concept_score = self._concept_score(
                query_concepts=query_concepts,
                chunk=chunk,
            )

            final_score = (
                self.SEMANTIC_WEIGHT * chunk.similarity
                + self.LEXICAL_WEIGHT * lexical_score
                + self.TITLE_WEIGHT * title_score
                + self.CONCEPT_WEIGHT * concept_score
            )

            scored_candidates.append(
                (final_score, chunk)
            )

        scored_candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # --------------------------------------------------------------
        # 5. Deduplicate by source AFTER reranking
        # --------------------------------------------------------------

        selected: list[RetrievedChunk] = []
        seen_sources: set[str] = set()

        for _, chunk in scored_candidates:
            if chunk.source_path in seen_sources:
                continue

            seen_sources.add(chunk.source_path)
            selected.append(chunk)

            if len(selected) >= self._top_k:
                break

        return selected

    @staticmethod
    def _build_query_text(ticket: Ticket) -> str:
        """
        Combine subject and issue into the retrieval query.

        The subject is useful because support-ticket subjects often
        contain the feature name. The issue provides additional intent
        and context.
        """

        subject = (ticket.subject or "").strip()
        issue = (ticket.issue or "").strip()

        if subject and issue:
            return f"{subject}. {issue}"

        return subject or issue

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        """
        Normalize text into meaningful lexical tokens.

        Stopwords are removed because generic words such as 'how',
        'the', 'do', and 'I' do not distinguish support articles.
        """

        if not text:
            return set()

        words = re.findall(
            r"[a-zA-Z0-9]+",
            text.lower(),
        )

        return {
            word
            for word in words
            if len(word) > 1 and word not in cls.STOPWORDS
        }

    @classmethod
    def _expand_concepts(cls, text: str) -> set[str]:
        """
        Expand user language into support-domain concepts.

        Example:

            "How long do tests stay active?"

        becomes conceptually:

            {
                "test",
                "active",
                "expiration",
                "duration"
            }

        This helps bridge user wording such as "stay active" with
        documentation terminology such as "expiration time".
        """

        normalized = re.sub(
            r"\s+",
            " ",
            text.lower().strip(),
        )

        concepts: set[str] = set()

        # First detect multi-word phrases.
        for phrase, phrase_concepts in cls.PHRASE_CONCEPTS.items():
            if phrase in normalized:
                concepts.update(phrase_concepts)

        # Then detect individual concept groups.
        tokens = cls._tokenize(text)

        for concept_name, concept_terms in cls.CONCEPT_GROUPS.items():
            if tokens & concept_terms:
                concepts.add(concept_name)

        return concepts

    @classmethod
    def _text_concepts(cls, text: str) -> set[str]:
        """
        Determine which support concepts are represented by document text.
        """

        return cls._expand_concepts(text)

    @classmethod
    def _lexical_score(
        cls,
        query_tokens: set[str],
        chunk: RetrievedChunk,
    ) -> float:
        """
        Calculate lexical relevance between query and document.

        The score considers:

        - title overlap
        - category overlap
        - body/content overlap

        Title and category matches receive more weight because they
        usually represent the primary subject of the document.
        """

        if not query_tokens:
            return 0.0

        title_tokens = cls._tokenize(chunk.title)
        category_tokens = cls._tokenize(chunk.category)
        text_tokens = cls._tokenize(chunk.text)

        title_overlap = query_tokens & title_tokens
        category_overlap = query_tokens & category_tokens
        text_overlap = query_tokens & text_tokens

        weighted_overlap = (
            2.5 * len(title_overlap)
            + 1.5 * len(category_overlap)
            + 1.0 * len(text_overlap)
        )

        max_score = 2.5 * len(query_tokens)

        if max_score <= 0:
            return 0.0

        return min(
            weighted_overlap / max_score,
            1.0,
        )

    @classmethod
    def _title_score(
        cls,
        query_tokens: set[str],
        title: str,
    ) -> float:
        """
        Calculate title relevance.

        Title relevance receives its own signal because support article
        titles often identify the exact feature being asked about.
        """

        if not query_tokens or not title:
            return 0.0

        title_tokens = cls._tokenize(title)

        if not title_tokens:
            return 0.0

        overlap = query_tokens & title_tokens

        token_score = len(overlap) / len(query_tokens)

        # Detect useful phrase matches in titles.
        normalized_title = re.sub(
            r"\s+",
            " ",
            title.lower().strip(),
        )

        phrase_bonus = 0.0

        for phrase in cls.PHRASE_CONCEPTS:
            if phrase in normalized_title:
                phrase_bonus = max(phrase_bonus, 0.25)

        return min(
            token_score + phrase_bonus,
            1.0,
        )

    @classmethod
    def _concept_score(
        cls,
        query_concepts: set[str],
        chunk: RetrievedChunk,
    ) -> float:
        """
        Calculate conceptual relevance between query and document.

        Concepts are extracted from both the query and document title,
        category, and body.

        This catches semantic-equivalent support terminology such as:

            user: "stay active"
            docs: "expiration"

        and:

            user: "delete account"
            docs: "account deletion"
        """

        if not query_concepts:
            return 0.0

        title_concepts = cls._text_concepts(chunk.title)
        category_concepts = cls._text_concepts(chunk.category)
        text_concepts = cls._text_concepts(chunk.text)

        # Title concepts are strongest, followed by category and body.
        matched_weight = 0.0

        for concept in query_concepts:
            if concept in title_concepts:
                matched_weight += 3.0
            elif concept in category_concepts:
                matched_weight += 2.0
            elif concept in text_concepts:
                matched_weight += 1.0

        max_weight = 3.0 * len(query_concepts)

        if max_weight <= 0:
            return 0.0

        return min(
            matched_weight / max_weight,
            1.0,
        )

    @staticmethod
    def _safe_float(value: object) -> float:
        """Convert a similarity value into a safe float."""

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0