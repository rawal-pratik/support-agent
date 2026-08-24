"""Intent-aware hybrid retrieval for support tickets."""

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
    Intent-aware hybrid semantic + lexical retriever.

    Retrieval pipeline:

        Ticket
          ↓
        Query normalization
          ↓
        Query intent / concept extraction
          ↓
        Semantic candidate retrieval
          ↓
        Hybrid scoring
          ↓
        Intent/object matching
          ↓
        Exact phrase/title matching
          ↓
        Source/article deduplication
          ↓
        Final top-K results

    The retriever intentionally uses ONE embedding request per ticket.
    Query expansion is performed locally so retrieval improvements do
    not multiply embedding API usage.
    """

    DEFAULT_CANDIDATE_K = 30

    # Semantic similarity remains the strongest signal, but the other
    # signals are now more meaningful than in the original implementation.
    SEMANTIC_WEIGHT = 0.55
    LEXICAL_WEIGHT = 0.10
    TITLE_WEIGHT = 0.12
    CONCEPT_WEIGHT = 0.08
    PHRASE_WEIGHT = 0.07
    INTENT_WEIGHT = 0.08

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
        "getting",
        "give",
        "given",
        "has",
        "have",
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

    # ------------------------------------------------------------------
    # Support-domain concept groups
    # ------------------------------------------------------------------

    CONCEPT_GROUPS = {
        "expiration": {
            "expiration",
            "expire",
            "expires",
            "expired",
            "expiry",
            "expiring",
            "expiration_date",
            "expiration_time",
            "expiry_date",
            "expiry_time",
            "expiration_period",
        },
        "active": {
            "active",
            "inactive",
            "activation",
            "deactivate",
            "deactivated",
            "remain",
            "remains",
            "stays",
            "stay",
            "available",
            "availability",
            "valid",
            "validity",
            "lifespan",
            "duration",
            "enabled",
            "disabled",
        },
        "test": {
            "test",
            "tests",
            "assessment",
            "assessments",
        },
        "invitation": {
            "invite",
            "invites",
            "invitation",
            "invitations",
            "invited",
        },
        "variant": {
            "variant",
            "variants",
        },
        "account": {
            "account",
            "accounts",
            "profile",
            "user",
            "users",
        },
        "community": {
            "community",
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
        "google": {
            "google",
            "workspace",
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
            "minutes",
            "percentage",
            "percent",
        },
        "reinvite": {
            "reinvite",
            "reinvitation",
            "reinvite",
            "resend",
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
        "subscription": {
            "subscription",
            "subscriptions",
            "plan",
            "plans",
            "cancel",
            "pause",
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
    }

    # ------------------------------------------------------------------
    # Phrase → concepts
    # ------------------------------------------------------------------

    PHRASE_CONCEPTS = {
        "stay active": {"expiration", "active", "test"},
        "remain active": {"expiration", "active", "test"},
        "how long": {"expiration", "duration"},
        "active in the system": {"expiration", "active", "test"},
        "test active": {"expiration", "active", "test"},
        "test expiration": {"expiration", "test"},
        "test expiry": {"expiration", "test"},
        "expiration time": {"expiration", "duration"},
        "expiry time": {"expiration", "duration"},
        "test lifespan": {"expiration", "active", "test"},
        "test duration": {"expiration", "duration", "test"},
        "invitation expiry": {"expiration", "invitation"},
        "invite expiry": {"expiration", "invitation"},
        "test invitation": {"invitation", "test"},
        "delete account": {"delete", "account"},
        "deleting account": {"delete", "account"},
        "remove account": {"delete", "account"},
        "close account": {"delete", "account"},
        "community account": {"account", "community"},
        "google login": {"login", "google", "account"},
        "google workspace": {"login", "google", "account"},
        "test variant": {"test", "variant"},
        "test variants": {"test", "variant"},
        "time accommodation": {"time_accommodation"},
        "extra time": {"time_accommodation"},
        "more time": {"time_accommodation"},
        "additional time": {"time_accommodation"},
        "reinvite candidate": {"reinvite", "candidate"},
        "reinvite the candidate": {"reinvite", "candidate"},
        "send new invitation": {"reinvite", "invite"},
    }

    # ------------------------------------------------------------------
    # Object / intent relationships
    #
    # These are deliberately asymmetric. For example, "test expiration"
    # is a much stronger match for a question about how long a test stays
    # active than merely seeing "expiration" somewhere in a document.
    # ------------------------------------------------------------------

    INTENT_PATTERNS = {
        "test_expiration": {
            "phrases": {
                "how long do tests stay active",
                "how long does a test stay active",
                "how long do tests remain active",
                "how long does the test remain active",
                "test active in the system",
                "test expiration",
                "test expiry",
                "test expiration time",
                "test lifespan",
            },
            "concepts": {
                "test",
                "expiration",
                "active",
            },
            "preferred_title_terms": {
                "test",
                "expiration",
                "expire",
                "expiry",
            },
            "avoid_title_terms": {
                "invite",
                "invitation",
            },
        },
        "invitation_expiration": {
            "phrases": {
                "invitation expiry",
                "invitation expiration",
                "invite expiry",
                "invite expiration",
                "how long invitations remain valid",
            },
            "concepts": {
                "invitation",
                "expiration",
            },
            "preferred_title_terms": {
                "invite",
                "invitation",
                "expiry",
                "expiration",
            },
        },
        "test_variant": {
            "phrases": {
                "test variant",
                "test variants",
                "new test versus variant",
                "new test vs variant",
                "test vs variant",
                "difference between test and variant",
                "difference between creating a new test and creating a test variant",
            },
            "concepts": {
                "test",
                "variant",
            },
            "preferred_title_terms": {
                "test",
                "variant",
            },
        },
        "account_deletion": {
            "phrases": {
                "delete account",
                "deleting account",
                "delete my account",
                "remove account",
                "close account",
                "community account",
            },
            "concepts": {
                "account",
                "delete",
            },
            "preferred_title_terms": {
                "account",
                "delete",
                "deletion",
            },
        },
        "community_account": {
            "phrases": {
                "community account",
                "hackerrank community account",
                "personal hackerrank account",
            },
            "concepts": {
                "account",
                "community",
            },
            "preferred_title_terms": {
                "account",
                "community",
            },
        },
        "google_account": {
            "phrases": {
                "google login",
                "google account",
                "google workspace",
                "created using google",
                "created with google",
            },
            "concepts": {
                "account",
                "login",
                "google",
            },
            "preferred_title_terms": {
                "google",
                "login",
                "account",
            },
        },
        "time_accommodation": {
            "phrases": {
                "extra time",
                "additional time",
                "time accommodation",
                "50% extra time",
                "give candidate more time",
                "extend test time",
            },
            "concepts": {
                "candidate",
                "time_accommodation",
            },
            "preferred_title_terms": {
                "time",
                "accommodation",
                "candidate",
            },
        },
        "candidate_reinvite": {
            "phrases": {
                "reinvite candidate",
                "reinvite the candidate",
                "send new invitation",
                "invite candidate again",
            },
            "concepts": {
                "candidate",
                "invite",
                "reinvite",
            },
            "preferred_title_terms": {
                "invite",
                "candidate",
                "reinvite",
            },
        },
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

    # ==================================================================
    # Main retrieval
    # ==================================================================

    def retrieve(self, ticket: Ticket) -> list[RetrievedChunk]:
        """
        Retrieve the strongest documentation for a support ticket.

        The semantic database is queried once. Local query expansion and
        intent detection are then used to rerank the semantic candidates.

        This deliberately avoids generating multiple embeddings for the
        same ticket, which would unnecessarily increase embedding API usage.
        """

        ticket_text = self._build_query_text(ticket)

        if not ticket_text:
            return []

        query_embedding = self._embedding_provider.embed_query(
            ticket_text
        )

        if not query_embedding:
            return []

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

        query_tokens = self._tokenize(ticket_text)
        query_concepts = self._expand_concepts(ticket_text)
        query_phrases = self._matched_phrases(ticket_text)
        query_intents = self._detect_intents(ticket_text)

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

            phrase_score = self._phrase_score(
                query_phrases=query_phrases,
                chunk=chunk,
            )

            intent_score = self._intent_score(
                query_intents=query_intents,
                chunk=chunk,
            )

            final_score = (
                self.SEMANTIC_WEIGHT * chunk.similarity
                + self.LEXICAL_WEIGHT * lexical_score
                + self.TITLE_WEIGHT * title_score
                + self.CONCEPT_WEIGHT * concept_score
                + self.PHRASE_WEIGHT * phrase_score
                + self.INTENT_WEIGHT * intent_score
            )

            scored_candidates.append(
                (final_score, chunk)
            )

        scored_candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # --------------------------------------------------------------
        # Article/source deduplication.
        #
        # A source may contain several chunks. Keep only the strongest
        # chunk from each source so the LLM gets diverse evidence.
        # --------------------------------------------------------------

        selected: list[RetrievedChunk] = []
        seen_sources: set[str] = set()

        for _, chunk in scored_candidates:
            source_key = self._source_key(chunk.source_path)

            if source_key in seen_sources:
                continue

            seen_sources.add(source_key)
            selected.append(chunk)

            if len(selected) >= self._top_k:
                break

        return selected

    # ==================================================================
    # Query construction
    # ==================================================================

    @staticmethod
    def _build_query_text(ticket: Ticket) -> str:
        """
        Combine subject and issue into the retrieval query.

        The issue receives slightly more importance conceptually because
        it normally contains the actual support intent.
        """

        subject = (ticket.subject or "").strip()
        issue = (ticket.issue or "").strip()

        if subject and issue:
            return f"{subject}. {issue}"

        return subject or issue

    @classmethod
    def _tokenize(cls, text: str) -> set[str]:
        """Normalize text into meaningful lexical tokens."""

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

    # ==================================================================
    # Concept expansion
    # ==================================================================

    @classmethod
    def _expand_concepts(cls, text: str) -> set[str]:
        """Expand user language into support-domain concepts."""

        normalized = re.sub(
            r"\s+",
            " ",
            text.lower().strip(),
        )

        concepts: set[str] = set()

        for phrase, phrase_concepts in cls.PHRASE_CONCEPTS.items():
            if phrase in normalized:
                concepts.update(phrase_concepts)

        tokens = cls._tokenize(text)

        for concept_name, concept_terms in cls.CONCEPT_GROUPS.items():
            if tokens & concept_terms:
                concepts.add(concept_name)

        return concepts

    @classmethod
    def _text_concepts(cls, text: str) -> set[str]:
        """Determine which support concepts are represented in text."""

        return cls._expand_concepts(text)

    @classmethod
    def _matched_phrases(cls, text: str) -> set[str]:
        """Return support phrases explicitly present in the query."""

        normalized = re.sub(
            r"\s+",
            " ",
            text.lower().strip(),
        )

        return {
            phrase
            for phrase in cls.PHRASE_CONCEPTS
            if phrase in normalized
        }

    # ==================================================================
    # Intent detection
    # ==================================================================

    @classmethod
    def _detect_intents(cls, text: str) -> set[str]:
        """
        Detect high-level support intents.

        Multiple intents may be active for a ticket.

        Example:

            "Give the candidate 50% extra time and reinvite them"

        produces:

            time_accommodation
            candidate_reinvite
        """

        normalized = re.sub(
            r"\s+",
            " ",
            text.lower().strip(),
        )

        tokens = cls._tokenize(text)
        concepts = cls._expand_concepts(text)

        intents: set[str] = set()

        for intent_name, definition in cls.INTENT_PATTERNS.items():
            phrases = definition.get("phrases", set())
            required_concepts = definition.get("concepts", set())

            phrase_match = any(
                phrase in normalized
                for phrase in phrases
            )

            concept_overlap = concepts & required_concepts

            # A phrase match is strong enough by itself.
            if phrase_match:
                intents.add(intent_name)
                continue

            # Otherwise require most of the intent's concepts.
            if required_concepts:
                coverage = len(concept_overlap) / len(
                    required_concepts
                )

                if coverage >= 0.67:
                    intents.add(intent_name)

        # Extra handling for combinations where individual tokens are
        # important but no exact phrase exists.
        if (
            "delete" in concepts
            and "account" in concepts
        ):
            intents.add("account_deletion")

        if (
            "community" in concepts
            and "account" in concepts
        ):
            intents.add("community_account")

        if (
            "google" in concepts
            and "login" in concepts
        ):
            intents.add("google_account")

        if (
            "time_accommodation" in concepts
            and "candidate" in concepts
        ):
            intents.add("time_accommodation")

        if (
            "reinvite" in concepts
            and (
                "candidate" in concepts
                or "invite" in concepts
            )
        ):
            intents.add("candidate_reinvite")

        return intents

    # ==================================================================
    # Lexical scoring
    # ==================================================================

    @classmethod
    def _lexical_score(
        cls,
        query_tokens: set[str],
        chunk: RetrievedChunk,
    ) -> float:
        """
        Calculate lexical relevance.

        Title and category matches are weighted more strongly than
        ordinary body matches.
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
            3.0 * len(title_overlap)
            + 1.5 * len(category_overlap)
            + 1.0 * len(text_overlap)
        )

        max_score = 3.0 * len(query_tokens)

        if max_score <= 0:
            return 0.0

        return min(
            weighted_overlap / max_score,
            1.0,
        )

    # ==================================================================
    # Title scoring
    # ==================================================================

    @classmethod
    def _title_score(
        cls,
        query_tokens: set[str],
        title: str,
    ) -> float:
        """
        Calculate title relevance.

        Exact meaningful title terms receive stronger influence than
        generic body-word overlap.
        """

        if not query_tokens or not title:
            return 0.0

        title_tokens = cls._tokenize(title)

        if not title_tokens:
            return 0.0

        overlap = query_tokens & title_tokens

        token_score = len(overlap) / len(query_tokens)

        normalized_title = re.sub(
            r"\s+",
            " ",
            title.lower().strip(),
        )

        phrase_bonus = 0.0

        for phrase in cls.PHRASE_CONCEPTS:
            if phrase in normalized_title:
                phrase_bonus = max(
                    phrase_bonus,
                    0.30,
                )

        return min(
            token_score + phrase_bonus,
            1.0,
        )

    # ==================================================================
    # Concept scoring
    # ==================================================================

    @classmethod
    def _concept_score(
        cls,
        query_concepts: set[str],
        chunk: RetrievedChunk,
    ) -> float:
        """
        Calculate conceptual relevance.

        Title concepts > category concepts > body concepts.
        """

        if not query_concepts:
            return 0.0

        title_concepts = cls._text_concepts(chunk.title)
        category_concepts = cls._text_concepts(chunk.category)
        text_concepts = cls._text_concepts(chunk.text)

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

    # ==================================================================
    # Phrase scoring
    # ==================================================================

    @classmethod
    def _phrase_score(
        cls,
        query_phrases: set[str],
        chunk: RetrievedChunk,
    ) -> float:
        """
        Score exact support phrases.

        Exact phrases are particularly valuable for support documentation
        because terminology often maps directly to article titles.
        """

        if not query_phrases:
            return 0.0

        title = re.sub(
            r"\s+",
            " ",
            chunk.title.lower().strip(),
        )

        category = re.sub(
            r"\s+",
            " ",
            chunk.category.lower().strip(),
        )

        text = re.sub(
            r"\s+",
            " ",
            chunk.text.lower().strip(),
        )

        score = 0.0

        for phrase in query_phrases:
            if phrase in title:
                score += 1.0
            elif phrase in category:
                score += 0.60
            elif phrase in text:
                score += 0.30

        return min(
            score / len(query_phrases),
            1.0,
        )

    # ==================================================================
    # Intent scoring
    # ==================================================================

    @classmethod
    def _intent_score(
        cls,
        query_intents: set[str],
        chunk: RetrievedChunk,
    ) -> float:
        """
        Score whether a document actually matches the user's intent.

        This is the most important distinction from generic semantic
        similarity.

        For example:

            User: "How long do tests stay active?"

        should strongly prefer:

            "Modify Test Expiration Time"

        over:

            "Invite Candidates to a Test"

        even though both documents contain expiration/invitation concepts.
        """

        if not query_intents:
            return 0.0

        normalized_title = re.sub(
            r"\s+",
            " ",
            chunk.title.lower().strip(),
        )

        normalized_category = re.sub(
            r"\s+",
            " ",
            chunk.category.lower().strip(),
        )

        document_text = (
            f"{normalized_title} "
            f"{normalized_category} "
            f"{chunk.text.lower()}"
        )

        total_score = 0.0
        matched_intents = 0

        for intent_name in query_intents:
            definition = cls.INTENT_PATTERNS.get(intent_name)

            if not definition:
                continue

            preferred_terms = definition.get(
                "preferred_title_terms",
                set(),
            )

            avoid_terms = definition.get(
                "avoid_title_terms",
                set(),
            )

            required_concepts = definition.get(
                "concepts",
                set(),
            )

            title_tokens = cls._tokenize(chunk.title)
            category_tokens = cls._tokenize(chunk.category)
            document_concepts = cls._text_concepts(
                document_text
            )

            intent_score = 0.0

            # ----------------------------------------------------------
            # Preferred title terms
            # ----------------------------------------------------------

            preferred_overlap = (
                preferred_terms & title_tokens
            )

            if preferred_terms:
                preferred_ratio = (
                    len(preferred_overlap)
                    / len(preferred_terms)
                )

                intent_score += 0.50 * preferred_ratio

            # ----------------------------------------------------------
            # Concept coverage
            # ----------------------------------------------------------

            if required_concepts:
                concept_overlap = (
                    required_concepts
                    & document_concepts
                )

                concept_ratio = (
                    len(concept_overlap)
                    / len(required_concepts)
                )

                intent_score += 0.35 * concept_ratio

            # ----------------------------------------------------------
            # Explicit phrase in document
            # ----------------------------------------------------------

            for phrase in definition.get("phrases", set()):
                if phrase in document_text:
                    intent_score += 0.20
                    break

            # ----------------------------------------------------------
            # Penalize competing object in the title.
            #
            # This is particularly useful for:
            #
            #   test expiration
            #
            # versus:
            #
            #   invitation expiration
            # ----------------------------------------------------------

            if avoid_terms:
                if avoid_terms & title_tokens:
                    intent_score -= 0.30

            intent_score = max(
                0.0,
                min(intent_score, 1.0),
            )

            if intent_score > 0:
                matched_intents += 1
                total_score += intent_score

        if matched_intents == 0:
            return 0.0

        return min(
            total_score / len(query_intents),
            1.0,
        )

    # ==================================================================
    # Source normalization
    # ==================================================================

    @staticmethod
    def _source_key(source_path: str) -> str:
        """
        Normalize source paths for article-level deduplication.

        Multiple chunks from one article should not consume multiple
        final retrieval slots.
        """

        if not source_path:
            return ""

        return source_path.strip().lower()

    # ==================================================================
    # Utility
    # ==================================================================

    @staticmethod
    def _safe_float(value: object) -> float:
        """Convert a similarity value into a safe float."""

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0