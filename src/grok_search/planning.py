"""Planning models and session management for staged web search workflows."""

import uuid
from typing import Literal, cast

from pydantic import BaseModel, Field


class IntentOutput(BaseModel):
    """
    Represent the structured output for intent analysis.

    Attributes:
        core_question: The distilled user question.
        query_type: The search intent classification.
        time_sensitivity: The expected freshness requirement.
        domain: The detected subject domain when available.
        premise_valid: Whether the user premise appears valid.
        ambiguities: Any unresolved ambiguities.
        unverified_terms: External terms that require verification.
    """

    core_question: str = Field(description="Distilled core question in one sentence")
    query_type: Literal["factual", "comparative", "exploratory", "analytical"] = Field(
        description=(
            "factual=single answer, comparative=A vs B, "
            "exploratory=broad understanding, analytical=deep reasoning"
        )
    )
    time_sensitivity: Literal["realtime", "recent", "historical", "irrelevant"] = Field(
        description=(
            "realtime=today, recent=days/weeks, historical=months+, irrelevant=timeless"
        )
    )
    domain: str | None = Field(
        default=None, description="Specific domain if identifiable"
    )
    premise_valid: bool | None = Field(
        default=None, description="False if the question contains a flawed assumption"
    )
    ambiguities: list[str] | None = Field(
        default=None,
        description="Unresolved ambiguities that may affect search direction",
    )
    unverified_terms: list[str] | None = Field(
        default=None,
        description=(
            "External classifications, rankings, or taxonomies that may be "
            "incomplete or outdated in training data. Examples include "
            "'CCF-A', 'Fortune 500', and 'OWASP Top 10'. Each should become "
            "a prerequisite sub-query in Phase 3."
        ),
    )


class ComplexityOutput(BaseModel):
    """
    Represent the complexity assessment for a search plan.

    Attributes:
        level: The overall planning complexity level.
        estimated_sub_queries: The expected number of sub-queries.
        estimated_tool_calls: The expected number of tool calls.
        justification: The reasoning behind the selected level.
    """

    level: Literal[1, 2, 3] = Field(
        description=(
            "1=simple (1-2 searches), 2=moderate (3-5 searches), "
            "3=complex (6+ searches)"
        )
    )
    estimated_sub_queries: int = Field(ge=1, le=20)
    estimated_tool_calls: int = Field(ge=1, le=50)
    justification: str


class SubQuery(BaseModel):
    """
    Represent one decomposed sub-query in the search plan.

    Attributes:
        id: The unique sub-query identifier.
        goal: The sub-query objective.
        expected_output: The success criteria for the sub-query.
        tool_hint: An optional suggested tool.
        boundary: The explicit scope boundary for the sub-query.
        depends_on: Optional prerequisite sub-query identifiers.
    """

    id: str = Field(description="Unique identifier (e.g., 'sq1')")
    goal: str
    expected_output: str = Field(description="What a successful result looks like")
    tool_hint: str | None = Field(
        default=None, description="Suggested tool: web_search | web_fetch | web_map"
    )
    boundary: str = Field(
        description=(
            "What this sub-query explicitly excludes. It must state mutual "
            "exclusion with sibling sub-queries, not just the broader domain."
        )
    )
    depends_on: list[str] | None = Field(
        default=None, description="IDs of prerequisite sub-queries"
    )


class SearchTerm(BaseModel):
    """
    Represent a search term associated with a single sub-query.

    Attributes:
        term: The search term text.
        purpose: The sub-query identifier served by the term.
        round: The search execution round number.
    """

    term: str = Field(
        description=(
            "Search query string. MUST be <=8 words. Drop redundant synonyms "
            "(for example, use 'RAG' instead of a fully expanded phrase)."
        )
    )
    purpose: str = Field(
        description=(
            "Single sub-query ID this term serves (for example, 'sq2'). "
            "Use one term per sub-query and do not combine IDs."
        )
    )
    round: int = Field(
        ge=1,
        description=(
            "Execution round: 1=broad discovery, 2+=targeted follow-up "
            "refined by round 1 findings"
        ),
    )


class StrategyOutput(BaseModel):
    """
    Represent the planned search strategy for a session.

    Attributes:
        approach: The overall search strategy style.
        search_terms: The search terms to execute.
        fallback_plan: The contingency plan if the primary strategy fails.
    """

    approach: Literal["broad_first", "narrow_first", "targeted"] = Field(
        description=(
            "broad_first=wide then narrow, narrow_first=precise then expand, "
            "targeted=known-item"
        )
    )
    search_terms: list[SearchTerm]
    fallback_plan: str | None = Field(
        default=None, description="Fallback if primary searches fail"
    )


class ToolPlanItem(BaseModel):
    """
    Represent the tool assignment for a sub-query.

    Attributes:
        sub_query_id: The target sub-query identifier.
        tool: The selected tool name.
        reason: The rationale for the mapping.
        params: Optional tool-specific parameters.
    """

    sub_query_id: str
    tool: Literal["web_search", "web_fetch", "web_map"]
    reason: str
    params: dict | None = Field(default=None, description="Tool-specific parameters")


class ExecutionOrderOutput(BaseModel):
    """
    Represent the execution ordering for sub-queries.

    Attributes:
        parallel: Groups of sub-queries that can run in parallel.
        sequential: Sub-queries that must run in order.
        estimated_rounds: The expected number of execution rounds.
    """

    parallel: list[list[str]] = Field(
        description="Groups of sub-query IDs runnable in parallel"
    )
    sequential: list[str] = Field(description="Sub-query IDs that must run in order")
    estimated_rounds: int = Field(ge=1)


PHASE_NAMES = [
    "intent_analysis",
    "complexity_assessment",
    "query_decomposition",
    "search_strategy",
    "tool_selection",
    "execution_order",
]

REQUIRED_PHASES: dict[int, set[str]] = {
    1: {"intent_analysis", "complexity_assessment", "query_decomposition"},
    2: {
        "intent_analysis",
        "complexity_assessment",
        "query_decomposition",
        "search_strategy",
        "tool_selection",
    },
    3: set(PHASE_NAMES),
}

_ACCUMULATIVE_LIST_PHASES = {"query_decomposition", "tool_selection"}
_MERGE_STRATEGY_PHASE = "search_strategy"


def _split_csv(value: str) -> list[str]:
    """
    Split a comma-separated string into trimmed tokens.

    Args:
        value: The comma-separated string.

    Returns:
        A list of non-empty trimmed values.
    """
    return [s.strip() for s in value.split(",") if s.strip()] if value else []


class PhaseRecord(BaseModel):
    """
    Store one recorded planning phase submission.

    Attributes:
        phase: The phase name.
        thought: The reasoning text for the phase.
        data: The structured payload associated with the phase.
        confidence: The reported confidence score.
    """

    phase: str
    thought: str
    data: dict | list | None = None
    confidence: float = 1.0


class PlanningSession:
    """
    Track phase submissions for one planning session.

    Attributes:
        session_id: The unique session identifier.
        phases: Recorded phases keyed by phase name.
        complexity_level: The selected planning complexity level.
    """

    def __init__(self, session_id: str):
        """
        Initialize an empty planning session.

        Args:
            session_id: The unique planning session identifier.

        Returns:
            None.
        """
        self.session_id = session_id
        self.phases: dict[str, PhaseRecord] = {}
        self.complexity_level: int | None = None

    @property
    def completed_phases(self) -> list[str]:
        """
        Return completed phases in canonical order.

        Returns:
            The ordered list of completed phase names.
        """
        return [p for p in PHASE_NAMES if p in self.phases]

    def required_phases(self) -> set[str]:
        """
        Return the phases required for the session complexity level.

        Returns:
            The set of required phase names.
        """
        return REQUIRED_PHASES.get(self.complexity_level or 3, REQUIRED_PHASES[3])

    def is_complete(self) -> bool:
        """
        Check whether all required phases have been recorded.

        Returns:
            True when the session has all required phases.
        """
        if self.complexity_level is None:
            return False
        return self.required_phases().issubset(self.phases.keys())

    def build_executable_plan(self) -> dict[str, dict | list | None]:
        """
        Build the executable plan payload from recorded phases.

        Returns:
            A mapping of phase names to their recorded payloads.
        """
        return {name: record.data for name, record in self.phases.items()}


class PlanningEngine:
    """
    Manage planning sessions and phase submissions.

    Attributes:
        _sessions: Active planning sessions keyed by session ID.
    """

    def __init__(self):
        """
        Initialize the in-memory planning session store.

        Returns:
            None.
        """
        self._sessions: dict[str, PlanningSession] = {}

    def get_session(self, session_id: str) -> PlanningSession | None:
        """
        Fetch an existing planning session by identifier.

        Args:
            session_id: The session identifier to look up.

        Returns:
            The matching planning session, or None when absent.
        """
        return self._sessions.get(session_id)

    def process_phase(
        self,
        phase: str,
        thought: str,
        session_id: str = "",
        is_revision: bool = False,
        revises_phase: str = "",
        confidence: float = 1.0,
        phase_data: dict | list | None = None,
    ) -> dict[str, object]:
        """
        Record a phase submission and return updated session metadata.

        Args:
            phase: The phase name being recorded.
            thought: The reasoning text associated with the phase.
            session_id: The target session identifier, or an empty string for a new one.
            is_revision: Whether this submission replaces an earlier phase entry.
            revises_phase: The explicit phase name to revise when different from phase.
            confidence: The reported confidence score.
            phase_data: The structured payload for the phase.

        Returns:
            A dictionary describing the updated planning session state.
        """
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
        else:
            sid = session_id if session_id else uuid.uuid4().hex[:12]
            session = PlanningSession(sid)
            self._sessions[sid] = session

        target = revises_phase if is_revision and revises_phase else phase
        if target not in PHASE_NAMES:
            return {
                "error": f"Unknown phase: {target}. Valid: {', '.join(PHASE_NAMES)}"
            }

        if target in _ACCUMULATIVE_LIST_PHASES:
            if is_revision:
                session.phases[target] = PhaseRecord(
                    phase=target,
                    thought=thought,
                    data=[phase_data]
                    if not isinstance(phase_data, list)
                    else phase_data,
                    confidence=confidence,
                )
            elif target in session.phases and isinstance(
                session.phases[target].data, list
            ):
                existing_items = cast(list, session.phases[target].data)
                existing_items.append(phase_data)
                session.phases[target].thought = thought
                session.phases[target].confidence = confidence
            else:
                session.phases[target] = PhaseRecord(
                    phase=target,
                    thought=thought,
                    data=[phase_data],
                    confidence=confidence,
                )
        elif target == _MERGE_STRATEGY_PHASE:
            existing = session.phases.get(target)
            if is_revision:
                session.phases[target] = PhaseRecord(
                    phase=target,
                    thought=thought,
                    data=phase_data,
                    confidence=confidence,
                )
            elif (
                existing
                and isinstance(existing.data, dict)
                and isinstance(phase_data, dict)
            ):
                existing.data.setdefault("search_terms", []).extend(
                    phase_data.get("search_terms", [])
                )
                if phase_data.get("approach"):
                    existing.data["approach"] = phase_data["approach"]
                if phase_data.get("fallback_plan"):
                    existing.data["fallback_plan"] = phase_data["fallback_plan"]
                existing.thought = thought
                existing.confidence = confidence
            else:
                session.phases[target] = PhaseRecord(
                    phase=target,
                    thought=thought,
                    data=phase_data,
                    confidence=confidence,
                )
        else:
            session.phases[target] = PhaseRecord(
                phase=target,
                thought=thought,
                data=phase_data,
                confidence=confidence,
            )

        if target == "complexity_assessment" and isinstance(phase_data, dict):
            level = phase_data.get("level")
            if level in (1, 2, 3):
                session.complexity_level = level

        complete = session.is_complete()
        result: dict = {
            "session_id": session.session_id,
            "completed_phases": session.completed_phases,
            "complexity_level": session.complexity_level,
            "plan_complete": complete,
        }

        remaining = [
            p
            for p in PHASE_NAMES
            if p in session.required_phases() and p not in session.phases
        ]
        if remaining:
            result["phases_remaining"] = remaining

        if complete:
            result["executable_plan"] = session.build_executable_plan()

        return result


engine = PlanningEngine()
