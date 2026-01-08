"""
Unit tests for complexity evaluation module.
"""

import pytest

from cognidoc.complexity import (
    ComplexityLevel,
    ComplexityScore,
    evaluate_complexity,
    should_use_agent,
    count_complex_keywords,
    count_subquestions,
    is_ambiguous,
    AGENT_THRESHOLD,
)
from cognidoc.query_orchestrator import (
    RoutingDecision,
    QueryType,
    RetrievalMode,
)


class TestComplexityScore:
    """Tests for ComplexityScore dataclass."""

    def test_simple_score_not_agent(self):
        """Simple complexity should not trigger agent."""
        score = ComplexityScore(
            score=0.2,
            level=ComplexityLevel.SIMPLE,
        )
        assert score.should_use_agent is False

    def test_complex_score_triggers_agent(self):
        """Complex level should trigger agent."""
        score = ComplexityScore(
            score=0.7,
            level=ComplexityLevel.COMPLEX,
        )
        assert score.should_use_agent is True

    def test_ambiguous_triggers_agent(self):
        """Ambiguous level should trigger agent."""
        score = ComplexityScore(
            score=0.6,
            level=ComplexityLevel.AMBIGUOUS,
        )
        assert score.should_use_agent is True

    def test_moderate_not_agent(self):
        """Moderate level should not trigger agent."""
        score = ComplexityScore(
            score=0.4,
            level=ComplexityLevel.MODERATE,
        )
        assert score.should_use_agent is False


class TestCountComplexKeywords:
    """Tests for complex keyword detection."""

    def test_no_keywords(self):
        """Query without complex keywords."""
        count, matches = count_complex_keywords("Quelle est la date?")
        assert count == 0
        assert matches == []

    def test_french_keywords(self):
        """French complex keywords detected."""
        count, matches = count_complex_keywords(
            "Pourquoi et comment analyser les conséquences?"
        )
        assert count >= 3  # pourquoi, analyser, conséquences

    def test_english_keywords(self):
        """English complex keywords detected."""
        count, matches = count_complex_keywords(
            "Why does this cause such an effect? Explain the reason."
        )
        assert count >= 3  # why, cause, effect, explain, reason

    def test_comparative_keywords(self):
        """Comparative keywords detected."""
        count, matches = count_complex_keywords(
            "Compare the advantages and differences between A and B"
        )
        assert count >= 2  # compare, advantage, difference


class TestCountSubquestions:
    """Tests for sub-question counting."""

    def test_no_subquestions(self):
        """Single question returns 1."""
        count = count_subquestions("What is the capital?")
        assert count == 1

    def test_bullet_points(self):
        """Count bullet point sub-questions."""
        rewritten = """
        - What is X?
        - What is Y?
        - How do they relate?
        """
        count = count_subquestions(rewritten)
        assert count == 3

    def test_numbered_list(self):
        """Count numbered sub-questions."""
        rewritten = """
        1. First question
        2. Second question
        3) Third question
        """
        count = count_subquestions(rewritten)
        assert count == 3

    def test_empty_returns_one(self):
        """Empty or None returns 1."""
        assert count_subquestions("") == 1
        assert count_subquestions(None) == 1


class TestIsAmbiguous:
    """Tests for ambiguity detection."""

    def test_short_query_ambiguous(self):
        """Very short queries are ambiguous."""
        assert is_ambiguous("Quoi?") is True
        assert is_ambiguous("Help") is True

    def test_multiple_questions_ambiguous(self):
        """Multiple question marks indicate ambiguity."""
        assert is_ambiguous("Is it A? Or B? Maybe C?") is True

    def test_normal_query_not_ambiguous(self):
        """Normal queries are not ambiguous."""
        assert is_ambiguous("What is the capital of France?") is False


class TestEvaluateComplexity:
    """Tests for main complexity evaluation."""

    def test_simple_factual_query(self):
        """Simple factual query has low complexity."""
        result = evaluate_complexity("Quelle est la date de création?")
        assert result.level == ComplexityLevel.SIMPLE
        assert result.score < AGENT_THRESHOLD
        assert result.should_use_agent is False

    def test_analytical_query_type_triggers_complex(self):
        """ANALYTICAL query type triggers complex level."""
        routing = RoutingDecision(
            query="Analyze the impact",
            query_type=QueryType.ANALYTICAL,
            mode=RetrievalMode.HYBRID,
            confidence=0.8,
            entities_detected=[],
        )
        result = evaluate_complexity("Analyze the impact", routing=routing)
        assert result.factors["query_type"] == 1.0

    def test_comparative_query_type_triggers_complex(self):
        """COMPARATIVE query type triggers complex level."""
        routing = RoutingDecision(
            query="Compare A and B",
            query_type=QueryType.COMPARATIVE,
            mode=RetrievalMode.HYBRID,
            confidence=0.8,
            entities_detected=["A", "B"],
        )
        result = evaluate_complexity("Compare A and B", routing=routing)
        assert result.factors["query_type"] == 1.0

    def test_multi_entity_increases_score(self):
        """Multiple entities increase complexity score."""
        routing = RoutingDecision(
            query="Relation between A, B, C and D",
            query_type=QueryType.RELATIONAL,
            mode=RetrievalMode.HYBRID,
            confidence=0.7,
            entities_detected=["A", "B", "C", "D"],
        )
        result = evaluate_complexity("Relation between A, B, C and D", routing=routing)
        assert result.factors["entity_count"] == 1.0

    def test_many_subquestions_increases_score(self):
        """Many sub-questions increase complexity score."""
        rewritten = """
        - Question 1
        - Question 2
        - Question 3
        - Question 4
        """
        result = evaluate_complexity(
            "Complex multi-part question",
            rewritten_query=rewritten,
        )
        assert result.factors["subquestion_count"] == 1.0

    def test_low_confidence_increases_score(self):
        """Low routing confidence increases complexity."""
        routing = RoutingDecision(
            query="Unclear question",
            query_type=QueryType.UNKNOWN,
            mode=RetrievalMode.HYBRID,
            confidence=0.2,
            entities_detected=[],
        )
        result = evaluate_complexity("Unclear question", routing=routing)
        assert result.factors["low_confidence"] == 1.0

    def test_combined_factors_trigger_agent(self):
        """Combined factors can trigger agent path."""
        routing = RoutingDecision(
            query="Compare Gemini et GPT-4 et explique pourquoi",
            query_type=QueryType.COMPARATIVE,
            mode=RetrievalMode.HYBRID,
            confidence=0.6,
            entities_detected=["Gemini", "GPT-4"],
        )
        rewritten = """
        - Avantages de Gemini
        - Avantages de GPT-4
        - Comparaison
        """
        result = evaluate_complexity(
            "Compare Gemini et GPT-4 et explique pourquoi",
            routing=routing,
            rewritten_query=rewritten,
        )
        assert result.should_use_agent is True
        assert result.level == ComplexityLevel.COMPLEX


class TestShouldUseAgent:
    """Tests for should_use_agent convenience function."""

    def test_returns_tuple(self):
        """Returns tuple of (bool, ComplexityScore)."""
        use_agent, score = should_use_agent("Simple query")
        assert isinstance(use_agent, bool)
        assert isinstance(score, ComplexityScore)

    def test_custom_threshold(self):
        """Custom threshold can be specified."""
        # With very low threshold, even simple queries trigger agent
        use_agent, _ = should_use_agent("Simple query", threshold=0.01)
        assert use_agent is True

        # With very high threshold, nothing triggers agent
        routing = RoutingDecision(
            query="Complex analytical comparison",
            query_type=QueryType.ANALYTICAL,
            mode=RetrievalMode.HYBRID,
            confidence=0.3,
            entities_detected=["A", "B", "C", "D"],
        )
        use_agent, _ = should_use_agent(
            "Complex analytical comparison",
            routing=routing,
            threshold=0.99,
        )
        assert use_agent is False


class TestComplexityLevelEnum:
    """Tests for ComplexityLevel enum values."""

    def test_level_values(self):
        """Verify enum values."""
        assert ComplexityLevel.SIMPLE.value == "simple"
        assert ComplexityLevel.MODERATE.value == "moderate"
        assert ComplexityLevel.COMPLEX.value == "complex"
        assert ComplexityLevel.AMBIGUOUS.value == "ambiguous"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
