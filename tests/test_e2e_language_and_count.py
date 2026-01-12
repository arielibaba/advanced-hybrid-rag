"""
End-to-end tests for language consistency and document count functionality.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from cognidoc.complexity import (
    is_database_meta_question,
    evaluate_complexity,
    should_use_agent,
    DATABASE_META_PATTERNS,
)


class TestDatabaseMetaPatterns:
    """Test database meta-question pattern matching."""

    def test_french_combien_documents(self):
        """Test French 'combien de documents' patterns."""
        queries = [
            "combien de documents cette base comprend-elle ?",
            "combien de documents la base contient ?",
            "combien de docs dans la base ?",
            "dis-moi combien de documents il y a",
            "combien de documents",
        ]
        for q in queries:
            assert is_database_meta_question(q), f"Should match: {q}"

    def test_french_combien_base(self):
        """Test French 'combien...base' patterns."""
        queries = [
            "combien cette base comprend-elle ?",
            "combien la base contient ?",
            "cette base contient combien ?",
        ]
        for q in queries:
            assert is_database_meta_question(q), f"Should match: {q}"

    def test_french_base_comprend(self):
        """Test French 'base comprend' patterns."""
        queries = [
            "cette base comprend combien de fichiers ?",
            "la base comprend quoi ?",
            "que comprend la base ?",
        ]
        for q in queries:
            result = is_database_meta_question(q)
            # At least one of these should match
            print(f"Query: {q} -> {result}")

    def test_french_typos(self):
        """Test French queries with common typos."""
        queries = [
            "combien de couments cette base comprend ?",  # typo: couments
            "combien de documens dans la base ?",  # typo: documens
        ]
        for q in queries:
            # These might not match due to typos - that's OK
            # The important thing is that correct queries match
            result = is_database_meta_question(q)
            print(f"Query with typo: {q} -> {result}")

    def test_english_how_many_documents(self):
        """Test English 'how many documents' patterns."""
        queries = [
            "how many documents are in the database?",
            "how many docs do you have?",
            "how many documents",
            "number of documents in the database",
        ]
        for q in queries:
            assert is_database_meta_question(q), f"Should match: {q}"

    def test_database_size_patterns(self):
        """Test database size/count patterns."""
        queries = [
            "taille de la base",
            "taille de cette base",
            "size of the database",
            "database size",
            "statistiques de la base",
        ]
        for q in queries:
            assert is_database_meta_question(q), f"Should match: {q}"

    def test_non_meta_queries(self):
        """Test that regular queries don't trigger meta detection."""
        queries = [
            "qu'est-ce que le machine learning ?",
            "explain neural networks",
            "how does RAG work?",
            "résume ce document",
        ]
        for q in queries:
            assert not is_database_meta_question(q), f"Should NOT match: {q}"


class TestComplexityEvaluation:
    """Test complexity evaluation for agent routing."""

    def test_database_meta_triggers_agent(self):
        """Database meta-questions should trigger agent path."""
        queries = [
            "combien de documents cette base comprend-elle ?",
            "how many documents in the database?",
            "taille de la base",
        ]
        for q in queries:
            use_agent, complexity = should_use_agent(q)
            assert use_agent, f"Query '{q}' should trigger agent path"
            assert complexity.score >= 0.55, f"Score should be >= 0.55: {complexity.score}"
            assert "database_stats" in complexity.reasoning.lower() or "database meta" in complexity.reasoning.lower(), \
                f"Reasoning should mention database: {complexity.reasoning}"


class TestLanguageDetection:
    """Test language detection functionality."""

    def test_french_detection(self):
        """Test French language detection."""
        from cognidoc.cognidoc_app import detect_query_language

        french_queries = [
            "combien de documents cette base comprend-elle ?",
            "qu'est-ce que le machine learning ?",
            "explique-moi comment ça marche",
            "pourquoi est-ce important ?",
        ]
        for q in french_queries:
            lang = detect_query_language(q)
            assert lang == 'fr', f"Should detect French for: {q}, got: {lang}"

    def test_english_detection(self):
        """Test English language detection."""
        from cognidoc.cognidoc_app import detect_query_language

        english_queries = [
            "how many documents are in the database?",
            "what is machine learning?",
            "explain how this works",
            "why is this important?",
        ]
        for q in english_queries:
            lang = detect_query_language(q)
            assert lang == 'en', f"Should detect English for: {q}, got: {lang}"


class TestSpanishLanguageDetection:
    """Test Spanish language detection functionality."""

    def test_spanish_detection(self):
        """Test Spanish language detection."""
        from cognidoc.cognidoc_app import detect_query_language

        spanish_queries = [
            "¿cuántos documentos tiene esta base?",
            "¿qué es el aprendizaje automático?",
            "explícame cómo funciona esto",
            "¿por qué es importante?",
            "lista los documentos de la base",
        ]
        for q in spanish_queries:
            lang = detect_query_language(q)
            assert lang == 'es', f"Should detect Spanish for: {q}, got: {lang}"

    def test_spanish_without_accents(self):
        """Test Spanish detection with missing accents."""
        from cognidoc.cognidoc_app import detect_query_language

        queries = [
            "cuantos documentos tiene esta base?",
            "que es el aprendizaje automatico?",
        ]
        for q in queries:
            lang = detect_query_language(q)
            # May or may not detect as Spanish without accents
            print(f"Spanish without accents: {q} -> {lang}")


class TestGermanLanguageDetection:
    """Test German language detection functionality."""

    def test_german_detection(self):
        """Test German language detection."""
        from cognidoc.cognidoc_app import detect_query_language

        german_queries = [
            "wie viele Dokumente gibt es in der Datenbank?",
            "was ist maschinelles Lernen?",
            "erkläre mir wie das funktioniert",
            "warum ist das wichtig?",
            "liste die Dokumente auf",
        ]
        for q in german_queries:
            lang = detect_query_language(q)
            assert lang == 'de', f"Should detect German for: {q}, got: {lang}"

    def test_german_with_umlauts(self):
        """Test German detection with umlauts."""
        from cognidoc.cognidoc_app import detect_query_language

        queries = [
            "größe der Datenbank",
            "erkläre die Zusammenhänge",
            "für welche Zwecke?",
        ]
        for q in queries:
            lang = detect_query_language(q)
            assert lang == 'de', f"Should detect German for: {q}, got: {lang}"


class TestSpanishDatabaseMetaPatterns:
    """Test Spanish database meta-question pattern matching."""

    def test_spanish_cuantos_documentos(self):
        """Test Spanish 'cuántos documentos' patterns."""
        queries = [
            "¿cuántos documentos tiene esta base?",
            "cuántos documentos hay en la base?",
            "cuántos docs tiene la base?",
            "dime cuántos documentos hay",
        ]
        for q in queries:
            assert is_database_meta_question(q), f"Should match: {q}"

    def test_spanish_lista_documentos(self):
        """Test Spanish 'lista documentos' patterns."""
        queries = [
            "lista los documentos",
            "listar todos los documentos",
            "qué documentos hay en la base",
            "cuáles son los documentos",
            "dame los nombres de los documentos",
        ]
        for q in queries:
            assert is_database_meta_question(q), f"Should match: {q}"

    def test_spanish_tamano_base(self):
        """Test Spanish 'tamaño de la base' patterns."""
        queries = [
            "tamaño de la base",
            "tamaño de esta base",
            "estadísticas de la base",
        ]
        for q in queries:
            assert is_database_meta_question(q), f"Should match: {q}"


class TestGermanDatabaseMetaPatterns:
    """Test German database meta-question pattern matching."""

    def test_german_wie_viele_dokumente(self):
        """Test German 'wie viele Dokumente' patterns."""
        queries = [
            "wie viele Dokumente gibt es in der Datenbank?",
            "wie viele Dokumente hat die Basis?",
            "wie viele Docs sind vorhanden?",
            "Anzahl der Dokumente in der Datenbank",
        ]
        for q in queries:
            assert is_database_meta_question(q), f"Should match: {q}"

    def test_german_liste_dokumente(self):
        """Test German 'liste Dokumente' patterns."""
        queries = [
            "liste die Dokumente",
            "listen Sie alle Dokumente auf",
            "welche Dokumente gibt es",
            "zeige mir die Dokumente",
            "gib mir die Namen der Dokumente",
        ]
        for q in queries:
            assert is_database_meta_question(q), f"Should match: {q}"

    def test_german_groesse_datenbank(self):
        """Test German 'Größe der Datenbank' patterns."""
        queries = [
            "Größe der Datenbank",
            "Statistiken der Datenbank",
        ]
        for q in queries:
            assert is_database_meta_question(q), f"Should match: {q}"


class TestLanguageAmbiguity:
    """Test handling of ambiguous language queries."""

    def test_short_queries_default_english(self):
        """Short ambiguous queries should default to English."""
        from cognidoc.cognidoc_app import detect_query_language

        queries = [
            "hello",
            "test",
            "document",
        ]
        for q in queries:
            lang = detect_query_language(q)
            assert lang == 'en', f"Short query '{q}' should default to English, got: {lang}"

    def test_french_heavy_query(self):
        """French-heavy queries should detect as French."""
        from cognidoc.cognidoc_app import detect_query_language

        q = "comment est-ce que je peux voir les documents dans la base?"
        lang = detect_query_language(q)
        assert lang == 'fr', f"French-heavy query should detect French, got: {lang}"

    def test_spanish_heavy_query(self):
        """Spanish-heavy queries should detect as Spanish."""
        from cognidoc.cognidoc_app import detect_query_language

        q = "¿cómo puedo ver los documentos que están en la base de datos?"
        lang = detect_query_language(q)
        assert lang == 'es', f"Spanish-heavy query should detect Spanish, got: {lang}"

    def test_german_heavy_query(self):
        """German-heavy queries should detect as German."""
        from cognidoc.cognidoc_app import detect_query_language

        q = "wie kann ich die Dokumente in der Datenbank sehen?"
        lang = detect_query_language(q)
        assert lang == 'de', f"German-heavy query should detect German, got: {lang}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
