# Session CogniDoc - 9 janvier 2026

## Résumé

Corrections majeures pour le routage agent, la détection de langue, les questions méta sur la base de données, et la **mémoire conversationnelle du chatbot**.

## Tâches complétées cette session

| Tâche | Fichier | Description |
|-------|---------|-------------|
| **Fix patterns meta-questions** | `complexity.py` | Patterns plus flexibles pour "combien de documents", typos inclus |
| **Fix language consistency** | `prompts/*.md` | Règles de langue dans tous les prompts (rewrite, final_answer, agent) |
| **DatabaseStatsTool** | `agent_tools.py` | Nouvel outil pour répondre aux méta-questions sur la base |
| **Language detection** | `cognidoc_app.py` | Détection automatique FR/EN avec préfixes de clarification |
| **Tests E2E** | `tests/test_e2e_language_and_count.py` | 10 nouveaux tests pour patterns et langue |
| **Fix Gemini SDK** | `pyproject.toml` | Ajout dépendance `google-genai` dans extras |
| **Fix helpers TypeError** | `helpers.py` | Gestion format multimodal Gradio (list/None) |
| **Fix reranking provider** | `advanced_rag.py` | Utilisation `llm_chat()` au lieu de `ollama.Client()` |
| **Fix agent response empty** | `cognidoc_app.py` | Capture correcte du retour du générateur `run_streaming()` |
| **Fix chatbot memory** | `agent.py`, `cognidoc_app.py`, `helpers.py` | Mémoire conversationnelle fonctionnelle |
| **Fix DatabaseStatsTool list_documents** | `agent_tools.py` | Retourne les noms des documents avec `list_documents=True` |

## Modifications clés

### 1. Patterns DATABASE_META_PATTERNS (`complexity.py`)

Patterns plus robustes pour détecter les questions sur la base :

```python
DATABASE_META_PATTERNS = [
    # French patterns - flexible matching
    r"\bcombien de doc",      # "combien de documents", typos
    r"\bcombien.{0,20}base\b", # "combien...base" avec 20 chars max
    r"\bbase.{0,15}comprend",  # "cette base comprend", "la base comprend-elle"
    r"\bbase.{0,15}contient",  # "la base contient"
    ...
]
```

### 2. DatabaseStatsTool (`agent_tools.py`)

Nouvel outil (9e outil) pour répondre aux questions sur la base :

```python
class DatabaseStatsTool(BaseTool):
    name = ToolName.DATABASE_STATS
    # Retourne: total_documents, total_chunks, graph_nodes, graph_edges
```

### 3. Détection de langue (`cognidoc_app.py`)

```python
def detect_query_language(query: str) -> str:
    """Détecte FR ou EN basé sur indicateurs linguistiques."""
    french_indicators = [" est ", " sont ", " que ", ...]
    ...

def get_clarification_prefix(lang: str) -> str:
    if lang == "fr":
        return "**Clarification requise :**"
    return "**Clarification needed:**"
```

### 4. Règles de langue dans les prompts

Tous les prompts incluent maintenant :

```markdown
## Language Rules
- ALWAYS respond in the SAME LANGUAGE as the user's question.
- If the user asks in French, respond in French.
- If the user asks in English, respond in English.
```

### 5. Mémoire conversationnelle (`cognidoc_app.py`, `agent.py`, `helpers.py`)

La mémoire du chatbot fonctionne maintenant correctement :

```
User: "Combien de documents cette base comprend-elle?"
Bot:  "Cette base de données comprend 5 documents."

User: "cite-les-moi"
Bot:  "Cette base de données comprend les 5 documents suivants: test_document, Rapport Sémantique, ..."
```

**Flux corrigé:**
1. Query rewriter transforme "cite-les-moi" → "Cite-moi les 5 documents que cette base comprend."
2. L'agent reçoit la query réécrite (pas le message brut)
3. DatabaseStatsTool retourne les noms des documents via `list_documents=True`

### 6. DatabaseStatsTool amélioré (`agent_tools.py`)

```python
class DatabaseStatsTool(BaseTool):
    parameters = {
        "list_documents": "Set to true to get the list of document names/titles"
    }

    def execute(self, list_documents: bool = False) -> ToolResult:
        # Utilise get_all_documents() au lieu de .documents
        docs = ki.get_all_documents()
        if list_documents:
            doc_names = [doc.metadata.get('source', {}).get('document') for doc in docs]
            stats["document_names"] = sorted(list(set(doc_names)))
```

## Tests (43+ tests passent)

| Module | Tests |
|--------|-------|
| `test_agent_tools.py` | 33 |
| `test_e2e_language_and_count.py` | 10 |
| **Total validé** | **43+** |

## Commandes CLI

```bash
# Lancer l'app (avec agent activé)
uv run python -m cognidoc.cognidoc_app

# Sans reranking (plus rapide)
uv run python -m cognidoc.cognidoc_app --no-rerank

# Tests
uv run python -m pytest tests/ -v
```

## Configuration

```
LLM:       gemini-2.0-flash (Gemini)
Embedding: qwen3-embedding:0.6b (Ollama)
Agent:     Activé (seuil complexité: 0.55)
DatabaseStatsTool: Activé pour meta-questions
```

## Structure mise à jour

```
src/cognidoc/
├── complexity.py        # DATABASE_META_PATTERNS améliorés
├── agent_tools.py       # 9 outils (NEW: database_stats)
├── agent.py             # Règles de langue dans SYSTEM_PROMPT
├── cognidoc_app.py      # detect_query_language(), get_clarification_prefix()
├── helpers.py           # Fix TypeError format multimodal
└── prompts/
    ├── system_prompt_rewrite_query.md      # Language Preservation rules
    └── system_prompt_generate_final_answer.md # Language Rules

tests/
├── test_agent_tools.py              # 33 tests
└── test_e2e_language_and_count.py   # 10 tests (NEW)
```

## Bugs corrigés

1. **Agent path non déclenché** - Patterns trop restrictifs pour "combien de documents"
2. **Réponses en anglais** - Règles de langue manquantes dans prompts
3. **TypeError helpers.py** - Format multimodal Gradio non géré
4. **Reranking 404** - Utilisait ollama.Client() avec modèle Gemini
5. **Gemini SDK manquant** - google-genai non installé dans venv
6. **Réponse agent vide** - Le générateur `run_streaming()` n'était pas correctement consommé, puis `run()` était appelé une seconde fois inutilement. Fix: capture du retour via `StopIteration.value`
7. **Mémoire chatbot cassée** - "cite-les-moi" après "combien de documents" causait "que voulez-vous citer?"
   - **Cause racine**: `KeyError: '"answer"'` dans `agent.py` dû aux accolades non échappées dans SYSTEM_PROMPT
   - **Fix**: `{"answer": "..."}` → `{{"answer": "..."}}`
8. **Agent utilisant raw query** - L'agent recevait "cite-les-moi" au lieu de la query réécrite avec contexte
   - **Fix**: `agent.run_streaming(candidates[0])` au lieu de `user_message`
9. **parse_rewritten_query incomplet** - Ne gérait que `- ` pas `* ` comme style de bullet
   - **Fix**: Ajout `elif stripped.startswith('* '):`
10. **DatabaseStatsTool sans noms de documents** - Utilisait `.documents` qui n'existe pas
    - **Fix**: Utilisation de `get_all_documents()` + extraction des métadonnées `source.document`

## Améliorations implémentées (session 2)

### 1. Cache des résultats d'outils (`agent_tools.py`)

```python
class ToolCache:
    """TTL-based cache for tool results."""
    TTL_CONFIG = {
        "database_stats": 300,      # 5 minutes
        "retrieve_vector": 120,     # 2 minutes
        "retrieve_graph": 120,
        "lookup_entity": 300,
        "compare_entities": 180,
    }

    @classmethod
    def get(cls, tool_name: str, **kwargs) -> Optional[Any]:
        # Check cache with MD5 hash key
        ...

    @classmethod
    def set(cls, tool_name: str, result: Any, **kwargs) -> None:
        # Store with timestamp
        ...
```

**Avantages:**
- Réduit la latence pour les requêtes répétées
- TTL configurable par outil
- Log cache hit/miss pour debug
- Indicateur `[cached]` dans les résultats

### 2. Streaming granulaire dans l'UI (`cognidoc_app.py`)

```python
state_emoji = {
    AgentState.THINKING: "🤔",
    AgentState.ACTING: "⚡",
    AgentState.OBSERVING: "👁️",
    AgentState.REFLECTING: "💭",
}
progress_lines.append(f"{state_emoji} {message}")
history[-1]["content"] = f"*Processing query...*\n\n{progress_display}"
yield convert_history_to_tuples(history)
```

L'utilisateur voit maintenant en temps réel:
- 🤔 [Step 1/7] Analyzing query...
- 🤔 Thought: I need to search for...
- ⚡ Calling retrieve_vector(query=...)
- 👁️ Result [cached]: Found 5 documents...
- 💭 Analysis: The documents contain...

### 3. Prompts optimisés pour réduire les steps (`agent.py`)

**Avant:** 5-7 steps typiques
**Après:** 2-3 steps pour la plupart des requêtes

```python
SYSTEM_PROMPT = """You are an efficient research assistant. Your goal is to answer questions QUICKLY with MINIMAL steps.

## Efficiency Guidelines - CRITICAL
1. **One retrieval is usually enough.** After ONE successful retrieve_vector or retrieve_graph call, you likely have enough information. Proceed to final_answer.
2. **Skip synthesis for simple questions.** Use final_answer directly after getting relevant documents.
3. **Target: 2-3 steps max for most queries.** Complex comparisons may need 4 steps.
...
"""
```

**Changements clés:**
- SYSTEM_PROMPT plus directif et efficace
- THINK_PROMPT simplifié (encourage action immédiate)
- REFLECT_PROMPT focalisé sur "Can you answer NOW?"
- Instructions claires pour éviter redondances

## Tests (127 tests passent)

| Module | Tests |
|--------|-------|
| `test_agent.py` | 27 |
| `test_agent_tools.py` | 33 |
| `test_complexity.py` | 24 |
| `test_e2e_language_and_count.py` | 10 |
| `test_providers.py` | 33 |
| **Total validé** | **127** |

### 4. Fix langue dans le Fast Path (`user_prompt_generate_final_answer.md`)

Le fast path répondait parfois en anglais (ex: "No relevant details are available").

**Avant:**
```markdown
- If insufficient information is available, respond clearly with:
  **"No relevant details are available."**
```

**Après:**
```markdown
- If insufficient information is available, respond in the user's language:
  - French: **"Je n'ai pas trouvé d'informations pertinentes..."**
  - English: **"I could not find relevant information..."**
- CRITICAL: Deliver your ENTIRE response in the SAME LANGUAGE as the user's question.
```

## Commits de cette session

| Hash | Description |
|------|-------------|
| `a56ecdf` | Improve agent performance: caching, streaming, and optimized prompts |
| `0a05114` | Update SESSION_RESUME.md with performance improvements |
| `c68164f` | Fix language consistency in fast path responses |

## Améliorations futures

1. **Support langues additionnelles** - Espagnol, Allemand, etc.
2. **Cache persistant** - Utiliser Redis ou SQLite pour le cache
3. **Métriques de performance** - Dashboard temps de réponse, cache hits
4. **Tests de charge** - Benchmarks avec multiple requêtes simultanées
