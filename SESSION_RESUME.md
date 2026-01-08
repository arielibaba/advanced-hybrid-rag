# Session CogniDoc - 8 janvier 2026

## Résumé

Implémentation d'un système RAG agentique conditionnel avec boucle ReAct.

## Tâches complétées cette session

| Tâche | Fichier | Description |
|-------|---------|-------------|
| Bug NameError 'LLM' | `hybrid_retriever.py:145` | Remplacé `LLM` par `DEFAULT_LLM_MODEL` |
| Tests providers | `tests/test_providers.py` | +10 nouveaux tests (32 au total) |
| LLM par défaut Gemini | `constants.py` | `DEFAULT_LLM_MODEL = gemini-2.0-flash` |
| **Agentic RAG** | 4 nouveaux fichiers | Système agent conditionnel complet |

## Architecture Agentic RAG

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │  Query Classification  │
                 │  + Complexity Score    │
                 └────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │ score < 0.55                  │ score >= 0.55
              ▼                               ▼
    ┌──────────────────┐           ┌──────────────────┐
    │   FAST PATH      │           │   AGENT PATH     │
    │   Standard RAG   │           │   ReAct Loop     │
    │   (~80% queries) │           │   (~20% queries) │
    └──────────────────┘           └──────────────────┘
```

## Nouveaux modules

### 1. `complexity.py` - Évaluation complexité (25 tests)

```python
from cognidoc.complexity import should_use_agent

use_agent, score = should_use_agent(query, routing, rewritten)
# score.score: 0.0-1.0
# score.level: SIMPLE | MODERATE | COMPLEX | AMBIGUOUS
```

Signaux évalués:
- Query type (ANALYTICAL, COMPARATIVE → agent)
- Nombre d'entités (≥3 → complex)
- Sous-questions (≥3 → complex)
- Mots-clés complexes (pourquoi, compare, analyse...)
- Faible confidence routing

### 2. `agent_tools.py` - 8 outils (33 tests)

| Outil | Description |
|-------|-------------|
| `RETRIEVE_VECTOR` | Recherche sémantique |
| `RETRIEVE_GRAPH` | Parcours knowledge graph |
| `LOOKUP_ENTITY` | Info entité spécifique |
| `COMPARE_ENTITIES` | Comparaison structurée |
| `SYNTHESIZE` | Fusion de contextes |
| `VERIFY_CLAIM` | Vérification factuelle |
| `ASK_CLARIFICATION` | Demander précision |
| `FINAL_ANSWER` | Réponse finale |

### 3. `agent.py` - CogniDocAgent ReAct (27 tests)

```python
from cognidoc.agent import create_agent

agent = create_agent(retriever, max_steps=7)
result = agent.run("Compare Gemini et GPT-4")
print(result.answer)  # Réponse multi-étapes
print(result.steps)   # Trace du raisonnement
```

Boucle ReAct:
1. **THINK** - Analyser, décider action
2. **ACT** - Exécuter outil
3. **OBSERVE** - Traiter résultat
4. **REFLECT** - Évaluer si objectif atteint

## Commandes CLI

```bash
# Avec agent (défaut)
python -m cognidoc.cognidoc_app

# Sans agent (fast path uniquement)
python -m cognidoc.cognidoc_app --no-agent

# Sans reranking + sans agent (le plus rapide)
python -m cognidoc.cognidoc_app --no-rerank --no-agent

# Tests
python -m pytest tests/ -v  # 117 tests
```

## Configuration par défaut

```
LLM:       gemini-2.0-flash (Gemini)
Embedding: qwen3-embedding:0.6b (Ollama)
Agent:     Activé (seuil complexité: 0.55)
```

## Tests unitaires (117 tests)

| Module | Tests |
|--------|-------|
| `test_complexity.py` | 25 |
| `test_agent_tools.py` | 33 |
| `test_agent.py` | 27 |
| `test_providers.py` | 32 |
| **Total** | **117** |

## Structure du package

```
src/cognidoc/
├── __init__.py
├── api.py               # Classe CogniDoc principale
├── cli.py               # Interface CLI
├── cognidoc_app.py      # Interface Gradio + intégration agent
├── complexity.py        # (NEW) Évaluation complexité query
├── agent.py             # (NEW) CogniDocAgent ReAct
├── agent_tools.py       # (NEW) 8 outils pour l'agent
├── hybrid_retriever.py  # Retriever hybride
├── constants.py         # DEFAULT_LLM_MODEL = gemini-2.0-flash
└── ...

tests/
├── test_complexity.py   # (NEW) 25 tests
├── test_agent_tools.py  # (NEW) 33 tests
├── test_agent.py        # (NEW) 27 tests
└── test_providers.py    # 32 tests
```

## Commits cette session

```
3606bb8 - Update SESSION_RESUME.md with session summary
5c77ae4 - Fix hybrid retriever bug and add provider tests
```

## Prochaines améliorations possibles

1. **UI feedback agent** - Afficher les étapes de raisonnement en temps réel
2. **Caching agent** - Mettre en cache les résultats des outils
3. **Agent memory** - Persistance du contexte entre sessions
4. **Métriques agent** - Dashboard des performances agent vs fast path
