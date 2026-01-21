# Plan d'implémentation : Entity Resolution sémantique

## Objectif

Implémenter un système de résolution d'entités en 4 phases qui :
1. Détecte les entités sémantiquement similaires (ex: "ML" ≈ "Machine Learning")
2. Vérifie les correspondances via LLM avec contexte relationnel
3. Fusionne en préservant et enrichissant toute l'information
4. Supporte le multi-langue (FR/EN/ES/DE)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    POST-GRAPH ENTITY RESOLUTION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: BLOCKING (Embeddings)           ← Coût: Faible                   │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │  Entities → Embed(name+type+desc) → ANN Search → Candidates     │       │
│  │  Réduction: O(n²) → O(n×k), ~99% de paires éliminées           │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  Phase 2: MATCHING (LLM)                  ← Coût: Moyen (filtré)           │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │  Pour chaque paire candidate:                                    │       │
│  │  - Contexte: noms, types, descriptions, relations               │       │
│  │  - Décision: SAME / DIFFERENT + canonical_name + confidence     │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  Phase 3: CLUSTERING (Union-Find)         ← Coût: Négligeable              │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │  Fermeture transitive: si A=B et B=C → cluster {A,B,C}          │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  Phase 4: MERGING (Enrichissement)        ← Coût: Faible-Moyen             │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │  - Descriptions: Fusionner/synthétiser (LLM optionnel)          │       │
│  │  - Attributs: Merger avec résolution de conflits                │       │
│  │  - Relations: Enrichir + dédupliquer                            │       │
│  │  - Aliases: Conserver tous les noms alternatifs                 │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Fichiers à créer/modifier

### Nouveaux fichiers

| Fichier | Description |
|---------|-------------|
| `src/cognidoc/entity_resolution.py` | Module principal (4 phases) |
| `src/cognidoc/prompts/entity_resolution.txt` | Prompt LLM pour matching |
| `src/cognidoc/prompts/description_merge.txt` | Prompt LLM pour fusion descriptions |
| `tests/test_entity_resolution.py` | Tests unitaires et intégration |

### Fichiers à modifier

| Fichier | Modifications |
|---------|---------------|
| `src/cognidoc/knowledge_graph.py` | Ajouter champ `aliases` à `GraphNode`, méthodes de merge |
| `src/cognidoc/graph_config.py` | Ajouter config `EntityResolutionConfig` |
| `src/cognidoc/run_ingestion_pipeline.py` | Intégrer phase de résolution |
| `src/cognidoc/constants.py` | Ajouter constantes (thresholds, prompts paths) |
| `src/cognidoc/checkpoint.py` | Ajouter `StageCheckpoint` pour entity_resolution |

---

## Implémentation détaillée

### 1. Structures de données

#### 1.1 Configuration (`graph_config.py`)

```python
@dataclass
class EntityResolutionConfig:
    """Entity resolution settings."""
    enabled: bool = True
    similarity_threshold: float = 0.75          # Min cosine pour candidats
    llm_confidence_threshold: float = 0.7       # Min confidence LLM pour merge
    max_concurrent_llm: int = 4                 # Appels LLM parallèles
    use_llm_for_descriptions: bool = True       # LLM pour fusionner descriptions
    batch_size: int = 500                       # Taille batch pour embeddings
    cache_decisions: bool = True                # Cache les décisions LLM
    cache_ttl_hours: int = 24                   # TTL du cache
```

#### 1.2 Extension de `GraphNode` (`knowledge_graph.py`)

```python
@dataclass
class GraphNode:
    # ... champs existants ...
    aliases: List[str] = field(default_factory=list)  # NOUVEAU
    merged_from: List[str] = field(default_factory=list)  # NOUVEAU: IDs originaux
```

#### 1.3 Structures de résolution (`entity_resolution.py`)

```python
@dataclass
class CandidatePair:
    """Paire candidate pour vérification LLM."""
    entity_a_id: str
    entity_b_id: str
    similarity_score: float

@dataclass
class ResolutionDecision:
    """Décision de résolution pour une paire."""
    same_entity: bool
    confidence: float
    canonical_name: str
    reasoning: str

@dataclass
class MergedEntity:
    """Entité après fusion enrichie."""
    canonical_id: str
    canonical_name: str
    type: str
    description: str
    attributes: Dict[str, Any]
    aliases: List[str]
    source_chunks: List[str]
    merged_from: List[str]
    confidence: float

@dataclass
class EntityResolutionResult:
    """Résultat global de la résolution."""
    original_entity_count: int
    final_entity_count: int
    clusters_found: int
    clusters_merged: int
    relationships_deduplicated: int
    relationships_enriched: int
    llm_calls_made: int
    cache_hits: int
    duration_seconds: float
```

---

### 2. Phase 1 : Blocking (Embeddings)

**Fichier:** `entity_resolution.py`

```python
async def compute_resolution_embeddings(
    entities: List[GraphNode],
    batch_size: int = 50,
) -> np.ndarray:
    """
    Compute embeddings for entity resolution.

    Text format: "{name} ({type}): {description}"
    Uses existing embedding infrastructure.
    """
    from .utils.embedding_providers import get_embedding_provider

    texts = []
    for entity in entities:
        text = entity.name
        if entity.type:
            text = f"{entity.name} ({entity.type})"
        if entity.description:
            text = f"{text}: {entity.description}"
        texts.append(text)

    provider = get_embedding_provider()
    embeddings = await provider.embed_async(texts, max_concurrent=4)

    return np.array(embeddings)


def find_candidate_pairs(
    entities: List[GraphNode],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.75,
) -> List[CandidatePair]:
    """
    Find candidate entity pairs using cosine similarity.

    Optimizations:
    - Batch matrix operations
    - Skip self-comparisons
    - Only keep upper triangle (avoid A,B and B,A duplicates)
    """
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)

    candidates = []
    n = len(entities)
    batch_size = 500

    for i in range(0, n, batch_size):
        batch_end = min(i + batch_size, n)
        batch = normalized[i:batch_end]

        # Similarity with all entities
        similarities = batch @ normalized.T

        for j, entity_idx in enumerate(range(i, batch_end)):
            sim_scores = similarities[j]
            sim_scores[entity_idx] = -1  # Exclude self

            # Find indices above threshold (only upper triangle)
            for cand_idx in range(entity_idx + 1, n):
                if sim_scores[cand_idx] >= similarity_threshold:
                    candidates.append(CandidatePair(
                        entity_a_id=entities[entity_idx].id,
                        entity_b_id=entities[cand_idx].id,
                        similarity_score=float(sim_scores[cand_idx]),
                    ))

    # Sort by similarity (highest first) for prioritization
    candidates.sort(key=lambda c: c.similarity_score, reverse=True)

    return candidates
```

**Optimisation optionnelle (FAISS) pour >10k entités:**

```python
def find_candidate_pairs_faiss(
    entities: List[GraphNode],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.75,
    k_neighbors: int = 20,
) -> List[CandidatePair]:
    """Use FAISS for efficient ANN on large entity sets."""
    try:
        import faiss
    except ImportError:
        logger.warning("FAISS not available, falling back to numpy")
        return find_candidate_pairs(entities, embeddings, similarity_threshold)

    # Normalize and build index
    embeddings_normalized = embeddings.copy()
    faiss.normalize_L2(embeddings_normalized)

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings_normalized)

    # Search
    D, I = index.search(embeddings_normalized, k_neighbors)

    candidates = []
    for i in range(len(entities)):
        for j, (neighbor_idx, similarity) in enumerate(zip(I[i], D[i])):
            if neighbor_idx > i and similarity >= similarity_threshold:
                candidates.append(CandidatePair(
                    entity_a_id=entities[i].id,
                    entity_b_id=entities[neighbor_idx].id,
                    similarity_score=float(similarity),
                ))

    return candidates
```

---

### 3. Phase 2 : Matching (LLM)

**Fichier:** `prompts/entity_resolution.txt`

```
You are an entity resolution expert. Determine if two entity mentions refer to the same real-world entity.

ENTITY A:
- Name: {name_a}
- Type: {type_a}
- Description: {desc_a}
- Relationships: {relations_a}

ENTITY B:
- Name: {name_b}
- Type: {type_b}
- Description: {desc_b}
- Relationships: {relations_b}

ANALYSIS GUIDELINES:
1. Name variations: Consider abbreviations (ML = Machine Learning), translations, synonyms
2. Types: Different types usually mean different entities (but not always)
3. Descriptions: Should be compatible, not contradictory
4. Relationships: Shared relationships strongly suggest same entity
5. Context: Use relationship targets to disambiguate (e.g., "Python" the language vs animal)

OUTPUT (JSON only):
{
  "same_entity": true/false,
  "confidence": 0.0-1.0,
  "canonical_name": "Best name (most complete/standard form)",
  "reasoning": "Brief explanation (1-2 sentences)"
}
```

**Fichier:** `entity_resolution.py`

```python
async def verify_candidate_pair(
    entity_a: GraphNode,
    entity_b: GraphNode,
    graph: KnowledgeGraph,
    config: EntityResolutionConfig,
) -> ResolutionDecision:
    """Verify a candidate pair using LLM."""

    # Get relationship context
    relations_a = get_entity_relations_summary(entity_a, graph, max_relations=5)
    relations_b = get_entity_relations_summary(entity_b, graph, max_relations=5)

    prompt = load_prompt("entity_resolution").format(
        name_a=entity_a.name,
        type_a=entity_a.type,
        desc_a=entity_a.description or "(no description)",
        relations_a=relations_a or "(no relationships)",
        name_b=entity_b.name,
        type_b=entity_b.type,
        desc_b=entity_b.description or "(no description)",
        relations_b=relations_b or "(no relationships)",
    )

    response = await llm_chat_async(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        json_mode=True,
    )

    result = extract_json_from_response(response, key="same_entity")

    return ResolutionDecision(
        same_entity=result.get("same_entity", False),
        confidence=result.get("confidence", 0.0),
        canonical_name=result.get("canonical_name", entity_a.name),
        reasoning=result.get("reasoning", ""),
    )


def get_entity_relations_summary(
    entity: GraphNode,
    graph: KnowledgeGraph,
    max_relations: int = 5,
) -> str:
    """Get a concise summary of entity relationships."""
    relations = []

    # Outgoing
    for successor in list(graph.graph.successors(entity.id))[:max_relations]:
        edge_data = graph.graph.edges[entity.id, successor]
        target = graph.nodes.get(successor)
        if target:
            rel_type = edge_data.get("relationship_type", "RELATED_TO")
            relations.append(f"--[{rel_type}]--> {target.name}")

    # Incoming
    remaining = max_relations - len(relations)
    for predecessor in list(graph.graph.predecessors(entity.id))[:remaining]:
        edge_data = graph.graph.edges[predecessor, entity.id]
        source = graph.nodes.get(predecessor)
        if source:
            rel_type = edge_data.get("relationship_type", "RELATED_TO")
            relations.append(f"<--[{rel_type}]-- {source.name}")

    return "; ".join(relations) if relations else ""


async def verify_candidates_batch(
    candidates: List[CandidatePair],
    graph: KnowledgeGraph,
    config: EntityResolutionConfig,
    cache: Optional[ToolCache] = None,
    show_progress: bool = True,
) -> List[Tuple[CandidatePair, ResolutionDecision]]:
    """Verify candidates in parallel with caching and progress."""

    semaphore = asyncio.Semaphore(config.max_concurrent_llm)
    results = []
    cache_hits = 0

    async def verify_one(pair: CandidatePair):
        nonlocal cache_hits

        # Check cache
        if cache and config.cache_decisions:
            cache_key = f"{pair.entity_a_id}:{pair.entity_b_id}"
            cached = cache.get("entity_resolution", cache_key)
            if cached:
                cache_hits += 1
                return (pair, ResolutionDecision(**cached))

        async with semaphore:
            entity_a = graph.nodes[pair.entity_a_id]
            entity_b = graph.nodes[pair.entity_b_id]

            decision = await verify_candidate_pair(
                entity_a, entity_b, graph, config
            )

            # Cache result
            if cache and config.cache_decisions:
                cache.set(
                    "entity_resolution",
                    cache_key,
                    asdict(decision),
                    ttl_minutes=config.cache_ttl_hours * 60,
                )

            return (pair, decision)

    if show_progress:
        from tqdm.asyncio import tqdm
        tasks = [verify_one(pair) for pair in candidates]
        results = await tqdm.gather(*tasks, desc="LLM verification")
    else:
        results = await asyncio.gather(*[verify_one(p) for p in candidates])

    # Filter to same_entity with sufficient confidence
    verified = [
        (pair, decision) for pair, decision in results
        if decision.same_entity and decision.confidence >= config.llm_confidence_threshold
    ]

    logger.info(f"Verified {len(verified)}/{len(candidates)} pairs as same entity (cache hits: {cache_hits})")

    return verified
```

---

### 4. Phase 3 : Clustering (Union-Find)

**Fichier:** `entity_resolution.py`

```python
class UnionFind:
    """Union-Find with canonical name tracking."""

    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.canonical_names = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: str, y: str, canonical_name: str = None) -> None:
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

        # Track canonical name
        if canonical_name:
            self.canonical_names[root_x] = canonical_name

    def get_clusters(self) -> Dict[str, List[str]]:
        clusters = defaultdict(list)
        for x in self.parent:
            clusters[self.find(x)].append(x)
        return {k: v for k, v in clusters.items() if len(v) > 1}

    def get_canonical_name(self, x: str) -> Optional[str]:
        return self.canonical_names.get(self.find(x))


def build_entity_clusters(
    verified_pairs: List[Tuple[CandidatePair, ResolutionDecision]]
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Build clusters from verified pairs using Union-Find.

    Returns:
        (clusters, canonical_names) where:
        - clusters: {canonical_id: [member_ids]}
        - canonical_names: {canonical_id: canonical_name}
    """
    uf = UnionFind()

    for pair, decision in verified_pairs:
        uf.union(pair.entity_a_id, pair.entity_b_id, decision.canonical_name)

    clusters = uf.get_clusters()
    canonical_names = {
        root: uf.get_canonical_name(root) or root
        for root in clusters.keys()
    }

    return clusters, canonical_names
```

---

### 5. Phase 4 : Merging (Enrichissement)

**Fichier:** `prompts/description_merge.txt`

```
Merge the following descriptions of "{entity_name}" into a single comprehensive description.

IMPORTANT: Preserve ALL unique information from each description. Do not lose any details.
Remove only exact redundancies.

DESCRIPTIONS:
{descriptions}

OUTPUT: A single merged description (2-4 sentences max) that contains all unique information from the sources above.
```

**Fichier:** `entity_resolution.py`

```python
async def merge_descriptions(
    descriptions: List[str],
    entity_name: str,
    use_llm: bool = True,
) -> str:
    """Merge multiple descriptions preserving all information."""

    valid = [d.strip() for d in descriptions if d and d.strip()]
    if not valid:
        return ""
    if len(valid) == 1:
        return valid[0]

    # Remove exact duplicates
    unique = list(dict.fromkeys(valid))
    if len(unique) == 1:
        return unique[0]

    if not use_llm:
        return concatenate_descriptions_smart(unique)

    prompt = load_prompt("description_merge").format(
        entity_name=entity_name,
        descriptions="\n".join(f"- {d}" for d in unique),
    )

    response = await llm_chat_async(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.strip()


def concatenate_descriptions_smart(descriptions: List[str]) -> str:
    """Fallback: concatenate with sentence-level deduplication."""
    import re

    all_sentences = []
    seen_normalized = set()

    for desc in descriptions:
        sentences = re.split(r'(?<=[.!?])\s+', desc)
        for sent in sentences:
            normalized = sent.lower().strip()
            if normalized and normalized not in seen_normalized:
                if not any(word_overlap(normalized, s) > 0.8 for s in seen_normalized):
                    all_sentences.append(sent.strip())
                    seen_normalized.add(normalized)

    return " ".join(all_sentences)


def word_overlap(s1: str, s2: str) -> float:
    """Calculate word overlap ratio."""
    w1, w2 = set(s1.split()), set(s2.split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / min(len(w1), len(w2))


def merge_attributes(attributes_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge attributes with conflict resolution."""
    merged = {}
    all_values = defaultdict(list)

    for attrs in attributes_list:
        for key, value in attrs.items():
            all_values[key].append(value)

    for key, values in all_values.items():
        unique = list(dict.fromkeys(str(v) for v in values))

        if len(unique) == 1:
            merged[key] = values[0]
        else:
            merged[key] = resolve_attribute_conflict(key, values)

    return merged


def resolve_attribute_conflict(key: str, values: List[Any]) -> Any:
    """Resolve conflicting attribute values."""
    types = set(type(v) for v in values)

    if len(types) == 1:
        val_type = types.pop()

        if val_type == list:
            merged = []
            for v in values:
                merged.extend(v)
            return list(dict.fromkeys(merged))

        if val_type in (int, float):
            return max(values)  # Or could be configurable

        if val_type == str:
            return max(values, key=lambda s: len(s) if s else 0)

    # Incompatible: keep all as list
    return list(dict.fromkeys(str(v) for v in values))


async def merge_entity_cluster(
    cluster: List[str],
    graph: KnowledgeGraph,
    canonical_name: str,
    config: EntityResolutionConfig,
) -> MergedEntity:
    """Merge a cluster with full enrichment."""

    entities = [graph.nodes[eid] for eid in cluster if eid in graph.nodes]
    if not entities:
        raise ValueError(f"No valid entities in cluster: {cluster}")

    # 1. Canonical name (provided by LLM or fallback to most frequent)
    if not canonical_name:
        name_counts = Counter(e.name for e in entities)
        canonical_name = max(name_counts.keys(), key=lambda n: (name_counts[n], len(n)))

    # 2. Aliases (all other names)
    aliases = list(set(e.name for e in entities if e.name != canonical_name))

    # 3. Merge descriptions (ENRICHED)
    all_descriptions = [e.description for e in entities]
    merged_description = await merge_descriptions(
        all_descriptions,
        canonical_name,
        use_llm=config.use_llm_for_descriptions,
    )

    # 4. Merge attributes (ENRICHED)
    all_attributes = [e.attributes for e in entities if e.attributes]
    merged_attributes = merge_attributes(all_attributes) if all_attributes else {}

    # 5. Type (majority vote)
    type_counts = Counter(e.type for e in entities)
    canonical_type = type_counts.most_common(1)[0][0]

    # 6. Merge source chunks
    all_chunks = []
    for e in entities:
        all_chunks.extend(e.source_chunks)
    unique_chunks = list(dict.fromkeys(all_chunks))

    # 7. Average confidence
    avg_confidence = sum(getattr(e, 'confidence', 1.0) for e in entities) / len(entities)

    return MergedEntity(
        canonical_id=cluster[0],
        canonical_name=canonical_name,
        type=canonical_type,
        description=merged_description,
        attributes=merged_attributes,
        aliases=aliases,
        source_chunks=unique_chunks,
        merged_from=cluster,
        confidence=avg_confidence,
    )


def merge_cluster_relationships(
    cluster: List[str],
    merged_entity: MergedEntity,
    graph: KnowledgeGraph,
    all_clusters: Dict[str, List[str]],
) -> List[Tuple[str, str, str, str, List[str], float]]:
    """
    Merge relationships with enrichment.

    Returns list of (source_id, target_id, rel_type, description, source_chunks, weight)
    """
    # Map entity IDs to their canonical cluster ID
    id_to_canonical = {}
    for canonical_id, members in all_clusters.items():
        for member in members:
            id_to_canonical[member] = canonical_id

    # Collect relationships grouped by (resolved_target, rel_type)
    relation_groups = defaultdict(lambda: {
        "descriptions": [],
        "source_chunks": [],
        "weight": 0.0,
    })

    for entity_id in cluster:
        if entity_id not in graph.graph:
            continue

        for successor in graph.graph.successors(entity_id):
            edge = graph.graph.edges[entity_id, successor]
            rel_type = edge.get("relationship_type", "RELATED_TO")

            # Resolve target to canonical
            resolved_target = id_to_canonical.get(successor, successor)

            key = (resolved_target, rel_type)
            group = relation_groups[key]

            desc = edge.get("description", "")
            if desc and desc not in group["descriptions"]:
                group["descriptions"].append(desc)
            group["source_chunks"].extend(edge.get("source_chunks", []))
            group["weight"] += edge.get("weight", 1.0)

    # Build merged relationships
    merged = []
    for (target_id, rel_type), group in relation_groups.items():
        # Merge descriptions
        if len(group["descriptions"]) > 1:
            merged_desc = concatenate_descriptions_smart(group["descriptions"])
        else:
            merged_desc = group["descriptions"][0] if group["descriptions"] else ""

        merged.append((
            merged_entity.canonical_id,
            target_id,
            rel_type,
            merged_desc,
            list(dict.fromkeys(group["source_chunks"])),
            group["weight"],
        ))

    return merged
```

---

### 6. Orchestration principale

**Fichier:** `entity_resolution.py`

```python
@dataclass
class EntityResolutionResult:
    """Complete resolution result with stats."""
    original_entity_count: int
    final_entity_count: int
    clusters_found: int
    clusters_merged: int
    entities_merged: int
    relationships_deduplicated: int
    relationships_enriched: int
    llm_calls_made: int
    cache_hits: int
    duration_seconds: float


async def resolve_entities(
    graph: KnowledgeGraph,
    config: EntityResolutionConfig = None,
    show_progress: bool = True,
) -> EntityResolutionResult:
    """
    Run complete entity resolution pipeline.

    Args:
        graph: Knowledge graph to deduplicate (modified in place)
        config: Resolution configuration
        show_progress: Show progress bars

    Returns:
        Resolution statistics
    """
    import time
    start_time = time.time()

    if config is None:
        config = EntityResolutionConfig()

    if not config.enabled:
        return EntityResolutionResult(
            original_entity_count=len(graph.nodes),
            final_entity_count=len(graph.nodes),
            clusters_found=0,
            clusters_merged=0,
            entities_merged=0,
            relationships_deduplicated=0,
            relationships_enriched=0,
            llm_calls_made=0,
            cache_hits=0,
            duration_seconds=0.0,
        )

    original_count = len(graph.nodes)
    entities = list(graph.nodes.values())

    logger.info(f"Starting entity resolution for {original_count} entities...")

    # Phase 1: Compute embeddings and find candidates
    logger.info("Phase 1: Computing embeddings and finding candidates...")
    embeddings = await compute_resolution_embeddings(
        entities,
        batch_size=config.batch_size,
    )

    candidates = find_candidate_pairs(
        entities,
        embeddings,
        config.similarity_threshold,
    )
    logger.info(f"Found {len(candidates)} candidate pairs")

    if not candidates:
        return EntityResolutionResult(
            original_entity_count=original_count,
            final_entity_count=original_count,
            clusters_found=0,
            clusters_merged=0,
            entities_merged=0,
            relationships_deduplicated=0,
            relationships_enriched=0,
            llm_calls_made=0,
            cache_hits=0,
            duration_seconds=time.time() - start_time,
        )

    # Phase 2: LLM verification
    logger.info("Phase 2: Verifying candidates with LLM...")
    cache = get_tool_cache() if config.cache_decisions else None

    verified = await verify_candidates_batch(
        candidates,
        graph,
        config,
        cache=cache,
        show_progress=show_progress,
    )

    llm_calls = len(candidates)
    cache_hits = llm_calls - len([v for v in verified])  # Approximation

    if not verified:
        logger.info("No verified matches found")
        return EntityResolutionResult(
            original_entity_count=original_count,
            final_entity_count=original_count,
            clusters_found=0,
            clusters_merged=0,
            entities_merged=0,
            relationships_deduplicated=0,
            relationships_enriched=0,
            llm_calls_made=llm_calls,
            cache_hits=0,
            duration_seconds=time.time() - start_time,
        )

    # Phase 3: Clustering
    logger.info("Phase 3: Building entity clusters...")
    clusters, canonical_names = build_entity_clusters(verified)
    logger.info(f"Found {len(clusters)} clusters to merge")

    # Phase 4: Merging
    logger.info("Phase 4: Merging entities and relationships...")
    stats = await apply_merges(graph, clusters, canonical_names, config)

    duration = time.time() - start_time

    result = EntityResolutionResult(
        original_entity_count=original_count,
        final_entity_count=len(graph.nodes),
        clusters_found=len(clusters),
        clusters_merged=len(clusters),
        entities_merged=stats["entities_merged"],
        relationships_deduplicated=stats["relations_deduped"],
        relationships_enriched=stats["relations_enriched"],
        llm_calls_made=llm_calls,
        cache_hits=cache_hits,
        duration_seconds=duration,
    )

    logger.info(
        f"Entity resolution complete: {original_count} → {result.final_entity_count} entities "
        f"({result.entities_merged} merged from {len(clusters)} clusters) in {duration:.1f}s"
    )

    return result


async def apply_merges(
    graph: KnowledgeGraph,
    clusters: Dict[str, List[str]],
    canonical_names: Dict[str, str],
    config: EntityResolutionConfig,
) -> Dict[str, int]:
    """Apply merges to the graph."""

    stats = {
        "entities_merged": 0,
        "relations_deduped": 0,
        "relations_enriched": 0,
    }

    for canonical_id, member_ids in clusters.items():
        # Merge entity
        merged = await merge_entity_cluster(
            member_ids,
            graph,
            canonical_names.get(canonical_id),
            config,
        )

        # Update graph node
        node = graph.nodes[canonical_id]
        node.name = merged.canonical_name
        node.description = merged.description
        node.attributes = merged.attributes
        node.aliases = merged.aliases
        node.source_chunks = merged.source_chunks

        # Update NetworkX node attributes
        graph.graph.nodes[canonical_id]["name"] = merged.canonical_name
        graph.graph.nodes[canonical_id]["description"] = merged.description
        graph.graph.nodes[canonical_id]["aliases"] = merged.aliases

        # Remove merged nodes (keep canonical)
        for member_id in member_ids:
            if member_id != canonical_id:
                # Redirect edges
                redirect_edges(graph, member_id, canonical_id)

                # Remove node
                if member_id in graph.nodes:
                    del graph.nodes[member_id]
                if member_id in graph.graph:
                    graph.graph.remove_node(member_id)

                # Update name mapping
                normalized = graph._normalize_name(graph.nodes.get(member_id, GraphNode(id="", name="")).name)
                if normalized in graph._name_to_id:
                    del graph._name_to_id[normalized]

        # Add aliases to name mapping
        for alias in merged.aliases:
            normalized = graph._normalize_name(alias)
            graph._name_to_id[normalized] = canonical_id

        stats["entities_merged"] += len(member_ids) - 1

    # Deduplicate and merge relationships
    stats.update(deduplicate_relationships(graph, clusters))

    return stats


def redirect_edges(graph: KnowledgeGraph, old_id: str, new_id: str) -> None:
    """Redirect all edges from old_id to new_id."""

    if old_id not in graph.graph:
        return

    # Outgoing edges
    for successor in list(graph.graph.successors(old_id)):
        edge_data = dict(graph.graph.edges[old_id, successor])
        if not graph.graph.has_edge(new_id, successor):
            graph.graph.add_edge(new_id, successor, **edge_data)
        else:
            # Merge edge data
            existing = graph.graph.edges[new_id, successor]
            existing["weight"] = existing.get("weight", 1) + edge_data.get("weight", 1)
            chunks = existing.get("source_chunks", [])
            chunks.extend(edge_data.get("source_chunks", []))
            existing["source_chunks"] = list(dict.fromkeys(chunks))

    # Incoming edges
    for predecessor in list(graph.graph.predecessors(old_id)):
        edge_data = dict(graph.graph.edges[predecessor, old_id])
        if not graph.graph.has_edge(predecessor, new_id):
            graph.graph.add_edge(predecessor, new_id, **edge_data)
        else:
            existing = graph.graph.edges[predecessor, new_id]
            existing["weight"] = existing.get("weight", 1) + edge_data.get("weight", 1)
            chunks = existing.get("source_chunks", [])
            chunks.extend(edge_data.get("source_chunks", []))
            existing["source_chunks"] = list(dict.fromkeys(chunks))


def deduplicate_relationships(
    graph: KnowledgeGraph,
    clusters: Dict[str, List[str]],
) -> Dict[str, int]:
    """Deduplicate relationships after entity merging."""

    # Build reverse mapping
    id_to_canonical = {}
    for canonical_id, members in clusters.items():
        for member in members:
            id_to_canonical[member] = canonical_id

    stats = {"relations_deduped": 0, "relations_enriched": 0}

    # Check all edges for duplicates
    edges_to_check = list(graph.graph.edges(data=True))
    seen = set()

    for source, target, data in edges_to_check:
        # Resolve to canonical IDs
        canonical_source = id_to_canonical.get(source, source)
        canonical_target = id_to_canonical.get(target, target)
        rel_type = data.get("relationship_type", "RELATED_TO")

        key = (canonical_source, canonical_target, rel_type)

        if key in seen:
            # This is a duplicate - remove it
            if graph.graph.has_edge(source, target):
                graph.graph.remove_edge(source, target)
            stats["relations_deduped"] += 1
        else:
            seen.add(key)

    return stats
```

---

### 7. Intégration pipeline

**Fichier:** `run_ingestion_pipeline.py` (modification)

```python
# Ajouter l'import
from .entity_resolution import resolve_entities, EntityResolutionConfig

# Ajouter l'argument
parser.add_argument(
    "--skip-resolution",
    action="store_true",
    help="Skip entity resolution (deduplication)"
)
parser.add_argument(
    "--resolution-threshold",
    type=float,
    default=0.75,
    help="Similarity threshold for entity resolution candidates (default: 0.75)"
)

# Dans run_ingestion_pipeline_async, après la construction du graphe:

# 10. Entity Resolution (after graph building)
if not skip_graph and not skip_resolution:
    pipeline_timer.stage("entity_resolution")
    try:
        logger.info("Running entity resolution...")

        resolution_config = EntityResolutionConfig(
            similarity_threshold=resolution_threshold,
            max_concurrent_llm=entity_max_concurrent,
        )

        resolution_result = await resolve_entities(
            knowledge_graph,
            config=resolution_config,
            show_progress=True,
        )

        stats["entity_resolution"] = {
            "original_entities": resolution_result.original_entity_count,
            "final_entities": resolution_result.final_entity_count,
            "clusters_merged": resolution_result.clusters_merged,
            "entities_merged": resolution_result.entities_merged,
            "duration_seconds": resolution_result.duration_seconds,
        }

        # Re-save graph after resolution
        knowledge_graph.save()

        logger.info(
            f"Entity resolution: {resolution_result.original_entity_count} → "
            f"{resolution_result.final_entity_count} entities"
        )
    except Exception as e:
        logger.error(f"Entity resolution failed: {e}")
        # Continue - resolution failure shouldn't stop pipeline
```

---

### 8. Tests

**Fichier:** `tests/test_entity_resolution.py`

```python
"""Tests for entity resolution module."""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from cognidoc.entity_resolution import (
    find_candidate_pairs,
    build_entity_clusters,
    merge_descriptions,
    merge_attributes,
    UnionFind,
    CandidatePair,
    ResolutionDecision,
)
from cognidoc.knowledge_graph import GraphNode, KnowledgeGraph


class TestBlocking:
    """Tests for Phase 1: Blocking."""

    def test_find_candidate_pairs_basic(self):
        """Should find similar entity pairs."""
        entities = [
            GraphNode(id="1", name="Machine Learning", type="CONCEPT", description="AI technique"),
            GraphNode(id="2", name="ML", type="CONCEPT", description="Machine learning abbrev"),
            GraphNode(id="3", name="Python", type="LANGUAGE", description="Programming language"),
        ]

        # Mock embeddings where 1 and 2 are similar
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # Machine Learning
            [0.95, 0.1, 0.0],  # ML (similar to Machine Learning)
            [0.0, 0.0, 1.0],  # Python (different)
        ])

        candidates = find_candidate_pairs(entities, embeddings, threshold=0.9)

        assert len(candidates) == 1
        assert candidates[0].entity_a_id == "1"
        assert candidates[0].entity_b_id == "2"
        assert candidates[0].similarity_score > 0.9

    def test_find_candidate_pairs_no_self_match(self):
        """Should not match entity with itself."""
        entities = [GraphNode(id="1", name="Test", type="TEST", description="")]
        embeddings = np.array([[1.0, 0.0]])

        candidates = find_candidate_pairs(entities, embeddings, threshold=0.5)

        assert len(candidates) == 0

    def test_find_candidate_pairs_threshold(self):
        """Should respect similarity threshold."""
        entities = [
            GraphNode(id="1", name="A", type="T", description=""),
            GraphNode(id="2", name="B", type="T", description=""),
        ]
        embeddings = np.array([
            [1.0, 0.0],
            [0.7, 0.7],  # cos similarity ~0.7
        ])

        # High threshold: no matches
        assert len(find_candidate_pairs(entities, embeddings, threshold=0.9)) == 0

        # Lower threshold: match
        assert len(find_candidate_pairs(entities, embeddings, threshold=0.6)) == 1


class TestClustering:
    """Tests for Phase 3: Clustering."""

    def test_union_find_basic(self):
        """Test basic union-find operations."""
        uf = UnionFind()

        uf.union("a", "b", "A")
        uf.union("b", "c", "A")  # Transitive: a, b, c same cluster

        assert uf.find("a") == uf.find("b") == uf.find("c")

        clusters = uf.get_clusters()
        assert len(clusters) == 1
        assert set(list(clusters.values())[0]) == {"a", "b", "c"}

    def test_union_find_separate_clusters(self):
        """Test separate clusters remain separate."""
        uf = UnionFind()

        uf.union("a", "b")
        uf.union("c", "d")

        assert uf.find("a") == uf.find("b")
        assert uf.find("c") == uf.find("d")
        assert uf.find("a") != uf.find("c")

    def test_build_entity_clusters(self):
        """Test cluster building from verified pairs."""
        verified = [
            (CandidatePair("1", "2", 0.9), ResolutionDecision(True, 0.95, "Entity A", "")),
            (CandidatePair("2", "3", 0.85), ResolutionDecision(True, 0.9, "Entity A", "")),
        ]

        clusters, names = build_entity_clusters(verified)

        assert len(clusters) == 1
        cluster = list(clusters.values())[0]
        assert set(cluster) == {"1", "2", "3"}


class TestMerging:
    """Tests for Phase 4: Merging."""

    @pytest.mark.asyncio
    async def test_merge_descriptions_single(self):
        """Single description returns as-is."""
        result = await merge_descriptions(["Only one"], "Entity", use_llm=False)
        assert result == "Only one"

    @pytest.mark.asyncio
    async def test_merge_descriptions_duplicates(self):
        """Duplicate descriptions are deduplicated."""
        result = await merge_descriptions(
            ["Same text", "Same text", "Same text"],
            "Entity",
            use_llm=False,
        )
        assert result == "Same text"

    @pytest.mark.asyncio
    async def test_merge_descriptions_concatenate(self):
        """Different descriptions are concatenated."""
        result = await merge_descriptions(
            ["First fact.", "Second fact."],
            "Entity",
            use_llm=False,
        )
        assert "First fact" in result
        assert "Second fact" in result

    def test_merge_attributes_no_conflict(self):
        """Non-conflicting attributes are merged."""
        attrs = [
            {"a": 1, "b": 2},
            {"c": 3},
        ]

        merged = merge_attributes(attrs)

        assert merged == {"a": 1, "b": 2, "c": 3}

    def test_merge_attributes_conflict_string(self):
        """String conflicts keep longest."""
        attrs = [
            {"name": "short"},
            {"name": "much longer name"},
        ]

        merged = merge_attributes(attrs)

        assert merged["name"] == "much longer name"

    def test_merge_attributes_conflict_list(self):
        """List conflicts are merged."""
        attrs = [
            {"tags": ["a", "b"]},
            {"tags": ["b", "c"]},
        ]

        merged = merge_attributes(attrs)

        assert set(merged["tags"]) == {"a", "b", "c"}


class TestIntegration:
    """Integration tests for full resolution pipeline."""

    @pytest.mark.asyncio
    async def test_resolve_entities_no_duplicates(self):
        """Graph with no duplicates returns unchanged."""
        from cognidoc.entity_resolution import resolve_entities, EntityResolutionConfig

        graph = KnowledgeGraph()
        graph.add_entity(Mock(id="1", name="Entity A", type="TYPE", description="Desc A",
                              attributes={}, source_chunk="c1"))
        graph.add_entity(Mock(id="2", name="Entity B", type="TYPE", description="Desc B",
                              attributes={}, source_chunk="c2"))

        config = EntityResolutionConfig(similarity_threshold=0.99)

        with patch("cognidoc.entity_resolution.compute_resolution_embeddings") as mock_embed:
            mock_embed.return_value = np.array([[1, 0], [0, 1]])  # Orthogonal = no match

            result = await resolve_entities(graph, config)

        assert result.clusters_merged == 0
        assert result.final_entity_count == 2
```

---

## Ordre d'implémentation

| Étape | Fichiers | Estimation |
|-------|----------|------------|
| 1 | `constants.py` - Ajouter constantes | 10 min |
| 2 | `graph_config.py` - Ajouter `EntityResolutionConfig` | 15 min |
| 3 | `knowledge_graph.py` - Ajouter `aliases` à `GraphNode` | 20 min |
| 4 | `prompts/entity_resolution.txt` | 10 min |
| 5 | `prompts/description_merge.txt` | 5 min |
| 6 | `entity_resolution.py` - Phase 1 (Blocking) | 45 min |
| 7 | `entity_resolution.py` - Phase 2 (Matching) | 60 min |
| 8 | `entity_resolution.py` - Phase 3 (Clustering) | 30 min |
| 9 | `entity_resolution.py` - Phase 4 (Merging) | 60 min |
| 10 | `entity_resolution.py` - Orchestration | 45 min |
| 11 | `run_ingestion_pipeline.py` - Intégration | 30 min |
| 12 | `tests/test_entity_resolution.py` | 60 min |
| 13 | Tests d'intégration et debug | 60 min |

**Total estimé:** ~7-8 heures de développement

---

## Configuration recommandée

```yaml
# config/graph_schema.yaml

entity_resolution:
  enabled: true
  similarity_threshold: 0.75      # Augmenter si trop de faux positifs
  llm_confidence_threshold: 0.7   # Augmenter pour plus de précision
  max_concurrent_llm: 4           # Adapter selon provider
  use_llm_for_descriptions: true  # false pour économiser des calls
  cache_decisions: true
  cache_ttl_hours: 24
```

---

## Prochaines étapes

1. **Valider ce plan** - Questions/modifications ?
2. **Implémenter par étapes** - Commencer par Phase 1 + tests
3. **Tester sur données réelles** - Ajuster thresholds
4. **Documenter** - Ajouter à CLAUDE.md
