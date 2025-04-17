# Multi-Domain RAG

This module implements the Retrieval-Augmented Generation (RAG) with Multiple Knowledge Bases architectural pattern described in the DomainFusion white paper. The pattern extends traditional RAG by incorporating domain-specific knowledge bases and retrieval strategies, enabling complex cross-domain information integration.

## Core Classes

### VectorStore

`VectorStore` provides functionality for storing, embedding, and retrieving domain-specific knowledge.

```python
from domainfusion.rag import VectorStore

# Create a vector store for a specific domain
biomechanics_store = VectorStore(
    embedding_model=embedding_model,
    name="biomechanics",
    storage_path="./vector_stores"
)

# Add documents
biomechanics_store.add_documents("path/to/biomechanics_docs", 
                               chunk_size=500,
                               metadata={"domain": "biomechanics"})
```

#### Methods

| Method | Description |
|--------|-------------|
| `__init__(embedding_model=None, name="default", storage_path=None)` | Initialize a vector store with an optional embedding model, name, and storage path |
| `add_document(document, metadata=None)` | Add a single document to the vector store |
| `add_documents(documents, chunk_size=1000, metadata=None)` | Add multiple documents or a directory of documents |
| `query(query, top_k=5, filter=None)` | Query the vector store for relevant documents |
| `query_by_embedding(embedding, top_k=5, filter=None)` | Query using a pre-computed embedding |
| `save(path=None)` | Save the vector store to disk |
| `load(path=None)` | Load the vector store from disk |

### Retriever Classes

The module provides several retriever implementations for fetching relevant information from knowledge bases:

#### Retriever (Abstract Base Class)

Defines the interface for retrievers:

| Method | Description |
|--------|-------------|
| `retrieve(query, top_k=5)` | Retrieve relevant documents for a query |
| `get_relevant_text(query, top_k=5)` | Get a combined text string of relevant information |

#### DomainRetriever

Retrieves information from a single domain knowledge base:

```python
from domainfusion.rag import DomainRetriever, VectorStore

# Create a vector store
biomechanics_store = VectorStore(embedding_model)

# Create a domain-specific retriever
biomechanics_retriever = DomainRetriever(
    vector_store=biomechanics_store,
    domain="biomechanics"
)

# Retrieve documents
results = biomechanics_retriever.retrieve("What factors affect sprint acceleration?")
```

| Method | Description |
|--------|-------------|
| `__init__(vector_store, domain=None)` | Initialize with a vector store and optional domain name |
| `retrieve(query, top_k=5)` | Retrieve documents from the domain-specific vector store |
| `get_relevant_text(query, top_k=5)` | Get relevant text as a formatted string |

#### HybridRetriever

Combines results from multiple domain retrievers:

```python
from domainfusion.rag import HybridRetriever

# Create a hybrid retriever with multiple domain retrievers
hybrid_retriever = HybridRetriever(
    retrievers=[biomechanics_retriever, physiology_retriever, nutrition_retriever],
    embedding_model=embedding_model,
    strategy="weighted"
)

# Retrieve documents across domains
results = hybrid_retriever.retrieve("How do muscle fiber types affect sprint performance?")
```

| Method | Description |
|--------|-------------|
| `__init__(retrievers, embedding_model=None, strategy="top_scoring")` | Initialize with domain retrievers and strategy |
| `retrieve(query, top_k=5)` | Retrieve documents from multiple domain-specific retrievers |
| `get_relevant_text(query, top_k=5)` | Get relevant text organized by domain |

The `strategy` parameter supports three retrieval strategies:
- `top_scoring`: Select the highest scoring documents regardless of domain
- `round_robin`: Take documents from each domain in turn
- `weighted`: Prioritize domains based on query similarity

### DomainRouter

`DomainRouter` determines which knowledge bases are relevant to a query:

```python
from domainfusion.rag import DomainRouter

# Create a router
router = DomainRouter(
    embedding_model=embedding_model,
    threshold=0.5
)

# Add domains with descriptions
router.add_domain("biomechanics", 
                 "Biomechanics focuses on the mechanical laws relating to the movement of living organisms")
router.add_domain("physiology", 
                 "Exercise physiology studies the physiological responses to physical activity")

# Route a query
relevant_domains = router.route("What muscles are most active during sprinting?")
```

| Method | Description |
|--------|-------------|
| `__init__(embedding_model=None, threshold=0.5)` | Initialize with embedding model and relevance threshold |
| `add_domain(name, description)` | Add a domain with semantic description |
| `route(query)` | Return list of relevant domain names for a query |
| `route_with_scores(query)` | Return dictionary of domain names and relevance scores |

### MultiDomainRAG

`MultiDomainRAG` integrates the routing, retrieval, and generation components:

```python
from domainfusion.rag import MultiDomainRAG, DomainRouter, VectorStore

# Create a multi-domain RAG system
rag = MultiDomainRAG(
    generation_model=llm_model,
    domain_router=router
)

# Add knowledge bases
rag.add_knowledge_base(
    name="biomechanics",
    vector_store=biomechanics_store,
    description="Biomechanics focuses on the mechanical laws relating to the movement of living organisms"
)

rag.add_knowledge_base(
    name="physiology",
    vector_store=physiology_store,
    description="Exercise physiology studies the physiological responses to physical activity"
)

# Generate a response
response = rag.generate("How does muscle fiber composition affect sprint performance?")
```

| Method | Description |
|--------|-------------|
| `__init__(generation_model, domain_router=None, retrievers=None, template=None)` | Initialize the RAG system |
| `add_knowledge_base(name, vector_store, description=None)` | Add a domain-specific knowledge base |
| `generate(query, max_tokens=1000, use_all_retrievers=False)` | Generate a response using relevant knowledge |
| `generate_with_sources(query, max_tokens=1000)` | Generate a response with source citations |
| `add_domain_specific_templates(templates)` | Add domain-specific prompt templates |
| `generate_with_domain_templates(query, max_tokens=1000)` | Generate using domain-specific templates |

## Example Usage

### Basic Single-Domain RAG

```python
from domainfusion import ModelRegistry
from domainfusion.rag import VectorStore, DomainRetriever, MultiDomainRAG

# Create model registry
registry = ModelRegistry()
registry.add_model("embedding", engine="openai", model_name="text-embedding-3-small")
registry.add_model("generation", engine="anthropic", model_name="claude-3-sonnet")

# Create vector store
biomechanics_store = VectorStore(
    embedding_model=registry.get("embedding"),
    name="biomechanics"
)

# Load documents
biomechanics_store.add_documents("data/biomechanics", 
                               metadata={"domain": "biomechanics"})

# Create simple RAG system
rag = MultiDomainRAG(
    generation_model=registry.get("generation")
)

# Add knowledge base
rag.add_knowledge_base(
    name="biomechanics",
    vector_store=biomechanics_store,
    description="Biomechanics focuses on the mechanical laws relating to the movement of living organisms"
)

# Generate a response
response = rag.generate("What's the ideal technique for 100m sprint start?")
```

### Multi-Domain RAG with Domain Router

```python
from domainfusion import ModelRegistry
from domainfusion.rag import VectorStore, DomainRouter, MultiDomainRAG

# Create model registry
registry = ModelRegistry()
registry.add_model("embedding", engine="openai", model_name="text-embedding-3-small")
registry.add_model("generation", engine="anthropic", model_name="claude-3-sonnet")

# Create vector stores for each domain
biomechanics_store = VectorStore(embedding_model=registry.get("embedding"), name="biomechanics")
physiology_store = VectorStore(embedding_model=registry.get("embedding"), name="physiology")
nutrition_store = VectorStore(embedding_model=registry.get("embedding"), name="nutrition")

# Load documents
biomechanics_store.add_documents("data/biomechanics")
physiology_store.add_documents("data/physiology")
nutrition_store.add_documents("data/nutrition")

# Create a domain router
router = DomainRouter(
    embedding_model=registry.get("embedding"),
    threshold=0.5
)

# Create the RAG system
rag = MultiDomainRAG(
    generation_model=registry.get("generation"),
    domain_router=router
)

# Add knowledge bases
rag.add_knowledge_base(
    name="biomechanics",
    vector_store=biomechanics_store,
    description="Biomechanics focuses on the mechanical laws relating to the movement of living organisms"
)

rag.add_knowledge_base(
    name="physiology",
    vector_store=physiology_store,
    description="Exercise physiology studies the physiological responses to physical activity"
)

rag.add_knowledge_base(
    name="nutrition",
    vector_store=nutrition_store,
    description="Sports nutrition focuses on the dietary needs and strategies to enhance athletic performance"
)

# Add domain-specific templates
rag.add_domain_specific_templates({
    "biomechanics": """
    Answer the biomechanics question based on the provided context.
    Focus on mechanical aspects like forces, angles, and movement patterns.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """,
    
    "physiology": """
    Answer the physiology question based on the provided context.
    Focus on physiological processes like energy systems, muscle fiber types, and metabolic responses.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
})

# Generate a response
response = rag.generate_with_domain_templates("How does muscle fiber type affect sprint acceleration?")
```

## Implementation Considerations

### Vector Database Management

The module supports several approaches to organizing and querying domain-specific knowledge:

1. **Separate Databases**: Using distinct `VectorStore` instances for each domain
2. **Single Database with Domain Tags**: Using a single `VectorStore` with domain metadata
3. **Hybrid Approach**: Using domain-specific embedding models with a unified database

### Retrieval Strategies

The `HybridRetriever` class implements three strategies for combining results:

1. **Top-Scoring**: Selects documents with the highest relevance scores regardless of domain
2. **Round-Robin**: Takes documents from each domain in alternating fashion
3. **Weighted**: Prioritizes domains based on their relevance to the query

### Domain-Specific Templates

The `MultiDomainRAG` class supports domain-specific prompt templates, allowing the generation process to be tailored to the most relevant domain for a given query.

## Performance Optimization

For optimal performance:

1. Use chunking when adding documents (`add_documents` with `chunk_size`)
2. Save and load vector stores to/from disk to avoid re-embedding
3. Adjust the router threshold based on your domain overlap
4. Use domain-specific templates for specialized queries
5. Consider the hybrid retrieval approach for queries that span domains
