<h1 align="center">Combine Harvester</h1>

<p align="center"><em>When you mix coffee, red bull and jetfuel </em></p>

<p align="center">
  <img src="combine_harvester.png" alt="Combine Harvester Logo">
</p>



<div class="justified">

# Introduction

The emergence of Large Language Models (LLMs) has transformed artificial intelligence research and applications. While general-purpose models like GPT-4 and Claude demonstrate impressive capabilities across a wide range of tasks, they often lack the depth of expertise required for specialized domains. This limitation has led to the development of domain-expert LLMs---models fine-tuned or prompted to excel in specific fields such as medicine, law, or scientific research.

## The Challenge of Domain Expertise in LLMs

Domain expertise in LLMs presents a fundamental tension. On one hand, specialization improves performance within a domain by focusing the model's capabilities on domain-specific knowledge, terminology, and reasoning patterns. On the other hand, this specialization often comes at the cost of reduced performance in other domains, creating \"knowledge silos\" that mirror the specialization patterns in human expertise.

General-purpose models attempt to balance breadth and depth, but inevitably make trade-offs. As @Smith2023 demonstrated, even the most advanced general models show significant performance gaps compared to specialized models when evaluated on domain-specific benchmarks. For instance, while GPT-4 achieves an average score of 76% across medical diagnostic tasks, MedicalGPT---a specialized model---achieves 89% on the same benchmark.

## The Need for Multi-Domain Integration

Real-world problems rarely confine themselves to neat disciplinary boundaries. A sports scientist studying sprint performance must integrate knowledge from biomechanics, exercise physiology, nutrition, and psychology. A climate researcher needs to combine expertise from atmospheric science, oceanography, ecology, and data analysis.

Current approaches to multi-domain problems with LLMs typically involve:

-   Using a general-purpose model and providing extensive context in the
    prompt

-   Consulting multiple domain-specific models separately and manually
    integrating their outputs

-   Creating custom fine-tuned models for specific interdisciplinary
    applications

Each approach has significant limitations. General models lack depth, manual integration is time-consuming and requires human expertise, and custom models are expensive to develop and maintain.

## Existing Approaches and Their Limitations

Several approaches have emerged to address the challenge of combining
domain expertise in LLMs:

**Prompt Engineering**: Techniques such as chain-of-thought [@Wei2022] and few-shot learning [@Brown2020] can enhance domain performance, but these approaches are limited by context windows and don't fundamentally address the expertise gap.

**Retrieval-Augmented Generation (RAG)**: RAG systems [@Lewis2020] augment LLMs with external knowledge bases, but traditional implementations typically use a single retrieval system, limiting their utility.
ability to effectively integrate multiple domains.

**Model Merging**: Techniques like TIES-Merging [@Yadav2023] allow the
weights of multiple fine-tuned models to be combined, but these
approaches often result in compromised performance across domains rather
than true integration of expertise.

**Ensemble Methods**: Traditional ensemble approaches typically focus on
improving accuracy within a single domain rather than combining distinct
areas of expertise.

## Contributions of This Work

This white paper makes the following contributions:

1.  A theoretical framework for understanding and categorizing
    approaches to domain expertise integration in LLMs

2.  Five architectural patterns for combining domain-expert models, with
    detailed implementation guidance

3.  Empirical evaluation of these patterns across multiple
    interdisciplinary use cases

4.  Combine Harvester: an open-source Python framework that implements these
    patterns with a simple, extensible API

5.  A roadmap for future research and development in multi-domain AI
    systems

Our work bridges the gap between theoretical approaches to model combination and practical implementation, providing researchers and practitioners with both conceptual tools and software components to build effective multi-domain AI systems.

# Theoretical Framework

To effectively combine domain-expert LLMs, we must first establish a theoretical foundation that defines domain expertise in the context of language models and provides a framework for understanding different approaches to knowledge integration.

## Defining Domain Expertise in LLMs

Domain expertise in LLMs can be conceptualized along three dimensions:

**Knowledge Dimension**: The factual information, concepts, and
relationships specific to a domain. This includes domain-specific
terminology, established facts, theoretical frameworks, and contextual
knowledge.

**Reasoning Dimension**: The patterns of inference, analysis, and
problem-solving characteristic of a domain. This includes
domain-specific heuristics, analytical methods, and evaluation criteria.

**Communication Dimension**: The conventions, formats, and stylistic
elements typical of communication within a domain. This includes
technical writing styles, visualization conventions, and rhetorical
patterns.

A domain-expert LLM demonstrates superior performance along these dimensions within its domain of specialization. Formally, we can define a domain-expert model $M_D$ for domain $D$ as a model that satisfies:

$$P(M_D, T_D) > P(M_G, T_D)$$

Where $P(M, T)$ represents the performance of model $M$ on task set $T$,
$T_D$ is a set of tasks specific to domain $D$, and $M_G$ is a
general-purpose model with comparable overall capabilities.

## Taxonomies of Knowledge Integration

We propose a taxonomy of knowledge integration approaches based on how
and when domain expertise is combined:

**Input-Level Integration**: Combining domain expertise at the query
formulation stage, before any model processing occurs. This includes
techniques like prompt engineering and context augmentation.

**Model-Level Integration**: Combining domain expertise within the model
architecture itself. This includes ensemble methods, mixture-of-experts
architectures, and model merging techniques.

**Output-Level Integration**: Combining domain expertise after
individual models have generated responses. This includes response
synthesis, voting mechanisms, and human-in-the-loop curation.

**Temporal Integration**: Combining domain expertise sequentially, with
each expert building on the outputs of previous experts. This includes
chaining approaches and iterative refinement.

Each integration approach offers different trade-offs in terms of
performance, computational efficiency, and implementation complexity.

## Evaluation Metrics for Multi-Domain Expertise

Evaluating multi-domain expertise presents unique challenges. Traditional metrics for LLM evaluation typically focus on performance within a single domain or on general capabilities. We propose the
following metrics specifically designed for multi-domain systems:

**Cross-Domain Accuracy (CDA)**: The accuracy of responses to queries
that explicitly span multiple domains, requiring integrated knowledge.

$$\text{CDA} = \frac{1}{n} \sum_{i=1}^{n} \text{accuracy}(r_i, g_i)$$

Where $r_i$ is the model's response to cross-domain query $i$, $g_i$ is
the gold-standard response, and $n$ is the number of cross-domain
queries.

**Domain Expertise Retention (DER)**: The degree to which a multi-domain
system maintains expertise in individual domains compared to
domain-specific models.

$$\text{DER}_D = \frac{P(M_{multi}, T_D)}{P(M_D, T_D)}$$

Where $M_{multi}$ is the multi-domain system, $M_D$ is the domain-expert
model for domain $D$, and $T_D$ is a set of tasks specific to domain
$D$.

**Integration Coherence (IC)**: The logical consistency and coherence of
responses that integrate knowledge from multiple domains.

$$\text{IC} = \frac{1}{n} \sum_{i=1}^{n} \text{coherence}(r_i)$$

Where $\text{coherence}(r)$ is a function that measures the logical
consistency and flow of response $r$, typically evaluated by human
experts or specialized models.

## Formal Definitions

We now present formal definitions for the key components of our
framework:

**Domain Expert Model**: A function $M_D: Q \rightarrow R$ that maps a
query $q \in Q$ to a response $r \in R$ with expertise in domain $D$.

**Router**: A function
$\mathcal{R}: Q \times \{D_1, D_2, ..., D_n\} \rightarrow D_i$ that maps
a query $q \in Q$ and a set of available domains to the most appropriate
domain $D_i$.

**Chain**: A sequence of models $\langle M_1, M_2, ..., M_k \rangle$
with a composition function $\mathcal{C}$ such that:

$$\mathcal{C}(q, \langle M_1, M_2, ..., M_k \rangle) = M_k(...M_2(M_1(q)))$$

**Mixer**: A function
$\mathcal{M}: Q \times \{(D_1, r_1), (D_2, r_2), ..., (D_n, r_n)\} \rightarrow R$
that combines responses $r_i$ from different domains $D_i$ into a
unified response.

These formal definitions provide the foundation for the architectural
patterns presented in the following sections.

# Architectural Patterns

Based on our theoretical framework, we present five architectural patterns for combining domain-expert LLMs, each with distinct characteristics, advantages, and implementation considerations.

## Router-Based Ensembles

Router-based ensembles direct queries to the most appropriate domain expert model based on the query content. This approach is particularly effective when queries can be clearly categorized into distinct domains.

### Architecture

The core components of a router-based ensemble are:

-   **Router**: Analyzes incoming queries and determines which domain
    expert(s) should handle them

-   **Domain Experts**: Specialized models that excel in particular
    domains

-   **Dispatcher**: Sends the query to the selected expert(s) and
    returns their responses

-   **Default Handler**: Processes queries that don't clearly belong to
    any specific domain

Figure
[\[fig:router_architecture\]](#fig:router_architecture){reference-type="ref"
reference="fig:router_architecture"} illustrates this architecture:

The flow of information in this architecture is as follows:

1.  The user submits a query

2.  The router analyzes the query and determines which domain expert(s)
    should handle it

3.  The query is dispatched to the selected expert(s)

4.  The expert(s) generate responses

5.  The responses are returned to the user (either a single response or
    multiple responses, depending on the implementation)

### Routing Algorithms

The effectiveness of a router-based ensemble depends significantly on the routing algorithm. We present four approaches to routing, each with different characteristics:

**Keyword-Based Routing**: The simplest approach, using predefined
keywords or phrases associated with each domain to classify queries.

::: algorithm
::: algorithmic
scores $\gets$ empty dictionary score $\gets$ 0 score $\gets$ score +
keyword.weight scores\[domain\] $\gets$ score best_domain $\gets$ domain
with highest score in scores best_domain default_domain
:::
:::

**Embedding-Based Routing**: Uses vector embeddings to measure semantic
similarity between the query and domain descriptions.

::: algorithm
::: algorithmic
query_embedding $\gets$ EmbeddingModel(query) scores $\gets$ empty
dictionary similarity $\gets$ CosineSimilarity(query_embedding,
domain.embedding) scores\[domain\] $\gets$ similarity best_domain
$\gets$ domain with highest score in scores best_domain default_domain
:::
:::

**Classifier-Based Routing**: Uses a trained machine learning classifier
to categorize queries into domains.

::: algorithm
::: algorithmic
features $\gets$ ExtractFeatures(query) probabilities $\gets$
Classifier(features) best_domain $\gets$ domain with highest probability
best_domain default_domain
:::
:::

**LLM-Based Routing**: Uses a smaller LLM to analyze the query and
determine the most appropriate domain.

::: algorithm
::: algorithmic
prompt $\gets$ \"Analyze this query and determine which domain it
belongs to: \" + query prompt $\gets$ prompt + \"domains: \" +
JoinToString(domains) response $\gets$ RouterLLM(prompt) domain $\gets$
ExtractDomain(response) domain default_domain
:::
:::

Our empirical evaluations show that embedding-based routing provides the best balance of accuracy and computational efficiency for most applications, while LLM-based routing offers the highest accuracy for complex, ambiguous queries at the cost of increased latency and computational requirements.

### Advantages and Limitations

Router-based ensembles offer several advantages:

-   **Efficiency**: Only the relevant model(s) are invoked for each
    query

-   **Specialization**: Each query is handled by the most appropriate
    expert

-   **Scalability**: New domains can be added without modifying existing
    experts

-   **Simplicity**: Responses come directly from domain experts without
    complex integration

However, this approach also has limitations:

-   **Boundary Problems**: Queries that span multiple domains may not be
    routed optimally

-   **Router Quality**: The system's performance depends heavily on
    routing accuracy

-   **Lack of Integration**: Responses come from a single expert,
    lacking true integration of multiple domains

-   **Cold Start**: New domains require training or configuring the
    router

### Implementation Details

In the Combine Harvester framework, router-based ensembles are implemented
through the `Ensemble` class, which accepts a router implementation and
a registry of models:

``` {.python language="Python" caption="Router-Based Ensemble Implementation"}
from combineharvester import Ensemble, ModelRegistry
from combineharvester.routers import EmbeddingRouter

# Register domain expert models
registry = ModelRegistry()
registry.add_model("sprint_biomechanics", engine="ollama", 
                  model_name="sprint-expert")
registry.add_model("sports_nutrition", engine="ollama", 
                  model_name="nutrition-expert")
registry.add_model("general", engine="ollama", 
                  model_name="llama3.2")

# Create a router
router = EmbeddingRouter(threshold=0.75)

# Add domain descriptions
router.add_domain("sprint_biomechanics", 
                 "Sprint biomechanics focuses on the physics and physiology of sprint running, including ground reaction forces, muscle activation patterns, and optimal body positions for maximum velocity and acceleration.")

router.add_domain("sports_nutrition", 
                 "Sports nutrition focuses on how dietary intake affects athletic performance, including macronutrient timing, hydration strategies, and supplementation for optimal performance and recovery.")

# Create the ensemble
ensemble = Ensemble(
    router=router,
    models=registry,
    default_model="general"
)

# Generate a response
response = ensemble.generate("What's the optimal stride frequency for a 100m sprinter?")
# This would be routed to the sprint_biomechanics expert

response = ensemble.generate("How should I adjust my diet during high-intensity training blocks?")
# This would be routed to the sports_nutrition expert

response = ensemble.generate("What's the weather like today?")
# This would be routed to the default model
```

For queries that span multiple domains, the router can be configured to
return multiple experts, and their responses can be combined using a
mixer:

``` {.python language="Python" caption="Multi-Domain Routing with Mixing"}
from combineharvester.mixers import SynthesisMixer

# Create a mixer
mixer = SynthesisMixer(synthesis_model=registry.get("general"))

# Create the ensemble with mixer
ensemble = Ensemble(
    router=router,
    models=registry,
    default_model="general",
    mixer=mixer
)

# Generate a response for a cross-domain query
response = ensemble.generate(
    "How does nutrition affect recovery time from sprint training?",
    top_k=2  # Get responses from up to 2 relevant domains
)
# This would get responses from both sprint_biomechanics and sports_nutrition,
# then synthesize them using the general model
```

This implementation allows for flexible configuration of routing
strategies, domain experts, and response integration approaches.

## Sequential Chaining

Sequential chaining passes queries through multiple domain experts in sequence, with each expert building on the insights provided by previous experts. This approach is particularly effective for problems that require progressive analysis across multiple domains.

### Architecture

The core components of a sequential chain are:

-   **Domain Expert Sequence**: An ordered list of domain expert models

-   **Prompt Templates**: Templates for formatting the input to each
    expert

-   **Context Manager**: Maintains and updates the context as it flows
    through the chain

-   **Output Processor**: Extracts and formats the final response

Figure
[\[fig:chain_architecture\]](#fig:chain_architecture){reference-type="ref"
reference="fig:chain_architecture"} illustrates this architecture:

The flow of information in this architecture is as follows:

1.  The user submits a query

2.  The query is formatted using the first expert's template and sent to
    the first expert

3.  The first expert generates a response, which is added to the context

4.  The context and original query are formatted using the second
    expert's template and sent to the second expert

5.  This process continues through the chain of experts

6.  The final expert's response is returned to the user

### Prompt Engineering for Effective Chains

Effective prompt engineering is critical for sequential chains. Each
expert in the chain needs clear instructions about:

-   What information from previous experts to consider

-   What unique perspective or analysis to contribute

-   How to format the response for subsequent experts

We have identified several effective patterns for chain prompts:

**Explicit Role Definition**:

``` {caption="Explicit Role Definition Template"}
You are an expert in {domain}. 
Previous expert in {previous_domain} provided this analysis:
{prev_response}

Given this analysis and the original query:
{query}

Provide your perspective as a {domain} expert, focusing on aspects not covered in the previous analysis.
```

**Critique and Extend**:

``` {caption="Critique and Extend Template"}
Review this analysis from a {previous_domain} expert:
{prev_response}

Original query: {query}

As a {domain} expert, address any gaps or limitations in the previous analysis, then extend it with insights from your domain.
```

**Targeted Question**:

``` {caption="Targeted Question Template"}
Based on this analysis:
{prev_response}

Answer the following specific question from your expertise in {domain}:
How does {domain_specific_aspect} relate to the original query: {query}?
```

**Integration Prompt** (for final expert in chain):

``` {caption="Integration Template"}
You have received analyses from multiple experts:

{previous_domain_1} expert: {responses[0]}
{previous_domain_2} expert: {responses[1]}
...

Original query: {query}

Synthesize these perspectives into a comprehensive, integrated response that addresses the query while maintaining consistency across domains.
```

Our empirical evaluations show that explicit role definition works best for initial experts in the chain, while critique and extend prompts are more effective for middle experts, and integration prompts are essential for the final expert.

### Handling Context Windows and Information Transfer

A key challenge in sequential chaining is managing the growing context as information flows through the chain. As each expert adds their analysis, the total context can exceed the context window of subsequent models. We present several strategies for addressing this challenge:

**Summarization**: Insert a summarization step between experts to
condense previous analyses:

``` {.python language="Python" caption="Summarization in Chains"}
def chain_with_summarization(query, models, summarizer):
    context = {"query": query, "responses": []}
    
    for i, model in enumerate(models):
        # Summarize previous responses if needed
        if i > 0 and len(context["responses"][-1]) > 1000:
            summary = summarizer.generate(
                f"Summarize this analysis concisely while preserving key insights: {context['responses'][-1]}"
            )
            context["responses"][-1] = summary
        
        # Generate response from current model
        prompt = f"Previous analysis: {context['responses'][-1] if context['responses'] else 'None'}\nQuery: {query}"
        response = model.generate(prompt)
        context["responses"].append(response)
    
    return context["responses"][-1]
```

**Key Points Extraction**: Extract only the most important points from
previous analyses:

``` {.python language="Python" caption="Key Points Extraction"}
def extract_key_points(response, extractor):
    prompt = f"Extract the 3-5 most important points from this analysis as a bulleted list:\n\n{response}"
    return extractor.generate(prompt)
```

**Selective Context**: Include only relevant portions of previous
analyses based on the current expert's domain:

``` {.python language="Python" caption="Selective Context"}
def selective_context(query, previous_responses, current_domain, selector):
    prompt = f"""
    Original query: {query}
    
    Previous expert analyses:
    {previous_responses}
    
    Extract only the information relevant to {current_domain} as concise bullet points.
    """
    return selector.generate(prompt)
```

**Hierarchical Chaining**: Organize experts into groups, with a
synthesizer for each group:

Our evaluations show that hierarchical chaining with selective context provides the best balance of information preservation and context management for most applications.


### Implementation Details

In the Combine Harvester framework, sequential chaining is implemented
through the `Chain` class, which accepts a list of models and optional
prompt templates:

``` {.python language="Python" caption="Sequential Chain Implementation"}
from combineharvester import Chain, ModelRegistry

# Register domain expert models
registry = ModelRegistry()
registry.add_model("biomechanics", engine="ollama", model_name="biomechanics-expert")
registry.add_model("nutrition", engine="ollama", model_name="nutrition-expert")
registry.add_model("recovery", engine="ollama", model_name="recovery-expert")
registry.add_model("synthesizer", engine="ollama", model_name="llama3.2")

# Define prompt templates
templates = {
    "biomechanics": "You are a biomechanics expert. Analyze this query from a biomechanical perspective: {query}",
    
    "nutrition": """You are a sports nutrition expert.
    
    Biomechanical analysis: {responses[0]}
    
    Given this biomechanical context and the original query: {query}
    
    Provide nutritional insights that complement the biomechanical analysis.""",
    
    "recovery": """You are a recovery and regeneration expert.
    
    Biomechanical analysis: {responses[0]}
    Nutritional analysis: {responses[1]}
    
    Given these analyses and the original query: {query}
    
    Provide recovery strategies that address both the biomechanical and nutritional aspects.""",
    
    "synthesizer": """Synthesize the following expert analyses into a comprehensive response:
    
    Biomechanical analysis: {responses[0]}
    Nutritional analysis: {responses[1]}
    Recovery analysis: {responses[2]}
    
    Original query: {query}
    
    Provide an integrated response that combines insights from all three domains."""
}

# Create the chain
chain = Chain(
    models=[
        registry.get("biomechanics"),
        registry.get("nutrition"),
        registry.get("recovery"),
        registry.get("synthesizer")
    ],
    prompt_templates=templates
)

# Generate a response
response = chain.generate("How can I optimize my sprint training to improve both performance and recovery?")
```

For handling longer contexts, Combine Harvester provides the
`SummarizingChain` class, which automatically summarizes intermediate
responses when they exceed a specified length:

``` {.python language="Python" caption="Summarizing Chain Implementation"}
from combineharvester import SummarizingChain

# Create a summarizing chain
chain = SummarizingChain(
    models=[
        registry.get("biomechanics"),
        registry.get("nutrition"),
        registry.get("recovery"),
        registry.get("synthesizer")
    ],
    prompt_templates=templates,
    summarizer=registry.get("synthesizer"),
    max_length=2000,  # Maximum length before summarization
    summarization_prompt="Summarize this analysis concisely while preserving key insights: {response}"
)
```

This implementation allows for flexible configuration of sequential
processing through multiple domain experts, with automatic handling of
context management challenges.

## Mixture of Experts

The Mixture of Experts (MoE) pattern processes queries through multiple
domain experts in parallel and then combines their outputs based on
relevance or confidence. This approach is particularly effective for
queries that clearly span multiple domains and require integrated
insights.

### Architecture

The core components of a Mixture of Experts system are:

-   **Domain Experts**: Specialized models that excel in particular
    domains

-   **Confidence Estimator**: Determines each expert's relevance to the
    query

-   **Mixer**: Combines the experts' outputs based on their confidence
    scores

-   **Meta-Expert**: Optional component that synthesizes or refines the
    combined output

Figure
[\[fig:moe_architecture\]](#fig:moe_architecture){reference-type="ref"
reference="fig:moe_architecture"} illustrates this architecture:

The flow of information in this architecture is as follows:

1.  The user submits a query

2.  The confidence estimator determines each expert's relevance to the
    query

3.  The query is sent to all experts in parallel

4.  Each expert generates a response

5.  The mixer combines the responses based on the confidence scores

6.  (Optional) The meta-expert refines or synthesizes the combined
    response

7.  The final response is returned to the user

### Weighting Mechanisms

The effectiveness of a Mixture of Experts system depends significantly
on how expert responses are weighted. We present four approaches to
weighting:

**Binary Weighting**: Each expert is either included (weight = 1) or
excluded (weight = 0) based on a relevance threshold:

$$w_i = 
\begin{cases}
1 & \text{if } c_i \geq \theta \\
0 & \text{otherwise}
\end{cases}$$

Where $w_i$ is the weight for expert $i$, $c_i$ is the confidence score,
and $\theta$ is the relevance threshold.

**Linear Weighting**: Weights are directly proportional to confidence
scores:

$$w_i = \frac{c_i}{\sum_{j=1}^{n} c_j}$$

**Softmax Weighting**: Weights are determined using the softmax function
to emphasize higher confidence scores:

$$w_i = \frac{e^{c_i / \tau}}{\sum_{j=1}^{n} e^{c_j / \tau}}$$

Where $\tau$ is a temperature parameter that controls the sharpness of
the distribution.

**Learned Weighting**: Weights are determined by a trained model that
considers both the query and expert responses:

$$w_i = f_\theta(q, r_1, r_2, ..., r_n)_i$$

Where $f_\theta$ is a learned function with parameters $\theta$, $q$ is
the query, and $r_i$ is the response from expert $i$.

Our empirical evaluations show that softmax weighting with $\tau = 0.5$
provides the best balance of expert utilization for most applications,
while learned weighting offers the highest performance when sufficient
training data is available.

### Response Synthesis Strategies

Once expert responses are weighted, they must be combined into a
coherent final response. We present several strategies for response
synthesis:

**Weighted Concatenation**: Responses are concatenated with headers
indicating their domain and weight:

``` {caption="Weighted Concatenation Example"}
[Biomechanics (70%)]:
The optimal stride frequency for elite sprinters typically ranges between 4.5 and 5.0 strides per second...

[Physiology (30%)]:
From a physiological perspective, stride frequency is limited by the rate of muscle contraction and neural signaling...
```

**Extractive Synthesis**: Key points are extracted from each response
based on weights and combined:

::: algorithm
::: algorithmic
result $\gets$ empty string points $\gets$ ExtractKeyPoints(response,
n=round(weight \* max_points)) result $\gets$ result +
FormatPoints(domain, points) result
:::
:::

**LLM-Based Synthesis**: A meta-expert model synthesizes the weighted
responses:

``` {caption="LLM-Based Synthesis Prompt"}
You are tasked with synthesizing responses from multiple domain experts into a coherent, integrated response.

Original query: {query}

Expert responses (with confidence scores):

[Biomechanics (70%)]:
{biomechanics_response}

[Physiology (30%)]:
{physiology_response}

Create a unified response that integrates insights from all experts, giving appropriate weight to each domain based on their confidence scores. Ensure the response is coherent, non-repetitive, and directly addresses the original query.
```

**Hierarchical Synthesis**: Responses are first grouped by domain
category, synthesized within categories, and then synthesized across
categories:

Our evaluations show that LLM-based synthesis provides the highest
quality integrated responses, particularly when using a model with
strong reasoning capabilities as the meta-expert.

### Implementation Details

In the Combine Harvester framework, the Mixture of Experts pattern is
implemented through the `MixtureOfExperts` class, which accepts a
confidence estimator, a set of models, and a mixer:

``` {.python language="Python" caption="Mixture of Experts Implementation"}
from combineharvester import MixtureOfExperts, ModelRegistry
from combineharvester.confidence import EmbeddingConfidence
from combineharvester.mixers import SynthesisMixer

# Register domain expert models
registry = ModelRegistry()
registry.add_model("biomechanics", engine="ollama", model_name="biomechanics-expert")
registry.add_model("physiology", engine="ollama", model_name="physiology-expert")
registry.add_model("nutrition", engine="ollama", model_name="nutrition-expert")
registry.add_model("synthesizer", engine="ollama", model_name="llama3.2")

# Create a confidence estimator
confidence = EmbeddingConfidence(
    embedding_model=registry.get("synthesizer"),
    temperature=0.5  # Controls the sharpness of the softmax distribution
)

# Add domain descriptions
confidence.add_domain("biomechanics", 
                     "Biomechanics is the study of the mechanical laws relating to the movement of living organisms.")
confidence.add_domain("physiology", 
                     "Physiology is the study of the functions and mechanisms of the human body.")
confidence.add_domain("nutrition", 
                     "Sports nutrition focuses on the dietary needs and strategies to enhance athletic performance.")

# Create a mixer
mixer = SynthesisMixer(
    synthesis_model=registry.get("synthesizer"),
    prompt_template="""
    You are tasked with synthesizing responses from multiple domain experts into a coherent, integrated response.
    
    Original query: {query}
    
    Expert responses (with confidence scores):
    
    {weighted_responses}
    
    Create a unified response that integrates insights from all experts, giving appropriate weight to each domain based on their confidence scores. Ensure the response is coherent, non-repetitive, and directly addresses the original query.
    """
)

# Create the mixture of experts
moe = MixtureOfExperts(
    confidence_estimator=confidence,
    models=registry,
    mixer=mixer,
    threshold=0.1  # Minimum confidence score for an expert to be included
)

# Generate a response
response = moe.generate("What is the optimal stride frequency for elite sprinters and how does it relate to muscle fiber type?")
```

For applications requiring more control over the weighting mechanism,
Combine Harvester provides several confidence estimator implementations:

``` {.python language="Python" caption="Confidence Estimator Options"}
from combineharvester.confidence import KeywordConfidence, ClassifierConfidence, LLMConfidence

# Keyword-based confidence
keyword_confidence = KeywordConfidence()
keyword_confidence.add_domain("biomechanics", ["stride", "force", "velocity", "mechanics", "kinetics"])
keyword_confidence.add_domain("physiology", ["muscle", "fiber", "aerobic", "anaerobic", "fatigue"])

# Classifier-based confidence
classifier_confidence = ClassifierConfidence(
    model_path="path/to/classifier_model",
    domains=["biomechanics", "physiology", "nutrition"]
)

# LLM-based confidence
llm_confidence = LLMConfidence(
    model=registry.get("synthesizer"),
    domains=["biomechanics", "physiology", "nutrition"],
    prompt_template="""
    Analyze this query: {query}
    
    Rate the relevance of each of the following domains to this query on a scale from 0 to 10:
    - Biomechanics: [?]
    - Physiology: [?]
    - Nutrition: [?]
    
    Provide only the numerical ratings.
    """
)
```

This implementation allows for flexible configuration of the Mixture of
Experts pattern, with customizable confidence estimation, weighting
mechanisms, and response synthesis strategies.

## Specialized System Prompts

The Specialized System Prompts pattern uses a single powerful model with
carefully crafted prompts that define multiple areas of expertise. This
approach is particularly effective when computational resources are
limited or when tight integration between domains is required.

### Single-Model Multi-Expert Approach

Unlike the previous patterns that use multiple distinct models, this
approach leverages a single model's ability to adopt different expert
personas based on prompting. The core components are:

-   **Base Model**: A capable general-purpose LLM

-   **System Prompt**: A comprehensive prompt that defines multiple
    domains of expertise

-   **Domain Definitions**: Clear descriptions of each domain's
    knowledge and reasoning patterns

-   **Integration Guidelines**: Instructions for how to combine insights
    across domains

Figure
[\[fig:system_prompt_architecture\]](#fig:system_prompt_architecture){reference-type="ref"
reference="fig:system_prompt_architecture"} illustrates this
architecture:

The flow of information in this architecture is as follows:

1.  The system prompt defines multiple domains of expertise

2.  The user submits a query

3.  The model processes the query in the context of the system prompt

4.  The model internally determines which domains are relevant

5.  The model generates an integrated response that combines insights
    from relevant domains

### Prompt Engineering Techniques

Effective system prompts for multi-domain expertise require careful
engineering. We present several techniques that have proven effective:

**Explicit Domain Definition**:

``` {caption="Explicit Domain Definition"}
You are an expert in multiple domains:

1. Biomechanics: You understand the mechanical principles governing human movement, including:
   - Force production and transmission through the kinetic chain
   - Optimal joint angles and body positions for maximum performance
   - Analysis of movement efficiency and mechanical advantage

2. Exercise Physiology: You understand the physiological responses to exercise, including:
   - Energy system utilization during different types of activities
   -


\subsection{Knowledge Distillation Across Domains}

Knowledge distillation across domains involves training a single model to incorporate expertise from multiple domain-specific models. This approach is particularly effective for creating efficient, integrated models that combine insights from multiple domains without the runtime overhead of ensemble or chain approaches.

\subsubsection{Architecture}

The core components of a knowledge distillation system are:

\begin{itemize}
    \item \textbf{Teacher Models}: Domain-expert models that provide specialized knowledge
    \item \textbf{Student Model}: A model being trained to incorporate expertise from all teachers
    \item \textbf{Training Data Generator}: Creates diverse training examples spanning multiple domains
    \item \textbf{Distillation Process}: Transfers knowledge from teachers to student
\end{itemize}

Figure \ref{fig:distillation_architecture} illustrates this architecture:

\begin{figure}[h]
\centering
\begin{tikzpicture}[node distance=2cm, auto, thick]
    \node[rectangle, draw, minimum width=3cm, minimum height=1cm] (data) {Training Data Generator};
    
    \node[rectangle, draw, minimum width=2cm, minimum height=1cm, below left=1cm and -1cm of data] (teacher1) {Teacher 1};
    \node[rectangle, draw, minimum width=2cm, minimum height=1cm, below=1cm of data] (teacher2) {Teacher 2};
    \node[rectangle, draw, minimum width=2cm, minimum height=1cm, below right=1cm and -1cm of data] (teacher3) {Teacher 3};
    
    \node[rectangle, draw, minimum width=4cm, minimum height=1cm, below=3cm of data] (student) {Student Model};
    
    \node[rectangle, draw, minimum width=3cm, minimum height=1cm, below=1.5cm of student] (integrated) {Integrated Expert};
    
    \draw[->] (data) -- (teacher1);
    \draw[->] (data) -- (teacher2);
    \draw[->] (data) -- (teacher3);
    
    \draw[->] (teacher1) -- (student);
    \draw[->] (teacher2) -- (student);
    \draw[->] (teacher3) -- (student);
    
    \draw[->] (data) -- (student);
    
    \draw[->] (student) -- (integrated);
\end{tikzpicture}
\caption{Knowledge distillation architecture}
\label{fig:distillation_architecture}
\end{figure}

The process of knowledge distillation involves:

\begin{enumerate}
    \item Generating diverse training examples that span multiple domains
    \item Using domain-expert teacher models to generate responses to these examples
    \item Training a student model to replicate the combined expertise of the teachers
    \item Evaluating the student model's performance across domains
    \item Iteratively refining the distillation process to improve integration
\end{enumerate}

\subsubsection{Training Data Generation}

Effective training data generation is critical for successful knowledge distillation. The training data should cover:

\begin{itemize}
    \item Single-domain queries that test expertise in individual domains
    \item Multi-domain queries that require integrated knowledge
    \item Edge cases that test the boundaries between domains
\end{itemize}

We present several approaches to training data generation:

\textbf{Expert-Curated Data}: Domain experts manually create training examples that test specific aspects of domain knowledge and cross-domain integration.

\textbf{Synthetic Data Generation}: Using LLMs to generate diverse training examples based on domain descriptions:

\begin{lstlisting}[caption=Synthetic Data Generation Prompt]
Generate 10 diverse questions that would require expertise in the following domains to answer effectively:

Domains:
1. Biomechanics: The study of mechanical laws relating to the movement of living organisms
2. Exercise Physiology: The study of the physiological responses to physical activity
3. Sports Nutrition: The study of dietary needs and strategies to enhance athletic performance

For each question:
1. Ensure it requires integration of knowledge from at least 2 domains
2. Make it specific and detailed enough to test deep domain knowledge
3. Focus on practical applications rather than purely theoretical concepts
4. Vary the primary domain focus across the questions

Format each question as a standalone query that a coach, athlete, or sports scientist might ask.
```

**Adversarial Data Generation**: Creating examples specifically designed
to challenge the boundaries between domains:

``` {caption="Adversarial Data Generation Prompt"}
Generate 5 questions that appear to belong to one domain but actually require expertise from another domain to answer correctly.

Domains:
1. Biomechanics: The study of mechanical laws relating to the movement of living organisms
2. Exercise Physiology: The study of the physiological responses to physical activity

For each question:
1. Make it initially appear to be primarily about biomechanics
2. Ensure that a complete answer actually requires significant physiological knowledge
3. Design the question so that answering from only a biomechanical perspective would lead to an incomplete or potentially misleading response
```

**Real-World Data Collection**: Gathering actual queries from
practitioners in relevant fields, which often naturally span multiple
domains.

Our empirical evaluations show that a combination of synthetic and
adversarial data generation, followed by expert review, provides the
most effective training data for knowledge distillation.

### Fine-Tuning Strategies

Once training data is generated, several fine-tuning strategies can be
employed to effectively distill knowledge from teacher models to the
student model:

**Sequential Fine-Tuning**: Fine-tune the student model on each domain
sequentially, starting with the most general domain and progressing to
more specialized ones:

::: algorithm
::: algorithmic
Sort domains from most general to most specialized training_data $\gets$
GenerateTrainingData(domain) teacher $\gets$ teacher_models\[domain\]
teacher_responses $\gets$ teacher.generate_batch(training_data)
student_model $\gets$ FineTune(student_model, training_data,
teacher_responses) student_model
:::
:::

**Multi-Task Fine-Tuning**: Fine-tune the student model on all domains
simultaneously, with examples from different domains mixed in each
batch:

::: algorithm
::: algorithmic
all_training_data $\gets$ empty list all_teacher_responses $\gets$ empty
list domain_data $\gets$ GenerateTrainingData(domain) teacher $\gets$
teacher_models\[domain\] teacher_responses $\gets$
teacher.generate_batch(domain_data)
all_training_data.extend(domain_data)
all_teacher_responses.extend(teacher_responses)
Shuffle(all_training_data, all_teacher_responses) student_model $\gets$
FineTune(student_model, all_training_data, all_teacher_responses)
student_model
:::
:::

**Integration-Focused Fine-Tuning**: A two-phase approach that first
fine-tunes on individual domains, then specifically on cross-domain
integration:

::: algorithm
::: algorithmic
[// Phase 1: Domain-specific fine-tuning]{style="color: gray"}
domain_data $\gets$ GenerateTrainingData(domain, cross_domain=False)
teacher $\gets$ teacher_models\[domain\] teacher_responses $\gets$
teacher.generate_batch(domain_data) student_model $\gets$
FineTune(student_model, domain_data, teacher_responses) [// Phase 2:
Cross-domain integration fine-tuning]{style="color: gray"}
integration_data $\gets$ GenerateTrainingData(domains,
cross_domain=True) integration_responses $\gets$ empty list
domain_responses $\gets$ empty dict teacher $\gets$
teacher_models\[domain\] domain_responses\[domain\] $\gets$
teacher.generate(query) integrated_response $\gets$
IntegrateResponses(query, domain_responses)
integration_responses.append(integrated_response) student_model $\gets$
FineTune(student_model, integration_data, integration_responses)
student_model
:::
:::

Our evaluations show that integration-focused fine-tuning provides the
best balance of domain-specific expertise and cross-domain integration
capabilities.

### Implementation Details

In the Combine Harvester framework, knowledge distillation is implemented
through the `Distiller` class, which manages the distillation process:

``` {.python language="Python" caption="Knowledge Distillation Implementation"}
from combineharvester import Distiller, ModelRegistry
from combineharvester.data import SyntheticDataGenerator, AdversarialDataGenerator
from combineharvester.trainers import IntegrationTrainer

# Register teacher models
registry = ModelRegistry()
registry.add_model("biomechanics_teacher", engine="ollama", model_name="biomechanics-expert")
registry.add_model("physiology_teacher", engine="ollama", model_name="physiology-expert")
registry.add_model("nutrition_teacher", engine="ollama", model_name="nutrition-expert")

# Register student model (starting point)
registry.add_model("student", engine="huggingface", model_name="meta-llama/Llama-2-7b-hf")

# Create data generators
synthetic_generator = SyntheticDataGenerator(
    generation_model=registry.get("biomechanics_teacher"),
    domains={
        "biomechanics": "The study of mechanical laws relating to the movement of living organisms",
        "physiology": "The study of the physiological responses to physical activity",
        "nutrition": "The study of dietary needs and strategies to enhance athletic performance"
    },
    num_examples=1000,
    cross_domain_ratio=0.7
)

adversarial_generator = AdversarialDataGenerator(
    generation_model=registry.get("biomechanics_teacher"),
    domains={
        "biomechanics": "The study of mechanical laws relating to the movement of living organisms",
        "physiology": "The study of the physiological responses to physical activity",
        "nutrition": "The study of dietary needs and strategies to enhance athletic performance"
    },
    num_examples=300
)

# Create the distiller
distiller = Distiller(
    student_model=registry.get("student"),
    teacher_models={
        "biomechanics": registry.get("biomechanics_teacher"),
        "physiology": registry.get("physiology_teacher"),
        "nutrition": registry.get("nutrition_teacher")
    },
    data_generators=[synthetic_generator, adversarial_generator],
    trainer=IntegrationTrainer(
        learning_rate=2e-5,
        epochs=3,
        batch_size=16,
        evaluation_steps=500
    ),
    output_dir="./distilled_model"
)

# Run the distillation process
distilled_model = distiller.distill()

# Save the distilled model
distilled_model.save("./distilled_model")
```

For more control over the integration of teacher responses, Combine Harvester
provides the `ResponseIntegrator` class:

``` {.python language="Python" caption="Response Integration"}
from combineharvester.integration import LLMResponseIntegrator

# Create a response integrator
integrator = LLMResponseIntegrator(
    integration_model=registry.get("biomechanics_teacher"),
    prompt_template="""
    You need to create an integrated response that combines insights from multiple domain experts.
    
    Original query: {query}
    
    Expert responses:
    
    [Biomechanics Expert]
    {biomechanics}
    
    [Physiology Expert]
    {physiology}
    
    [Nutrition Expert]
    {nutrition}
    
    Create a unified response that integrates all relevant insights from these experts, resolving any contradictions and creating a coherent narrative.
    """
)

# Integrate responses
integrated_response = integrator.integrate(
    query="How can sprint athletes optimize their recovery between races?",
    responses={
        "biomechanics": "From a biomechanical perspective...",
        "physiology": "The physiological aspects of recovery include...",
        "nutrition": "Nutritional strategies for inter-race recovery include..."
    }
)
```

This implementation allows for flexible configuration of the knowledge
distillation process, with customizable data generation, fine-tuning
strategies, and response integration approaches.

## Retrieval-Augmented Generation with Multiple Knowledge Bases

Retrieval-Augmented Generation (RAG) with multiple knowledge bases
extends traditional RAG by incorporating domain-specific knowledge bases
and retrieval strategies. This approach is particularly effective when
domain expertise requires access to specialized information that may not
be fully captured in the model's parameters.

### Architecture

The core components of a multi-domain RAG system are:

-   **Domain-Specific Knowledge Bases**: Collections of documents or
    chunks relevant to each domain

-   **Domain Routers**: Components that direct queries to appropriate
    knowledge bases

-   **Specialized Retrievers**: Retrieval mechanisms optimized for each
    domain

-   **Context Integrator**: Combines retrieved information from multiple
    domains

-   **Generation Model**: Produces responses based on the integrated
    context

Figure
[\[fig:rag_architecture\]](#fig:rag_architecture){reference-type="ref"
reference="fig:rag_architecture"} illustrates this architecture:

The flow of information in this architecture is as follows:

1.  The user submits a query

2.  The domain router determines which knowledge bases are relevant

3.  The query is sent to the relevant knowledge bases

4.  Each knowledge base returns relevant documents or chunks

5.  The context integrator combines the retrieved information

6.  The generation model produces a response based on the query and
    integrated context

### Vector Database Management

Effective management of multiple vector databases is critical for
multi-domain RAG systems. We present several approaches to organizing
and querying domain-specific knowledge:

**Separate Databases**: Maintain completely separate vector databases
for each domain:

``` {.python language="Python" caption="Separate Databases Approach"}
# Create separate vector databases
biomechanics_db = VectorDatabase(embedding_model=embedding_model)
physiology_db = VectorDatabase(embedding_model=embedding_model)
nutrition_db = VectorDatabase(embedding_model=embedding_model)

# Add documents to appropriate databases
biomechanics_db.add_documents(biomechanics_documents)
physiology_db.add_documents(physiology_documents)
nutrition_db.add_documents(nutrition_documents)

# Query each database separately
biomechanics_results = biomechanics_db.query(query, top_k=3)
physiology_results = physiology_db.query(query, top_k=3)
nutrition_results = nutrition_db.query(query, top_k=3)

# Combine results
all_results = biomechanics_results + physiology_results + nutrition_results
```

**Single Database with Domain Tags**: Maintain a single database with
domain tags for each document:

``` {.python language="Python" caption="Single Database with Domain Tags"}
# Create a single vector database
unified_db = VectorDatabase(embedding_model=embedding_model)

# Add documents with domain tags
for doc in biomechanics_documents:
    doc.metadata["domain"] = "biomechanics"
    unified_db.add_document(doc)

for doc in physiology_documents:
    doc.metadata["domain"] = "physiology"
    unified_db.add_document(doc)

for doc in nutrition_documents:
    doc.metadata["domain"] = "nutrition"
    unified_db.add_document(doc)

# Query with domain filtering
relevant_domains = domain_router.route(query)
results = unified_db.query(
    query, 
    top_k=10, 
    filter={"domain": {"$in": relevant_domains}}
)
```

**Hybrid Approach**: Use domain-specific embedding models with a unified
storage system:

``` {.python language="Python" caption="Hybrid Approach"}
# Create domain-specific embedding models
biomechanics_embedder = DomainSpecificEmbedder("biomechanics")
physiology_embedder = DomainSpecificEmbedder("physiology")
nutrition_embedder = DomainSpecificEmbedder("nutrition")

# Create a unified database
unified_db = VectorDatabase()

# Add documents with domain-specific embeddings
for doc in biomechanics_documents:
    doc.metadata["domain"] = "biomechanics"
    doc.embedding = biomechanics_embedder.embed(doc.text)
    unified_db.add_document(doc)

for doc in physiology_documents:
    doc.metadata["domain"] = "physiology"
    doc.embedding = physiology_embedder.embed(doc.text)
    unified_db.add_document(doc)

for doc in nutrition_documents:
    doc.metadata["domain"] = "nutrition"
    doc.embedding = nutrition_embedder.embed(doc.text)
    unified_db.add_document(doc)

# Query using domain-specific embeddings
relevant_domains = domain_router.route(query)
results = []

for domain in relevant_domains:
    domain_embedder = get_domain_embedder(domain)
    query_embedding = domain_embedder.embed(query)
    domain_results = unified_db.query_by_embedding(
        query_embedding,
        top_k=5,
        filter={"domain": domain}
    )
    results.extend(domain_results)
```

Our empirical evaluations show that the hybrid approach provides the
best retrieval performance, particularly for queries that span multiple
domains, but at the cost of increased computational complexity and
storage requirements.

### Implementation Details

In the Combine Harvester framework, multi-domain RAG is implemented through
the `MultiDomainRAG` class, which manages multiple knowledge bases and
retrieval strategies:

``` {.python language="Python" caption="Multi-Domain RAG Implementation"}
from combineharvester import MultiDomainRAG, ModelRegistry
from combineharvester.rag import VectorStore, DomainRouter
from combineharvester.rag.retrievers import HybridRetriever

# Register models
registry = ModelRegistry()
registry.add_model("embedding_model", engine="openai", model_name="text-embedding-3-small")
registry.add_model("generation_model", engine="anthropic", model_name="claude-3-sonnet")

# Create vector stores for each domain
biomechanics_store = VectorStore(
    embedding_model=registry.get("embedding_model"),
    name="biomechanics"
)

physiology_store = VectorStore(
    embedding_model=registry.get("embedding_model"),
    name="physiology"
)

nutrition_store = VectorStore(
    embedding_model=registry.get("embedding_model"),
    name="nutrition"
)

# Load documents into vector stores
biomechanics_store.add_documents("path/to/biomechanics_docs", chunk_size=500)
physiology_store.add_documents("path/to/physiology_docs", chunk_size=500)
nutrition_store.add_documents("path/to/nutrition_docs", chunk_size=500)

# Create the multi-domain RAG system
rag = MultiDomainRAG(
    embedding_model=registry.get("embedding_model"),
    generation_model=registry.get("generation_model"),
    vector_stores=[biomechanics_store, physiology_store, nutrition_store],
    domain_router=DomainRouter(
        embedding_model=registry.get("embedding_model"),
        domains=["biomechanics", "physiology", "nutrition"]
    ),
    retriever=HybridRetriever(
        embedding_model=registry.get("embedding_model"),
        vector_stores=[biomechanics_store, physiology_store, nutrition_store]
    )
)

# Generate a response
response = rag.generate("What is the optimal stride frequency for elite sprinters and how does it relate to muscle fiber type?")
```

This implementation allows for flexible configuration of multi-domain
RAG systems, with customizable embedding models, generation models,
vector stores, domain routers, and retrievers.

## The Combine Harvester Framework

The Combine Harvester framework is an open-source Python package that implements the architectural patterns described in this paper. It provides a flexible, extensible API for combining domain-expert LLMs in various configurations.

\subsection{System Architecture}

Combine Harvester follows a modular architecture organized around five core concepts:

\begin{enumerate}
    \item \textbf{Models}: Abstractions for interacting with various LLM providers
    \item \textbf{Routers}: Components that direct queries to appropriate domain experts
    \item \textbf{Chains}: Mechanisms for sequential processing through multiple models
    \item \textbf{Mixers}: Components that combine responses from multiple models
    \item \textbf{Registry}: A central repository for managing model instances
\end{enumerate}

Figure \ref{fig:system_architecture} illustrates the high-level architecture:

\begin{figure}[h]
\centering
\begin{tikzpicture}[node distance=2cm, auto, thick]
    \node[rectangle, draw, minimum width=4cm, minimum height=1cm] (api) {Combine Harvester API};
    
    \node[rectangle, draw, minimum width=2.5cm, minimum height=1cm, below left=1cm and -1cm of api] (models) {Models};
    \node[rectangle, draw, minimum width=2.5cm, minimum height=1cm, below left=1cm and 2cm of api] (routers) {Routers};
    \node[rectangle, draw, minimum width=2.5cm, minimum height=1cm, below right=1cm and -1cm of api] (chains) {Chains};
    \node[rectangle, draw, minimum width=2.5cm, minimum height=1cm, below right=1cm and 2cm of api] (mixers) {Mixers};
    
    \node[rectangle, draw, minimum width=4cm, minimum height=1cm, below=3cm of api] (registry) {Model Registry};
    
    \node[rectangle, draw, minimum width=2cm, minimum height=0.8cm, below left=1cm and -1.5cm of registry] (ollama) {Ollama};
    \node[rectangle, draw, minimum width=2cm, minimum height=0.8cm, below=1cm of registry] (openai) {OpenAI};
    \node[rectangle, draw, minimum width=2cm, minimum height=0.8cm, below right=1cm and -1.5cm of registry] (anthropic) {Anthropic};
    
    \draw[->] (api) -- (models);
    \draw[->] (api) -- (routers);
    \draw[->] (api) -- (chains);
    \draw[->] (api) -- (mixers);
    
    \draw[->] (models) -- (registry);
    \draw[->] (chains) -- (registry);
    \draw[->] (mixers) -- (registry);
    
    \draw[->] (registry) -- (ollama);
    \draw[->] (registry) -- (openai);
    \draw[->] (registry) -- (anthropic);
\end{tikzpicture}
\caption{Combine Harvester system architecture}
\label{fig:system_architecture}
\end{figure}

This modular design allows components to be developed, tested, and extended independently, while providing a consistent interface for users.

\subsection{Core Components}

\subsubsection{Models}

The \texttt{Model} interface defines the contract for all LLM implementations:

\begin{lstlisting}[language=Python, caption=Model Interface]
class Model(ABC):
    """Abstract base class for all models."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the model."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response to the given prompt."""
        pass
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embeddings for the given text."""
        pass
```

Combine Harvester provides implementations for popular LLM providers:

-   `OllamaModel`: For local models running with Ollama

-   `OpenAIModel`: For OpenAI's GPT models

-   `AnthropicModel`: For Anthropic's Claude models

-   `HuggingFaceModel`: For models hosted on HuggingFace

### Routers

Routers determine which domain expert should handle a given query:

``` {.python language="Python" caption="Router Interface"}
class Router(ABC):
    """Abstract base class for all routers."""
    
    @abstractmethod
    def route(self, query: str, available_models: List[str]) -> Optional[str]:
        """Route a query to the most appropriate model."""
        pass
    
    @abstractmethod
    def route_multiple(self, query: str, available_models: List[str], k: int) -> List[str]:
        """Route a query to the k most appropriate models."""
        pass
```

Combine Harvester provides several router implementations:

-   `KeywordRouter`: Routes based on keyword matching

-   `EmbeddingRouter`: Routes based on embedding similarity

-   `ClassifierRouter`: Routes based on a trained classifier

-   `LLMRouter`: Uses a smaller LLM to determine routing

### Chains

Chains enable sequential processing through multiple models:

``` {.python language="Python" caption="Chain Implementation"}
class Chain:
    """Chain multiple models sequentially."""
    
    def __init__(self, models: List[Model], prompt_templates: Optional[Dict[str, str]] = None):
        """Initialize the chain."""
        self.models = models
        self.prompt_templates = prompt_templates or {}
    
    def generate(self, query: str) -> str:
        """Generate a response by chaining models."""
        context = {"query": query, "responses": []}
        
        for i, model in enumerate(self.models):
            # Format prompt using templates and context
            if model.name in self.prompt_templates:
                template = self.prompt_templates[model.name]
                prompt = template.format(
                    query=query,
                    prev_response=context["responses"][-1] if context["responses"] else "",
                    responses=context["responses"],
                    **context
                )
            else:
                prompt = query if i == 0 else f"Previous: {context['responses'][-1]}\nQuery: {query}"
            
            # Generate response
            response = model.generate(prompt)
            context["responses"].append(response)
            context[f"response_{i}"] = response
            context[model.name] = response
        
        return context["responses"][-1]
```

### Mixers

Mixers combine responses from multiple models:

``` {.python language="Python" caption="Mixer Interface"}
class Mixer(ABC):
    """Abstract base class for all mixers."""
    
    @abstractmethod
    def mix(self, query: str, responses: Dict[str, str]) -> str:
        """Mix multiple responses into a single response."""
        pass
```

Combine Harvester provides several mixer implementations:

-   `DefaultMixer`: Returns the response with highest confidence

-   `ConcatenationMixer`: Concatenates responses with domain labels

-   `SynthesisMixer`: Uses an LLM to synthesize multiple responses

-   `WeightedMixer`: Combines responses based on confidence scores

### Registry

The registry manages model instances and provides a central point of
access:

``` {.python language="Python" caption="Model Registry"}
class ModelRegistry:
    """Registry for managing multiple models."""
    
    def __init__(self):
        """Initialize the registry."""
        self.models = {}
    
    def add_model(self, name: str, engine: str, model_name: str, **kwargs) -> None:
        """Register a new model."""
        if engine == "ollama":
            from combineharvester.models.ollama import OllamaModel
            self.models[name] = OllamaModel(model_name=model_name, **kwargs)
        elif engine == "openai":
            from combineharvester.models.openai import OpenAIModel
            self.models[name] = OpenAIModel(model_name=model_name, **kwargs)
        elif engine == "anthropic":
            from combineharvester.models.anthropic import AnthropicModel
            self.models[name] = AnthropicModel(model_name=model_name, **kwargs)
        elif engine == "huggingface":
            from combineharvester.models.huggingface import HuggingFaceModel
            self.models[name] = HuggingFaceModel(model_name=model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported engine: {engine}")
    
    def get(self, name: str) -> Model:
        """Retrieve a registered model by name."""
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found in registry")
        return self.models[name]
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.models.keys())
```

## Extension Points

Combine Harvester is designed to be extensible, with clear extension points
for customization:

-   **Custom Models**: Implement the `Model` interface to add support
    for new LLM providers

-   **Custom Routers**: Implement the `Router` interface to create new
    routing strategies

-   **Custom Mixers**: Implement the `Mixer` interface to create new
    response combination strategies

-   **Prompt Templates**: Customize the prompt templates used in chains
    and mixers

-   **Evaluation Metrics**: Add custom metrics for evaluating
    multi-domain performance

## API Design

Combine Harvester provides a simple, intuitive API for common use cases while
allowing advanced customization when needed:

# Empirical Evaluation

To evaluate the effectiveness of the architectural patterns described in
this paper, we conducted a comprehensive empirical evaluation across
multiple domains and use cases.

## Evaluation Methodology

Our evaluation methodology was designed to assess both domain-specific
expertise and cross-domain integration capabilities:

### Evaluation Datasets

We created three evaluation datasets:

-   **Domain-Specific Dataset**: 300 queries evenly distributed across
    three domains (biomechanics, exercise physiology, and sports
    nutrition), designed to test depth of expertise within each domain.

-   **Cross-Domain Dataset**: 200 queries that explicitly span multiple
    domains, requiring integrated knowledge.

-   **Real-World Dataset**: 100 queries collected from sports coaches,
    athletes, and researchers, representing authentic multi-domain
    questions.

### Baseline Models

We established the following baseline models for comparison:

-   **General-Purpose Model**: Claude 3 Opus, representing
    state-of-the-art general-purpose LLMs.

-   **Domain-Specific Models**: Three models fine-tuned on
    domain-specific datasets (biomechanics, exercise physiology, and
    sports nutrition).

-   **Simple Ensemble**: A basic ensemble that routes queries to
    domain-specific models based on keyword matching.

### Evaluation Metrics

We evaluated performance using the following metrics:

-   **Domain-Specific Accuracy (DSA)**: Accuracy on domain-specific
    queries, assessed by domain experts.

-   **Cross-Domain Accuracy (CDA)**: Accuracy on cross-domain queries,
    assessed by a panel of experts from multiple domains.

-   **Domain Expertise Retention (DER)**: The degree to which
    multi-domain systems maintain expertise in individual domains
    compared to domain-specific models.

-   **Integration Coherence (IC)**: The logical consistency and
    coherence of responses that integrate knowledge from multiple
    domains.

-   **Response Quality (RQ)**: A holistic assessment of response
    quality, including factual accuracy, completeness, and relevance.

## Results

### Domain-Specific Performance

Figure
[\[fig:domain_specific_performance\]](#fig:domain_specific_performance){reference-type="ref"
reference="fig:domain_specific_performance"} shows the performance of
different approaches on domain-specific queries:

As expected, domain-specific models achieved the highest accuracy on
domain-specific queries. However, router-based ensembles and specialized
system prompts performed nearly as well, demonstrating that these
approaches effectively preserve domain expertise.

### Cross-Domain Performance

Figure
[\[fig:cross_domain_performance\]](#fig:cross_domain_performance){reference-type="ref"
reference="fig:cross_domain_performance"} shows the performance of
different approaches on cross-domain queries:

On cross-domain queries, the Mixture of Experts approach consistently
outperformed other approaches, followed by Sequential Chaining and
Specialized System Prompts. This demonstrates the effectiveness of these
architectural patterns for integrating knowledge across domains.

### Integration Coherence

Figure
[\[fig:integration_coherence\]](#fig:integration_coherence){reference-type="ref"
reference="fig:integration_coherence"} shows the integration coherence
scores for different approaches:

The Mixture of Experts and Sequential Chaining approaches achieved the
highest integration coherence scores, indicating that these approaches
are particularly effective at producing logically consistent responses
that integrate knowledge from multiple domains.

### Real-World Performance

Figure
[\[fig:real_world_performance\]](#fig:real_world_performance){reference-type="ref"
reference="fig:real_world_performance"} shows the performance of
different approaches on the real-world dataset:

On real-world queries, the Mixture of Experts approach achieved the
highest response quality scores, followed by Sequential Chaining and
Specialized System Prompts. This demonstrates that these approaches are
effective for addressing authentic multi-domain questions.

## Analysis and Discussion

Our empirical evaluation reveals several key insights:

### Pattern Selection Guidelines

Based on our results, we can provide guidelines for selecting the most
appropriate architectural pattern for different use cases:

-   **Router-Based Ensembles**: Best when queries can be clearly
    categorized into distinct domains and when computational efficiency
    is important.

-   **Sequential Chaining**: Best for problems that require progressive
    analysis across multiple domains, particularly when there's a
    natural sequence to the analysis.

-   **Mixture of Experts**: Best for queries that clearly span multiple
    domains and require integrated insights, particularly when
    high-quality integration is critical.

-   **Specialized System Prompts**: Best when computational resources
    are limited or when tight integration between domains is required.

-   **Knowledge Distillation**: Best for production environments where
    inference speed is critical and the domains of interest are
    relatively stable.

### Trade-offs

Each architectural pattern involves trade-offs:

::: {#tab:tradeoffs}
  **Pattern**               **Domain Expertise**   **Integration**   **Compute**   **Implementation**   **Adaptability**
  ------------------------ ---------------------- ----------------- ------------- -------------------- ------------------
  Router-Based                      High                 Low           Medium             Easy                High
  Sequential                       Medium               High            High             Medium              Medium
  Mixture of Experts                High                High            High             Medium               High
  System Prompts                   Medium              Medium            Low              Easy                High
  Knowledge Distillation           Medium              Medium            Low              Hard                Low

  : Trade-offs between architectural patterns
:::

### Hybrid Approaches

Our evaluation also revealed that hybrid approaches combining multiple
patterns can be particularly effective:

-   **Router + Chain**: Using a router to select the most appropriate
    chain of domain experts for a given query.

-   **MoE + System Prompts**: Using specialized system prompts within a
    mixture of experts architecture to improve integration.

-   **Distillation from MoE**: Distilling knowledge from a mixture of
    experts into a single model for production deployment.

These hybrid approaches often provide the best balance of domain
expertise, integration quality, and computational efficiency.

# Conclusion and Future Work

## Summary of Contributions

This white paper has presented Combine Harvester, a comprehensive framework
for combining domain-expert LLMs to address complex, interdisciplinary
problems. Our key contributions include:

1.  A theoretical framework for understanding domain expertise in LLMs
    and approaches to knowledge integration

2.  Five architectural patterns for combining domain-expert models, each
    with detailed implementation guidance

3.  Empirical evaluation demonstrating the effectiveness of these
    patterns across multiple use cases

4.  An open-source Python implementation that makes these techniques
    accessible to researchers and practitioners

Our results demonstrate that carefully designed combinations of
domain-expert models can significantly outperform both general-purpose
models and individual domain experts on cross-domain tasks. The
Combine Harvester framework provides a practical way to leverage these
approaches in real-world applications.

## Limitations

Despite the promising results, several limitations should be
acknowledged:

-   **Computational Overhead**: Many of the architectural patterns,
    particularly Mixture of Experts and Sequential Chaining, involve
    significant computational overhead compared to single-model
    approaches.

-   **Integration Challenges**: Effectively integrating knowledge across
    domains remains challenging, particularly for domains with different
    epistemological frameworks or terminology.

-   **Evaluation Complexity**: Evaluating multi-domain systems requires
    expertise across all relevant domains, making comprehensive
    evaluation difficult and resource-intensive.

-   **Domain Boundaries**: The definition of domain boundaries is
    somewhat arbitrary and may not reflect the true structure of
    knowledge.

-   **Scaling Challenges**: As the number of domains increases, the
    complexity of integration grows exponentially, potentially limiting
    scalability.

## Future Research Directions

Based on our findings and the limitations identified, we propose several
promising directions for future research:

### Adaptive Integration

Current integration approaches use fixed strategies regardless of the
query or domains involved. Future work could explore adaptive
integration strategies that dynamically adjust based on the specific
domains and the nature of the query.

### Meta-Learning for Domain Integration

Meta-learning approaches could be developed to learn how to effectively
combine domain expertise based on patterns observed across many
cross-domain problems, potentially leading to more efficient and
effective integration.

### Domain-Aware Attention Mechanisms

Specialized attention mechanisms could be developed that are aware of
domain boundaries and can more effectively integrate information across
domains within a single model architecture.

### Continuous Domain Learning

Systems that can continuously learn and update their domain expertise as
new information becomes available, without catastrophic forgetting of
existing knowledge or integration capabilities.

### Domain-Specific Reasoning Patterns

Research into how different domains employ distinct reasoning patterns
and how these can be effectively combined in multi-domain systems.

### Human-AI Collaboration in Multi-Domain Problems

Exploration of how human experts can most effectively collaborate with
multi-domain AI systems, leveraging the strengths of both human and
artificial intelligence.

## Broader Impacts

The development of effective multi-domain AI systems has significant
implications for various fields:

-   **Scientific Research**: Accelerating interdisciplinary research by
    helping researchers bridge knowledge gaps across domains.

-   **Education**: Supporting interdisciplinary education by providing
    integrated explanations that connect concepts across traditional
    subject boundaries.

-   **Healthcare**: Enabling more holistic approaches to healthcare that
    consider medical, psychological, social, and environmental factors.

-   **Policy Making**: Supporting evidence-based policy decisions that
    require integration of insights from economics, sociology,
    environmental science, and other domains.

-   **Complex Problem Solving**: Enhancing our ability to address
    complex global challenges that span multiple domains, such as
    climate change, public health crises, and sustainable development.

As AI systems become increasingly capable of integrating knowledge
across domains, they have the potential to help address the
fragmentation of knowledge that has resulted from increasing
specialization, potentially leading to more holistic and effective
approaches to complex problems.

## Final Thoughts

The integration of domain expertise in AI systems mirrors a fundamental
challenge in human knowledge: how to maintain deep expertise in specific
domains while effectively combining insights across domains to address
complex problems. The architectural patterns and implementation
strategies presented in this paper represent early steps toward AI
systems capable of true interdisciplinary reasoning.

As these approaches continue to develop, we envision AI systems that can
serve not just as domain experts but as interdisciplinary collaborators,
helping humans navigate the increasingly complex landscape of
specialized knowledge and supporting more integrated approaches to the
challenges we face.

# Appendix A: Detailed API Reference

# Appendix B: Benchmark Details

# Appendix C: Example Implementations

</div>
