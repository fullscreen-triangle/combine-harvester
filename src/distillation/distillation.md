# DomainFusion Distillation Module

The distillation module provides tools for knowledge distillation across domains, allowing a single model to incorporate expertise from multiple domain-specific models. This approach is particularly effective for creating efficient, integrated models that combine insights from multiple domains without the runtime overhead of ensemble or chain approaches.

## Overview

This module implements the knowledge distillation approach described in the whitepaper, providing:

1. **Data Generation** - Tools for creating training examples that span multiple domains
2. **Training** - Methods for fine-tuning student models from teacher models
3. **Evaluation** - Metrics for assessing the performance of distilled models
4. **Response Integration** - Techniques for combining responses from multiple domains

## Files and Components

### distiller.py

Contains the core components for knowledge distillation:

- `Distiller` - Main class that coordinates the distillation process
  - Generate training data from multiple domains
  - Collect responses from teacher models
  - Integrate responses from different domains
  - Train the student model
  - Evaluate the student model's performance

- `ResponseIntegrator` - Class for combining responses from multiple domain experts
  - Integrates responses using either a simple merging strategy or an LLM-based approach
  - Preserves the contributions from each domain
  - Resolves conflicts between domains

### data_generation.py

Provides tools for generating training data:

- `DataGenerator` - Abstract base class for data generators
  - Defines the interface for generating training examples
  
- `SyntheticDataGenerator` - Creates synthetic training data using an LLM
  - Generates diverse questions that span multiple domains
  - Controls the ratio of single-domain to cross-domain examples
  - Uses domain descriptions to guide generation
  
- `AdversarialDataGenerator` - Creates adversarial examples that challenge domain boundaries
  - Makes questions that appear to belong to one domain but require expertise from another
  - Helps identify edge cases between domains
  
- `CuratedDataGenerator` - Uses pre-defined expert-curated examples
  - Allows for high-quality, manually verified training data
  - Useful for domains requiring specialized knowledge

### trainer.py

Implements trainers for fine-tuning student models:

- `Trainer` - Abstract base class for trainers
  - Defines the interface for training student models
  
- `IntegrationTrainer` - Two-phase trainer that first fine-tunes on individual domains, then on cross-domain integration
  - Preserves domain-specific expertise
  - Explicitly focuses on cross-domain integration
  - Most effective for balanced domain expertise and integration
  
- `MultiTaskTrainer` - Trains on all domains simultaneously
  - Mixes examples from different domains in each batch
  - Efficient training approach
  - Good for domains with similar complexity
  
- `SequentialTrainer` - Trains on each domain sequentially
  - Starts with general domains and progresses to specialized ones
  - Useful when domains have a clear hierarchy
  - Prevents catastrophic forgetting

### evaluator.py

Provides tools for evaluating distilled models:

- `Evaluator` - Abstract base class for evaluators
  - Defines the interface for evaluating student models
  
- `DomainRetentionEvaluator` - Evaluates how well the student model retains expertise from each teacher domain
  - Calculates Domain Expertise Retention (DER) for individual domains
  - Measures Cross-Domain Accuracy (CDA) for integration
  - Assesses Integration Coherence (IC) for response quality
  
- `BenchmarkEvaluator` - Evaluates the student model on standard benchmarks for each domain
  - Uses domain-specific benchmarks
  - Compares against teacher models and baselines
  - Provides detailed performance metrics

## Usage Examples

### Basic Distillation

```python
from domainfusion.distillation import Distiller, SyntheticDataGenerator, IntegrationTrainer
from domainfusion.models import LlamaModel

# Create teacher models
biomechanics_teacher = LlamaModel(model_path="path/to/biomechanics-expert")
physiology_teacher = LlamaModel(model_path="path/to/physiology-expert")
nutrition_teacher = LlamaModel(model_path="path/to/nutrition-expert")

# Create a student model (starting point)
student_model = LlamaModel(model_path="path/to/base-model")

# Create a data generator
data_generator = SyntheticDataGenerator(
    generation_model=biomechanics_teacher,
    domains={
        "biomechanics": "The study of mechanical laws relating to the movement of living organisms",
        "physiology": "The study of the physiological responses to physical activity",
        "nutrition": "The study of dietary needs and strategies to enhance athletic performance"
    },
    num_examples=500,
    cross_domain_ratio=0.7
)

# Create a trainer
trainer = IntegrationTrainer(
    learning_rate=2e-5,
    epochs=3,
    batch_size=16,
    evaluation_steps=500
)

# Create a distiller
distiller = Distiller(
    student_model=student_model,
    teacher_models={
        "biomechanics": biomechanics_teacher,
        "physiology": physiology_teacher,
        "nutrition": nutrition_teacher
    },
    data_generators=[data_generator],
    trainer=trainer
)

# Run the distillation process
distilled_model = distiller.distill()

# Save the distilled model
distiller.save("./distilled_model")
```

### Using Multiple Data Generators

```python
from domainfusion.distillation import (
    Distiller,
    SyntheticDataGenerator,
    AdversarialDataGenerator,
    CuratedDataGenerator
)

# Create different types of data generators
synthetic_generator = SyntheticDataGenerator(...)
adversarial_generator = AdversarialDataGenerator(...)

# Create a distiller with multiple data generators
distiller = Distiller(
    student_model=student_model,
    teacher_models=teacher_models,
    data_generators=[synthetic_generator, adversarial_generator],
    trainer=trainer
)

# Run the distillation process
distilled_model = distiller.distill()
```

### Evaluating a Distilled Model

```python
from domainfusion.distillation import DomainRetentionEvaluator, BenchmarkEvaluator

# Create evaluators
retention_evaluator = DomainRetentionEvaluator()
benchmark_evaluator = BenchmarkEvaluator(benchmarks={
    "biomechanics": biomechanics_benchmarks,
    "physiology": physiology_benchmarks,
    "nutrition": nutrition_benchmarks
})

# Create a distiller with evaluation
distiller = Distiller(
    student_model=student_model,
    teacher_models=teacher_models,
    data_generators=[data_generator],
    trainer=trainer,
    evaluator=retention_evaluator
)

# Run distillation with built-in evaluation
distilled_model = distiller.distill()

# Or evaluate separately after distillation
benchmark_results = benchmark_evaluator.evaluate(
    student_model=distilled_model,
    teacher_models=teacher_models,
    examples=test_examples
)
print(benchmark_results)
```

### Using the ResponseIntegrator

```python
from domainfusion.distillation import ResponseIntegrator

# Create a response integrator with an LLM
integrator = ResponseIntegrator(
    integration_model=llm_model,
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
    
    Create a unified response that integrates all relevant insights from these experts,
    resolving any contradictions and creating a coherent narrative.
    """
)

# Integrate responses
integrated_response = integrator.integrate(
    query="How can sprint athletes optimize their recovery between races?",
    responses={
        "biomechanics": biomechanics_response,
        "physiology": physiology_response,
        "nutrition": nutrition_response
    }
)
```

## Extension Points

The distillation module is designed to be extended in various ways:

1. Create custom data generators by implementing the `DataGenerator` interface
2. Create custom trainers by implementing the `Trainer` interface
3. Create custom evaluators by implementing the `Evaluator` interface
4. Implement domain-specific accuracy and coherence metrics
5. Create custom response integration strategies

These extension points allow for flexibility in implementing knowledge distillation for specific domains and requirements.
