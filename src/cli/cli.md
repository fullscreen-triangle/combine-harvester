# DomainFusion CLI

The DomainFusion Command Line Interface (CLI) provides a convenient way to interact with domain-expert language models using the architectural patterns implemented in the DomainFusion framework.

## Installation

The CLI is automatically installed with the DomainFusion package:

```bash
pip install domainfusion
```

## Basic Usage

After installation, you can use the CLI with the following command:

```bash
python -m domainfusion [options] [command]
```

For help:

```bash
python -m domainfusion --help
```

## Global Options

- `--verbose, -v`: Increase verbosity (can be used multiple times)
- `--config, -c`: Path to configuration file (YAML or JSON)

## Commands

The CLI provides the following commands:

### query

Send a query to domain-expert models. This uses the Router-Based Ensemble pattern to route your query to the most appropriate domain expert.

```bash
python -m domainfusion query "What's the optimal stride frequency for a 100m sprinter?"
```

Options:
- `--router, -r`: Router type (keyword, embedding, classifier, llm)
- `--models, -m`: Model names to consider for routing
- `--top-k, -k`: Number of top domains to consider
- `--mixer`: Mixer type for combining multiple responses
- `--show-confidence`: Show confidence scores in the output

### chain

Chain multiple models sequentially. This implements the Sequential Chaining pattern, passing your query through multiple domain experts in sequence.

```bash
python -m domainfusion chain "How can I improve sprint performance?" --models biomechanics nutrition recovery
```

Options:
- `--models, -m`: Models to chain in sequence (required)
- `--show-intermediate, -s`: Show intermediate responses

### route

Route a query without executing any models. This is useful for testing routing configurations.

```bash
python -m domainfusion route "What's the best nutrition strategy for marathon runners?"
```

Options:
- `--router, -r`: Router type (keyword, embedding, classifier, llm)
- `--models, -m`: Model names to consider for routing
- `--top-k, -k`: Number of top domains to show

### models

Manage models in the DomainFusion registry.

#### list

List all available models:

```bash
python -m domainfusion models list
```

#### add

Add a new model to the registry:

```bash
python -m domainfusion models add sprint_expert --engine ollama --model-name sprint-expert --domain sprint_biomechanics
```

Options:
- `--engine, -e`: Engine type (ollama, openai, anthropic, huggingface)
- `--model-name`: Name of the model in the provider
- `--domain, -d`: Domain of expertise for this model
- `--api-key`: API key for the provider

#### remove

Remove a model from the registry:

```bash
python -m domainfusion models remove sprint_expert
```

## Configuration File

The CLI can load configuration from a YAML or JSON file. This is useful for setting up complex configurations with multiple models, routers, and mixers.

Example configuration file (`domainfusion_config.yaml`):

```yaml
# Models configuration
models:
  sprint_biomechanics:
    engine: ollama
    model_name: sprint-expert
    domain: sprint_biomechanics
  
  sports_nutrition:
    engine: ollama
    model_name: nutrition-expert
    domain: sports_nutrition
  
  general:
    engine: openai
    model_name: gpt-4
    api_key: sk-yourapikey

# Default models to use if not specified
default_models:
  - sprint_biomechanics
  - sports_nutrition
  - general

# Default model if no domain-specific model is found
default_model: general

# Domain descriptions for routers
domains:
  sprint_biomechanics: "Sprint biomechanics focuses on the physics and physiology of sprint running, including ground reaction forces, muscle activation patterns, and optimal body positions for maximum velocity and acceleration."
  sports_nutrition: "Sports nutrition focuses on how dietary intake affects athletic performance, including macronutrient timing, hydration strategies, and supplementation for optimal performance and recovery."

# Router configurations
routers:
  keyword:
    threshold: 0.2
  
  embedding:
    embedding_model: general
    threshold: 0.7
    temperature: 0.5

# Mixer configurations
mixers:
  synthesis:
    synthesis_model: general

# Prompt templates for chains
prompt_templates:
  sprint_biomechanics: "You are a biomechanics expert. Analyze this query from a biomechanical perspective: {query}"
  sports_nutrition: "You are a sports nutrition expert. Given this biomechanical context: {prev_response} and the original query: {query}, provide nutritional insights that complement the biomechanical analysis."
```

## Examples

### Using Router-Based Ensemble

```bash
python -m domainfusion query "What's the optimal stride frequency for a 100m sprinter?" --router embedding --top-k 1
```

### Using Sequential Chaining

```bash
python -m domainfusion chain "How can nutrition affect recovery from sprint training?" --models sprint_biomechanics sports_nutrition recovery --show-intermediate
```

### Routing Analysis

```bash
python -m domainfusion route "How should I adjust my diet during high-intensity training blocks?" --router embedding --top-k 3
```

### Using Configuration File

```bash
python -m domainfusion --config my_config.yaml query "What's the relationship between stride length and frequency?"
```

## Implementation Details

The CLI is implemented in the `src/cli` package with the following structure:

- `__init__.py`: Exports the main entry point
- `main.py`: Contains the core CLI functionality
- `cli.md`: This documentation file

The CLI uses the following architectural patterns from the DomainFusion framework:

1. **Router-Based Ensembles**: Implemented in the `query` command, routing queries to the most appropriate domain expert model
2. **Sequential Chaining**: Implemented in the `chain` command, passing queries through multiple domain experts in sequence
3. **Mixture of Experts**: Implemented in the `query` command with `--top-k` > 1, using a mixer to combine responses from multiple experts

## Error Handling

The CLI provides helpful error messages when:

- A configuration file is not found or has an invalid format
- Required models are not available in the registry
- Invalid command-line arguments are provided

## Extending the CLI

You can extend the CLI by:

1. Adding new router or mixer types in the DomainFusion framework
2. Creating custom models by implementing the Model interface
3. Defining custom prompt templates in your configuration file
