"""
Command Line Interface for the DomainFusion framework.

This module provides command-line tools for working with domain-expert LLMs,
using the architectural patterns implemented in the DomainFusion framework.
"""

import argparse
import sys
import os
import json
import yaml
from typing import Dict, List, Optional, Any
import logging
import importlib.metadata

# Import DomainFusion core components
from ..core.registry import ModelRegistry
from ..routers import get_router
from ..mixers import get_mixer
from ..utils import get_logger
from ..utils.formatting import format_multiple_responses, format_confidence_distribution


def get_version() -> str:
    """Get the DomainFusion version."""
    try:
        return importlib.metadata.version("domainfusion")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description=f"DomainFusion CLI v{get_version()} - Tools for working with domain-expert LLMs"
    )
    
    # Global arguments
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Increase verbosity (can be used multiple times)')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to configuration file (YAML or JSON)')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Send a query to domain-expert models')
    query_parser.add_argument('text', help='The query text')
    query_parser.add_argument('--router', '-r', type=str, default='keyword',
                             help='Router type (keyword, embedding, classifier, llm)')
    query_parser.add_argument('--models', '-m', type=str, nargs='+',
                             help='Model names to consider for routing')
    query_parser.add_argument('--top-k', '-k', type=int, default=1,
                             help='Number of top domains to consider')
    query_parser.add_argument('--mixer', type=str, default='default',
                             help='Mixer type for combining multiple responses')
    query_parser.add_argument('--show-confidence', action='store_true',
                             help='Show confidence scores in the output')
    
    # Chain command
    chain_parser = subparsers.add_parser('chain', help='Chain multiple models sequentially')
    chain_parser.add_argument('text', help='The query text')
    chain_parser.add_argument('--models', '-m', type=str, nargs='+', required=True,
                             help='Models to chain in sequence')
    chain_parser.add_argument('--show-intermediate', '-s', action='store_true',
                             help='Show intermediate responses')
    
    # Router command
    router_parser = subparsers.add_parser('route', help='Route a query without executing models')
    router_parser.add_argument('text', help='The query text')
    router_parser.add_argument('--router', '-r', type=str, default='keyword',
                             help='Router type (keyword, embedding, classifier, llm)')
    router_parser.add_argument('--models', '-m', type=str, nargs='+',
                             help='Model names to consider for routing')
    router_parser.add_argument('--top-k', '-k', type=int, default=3,
                             help='Number of top domains to show')
    
    # Model management commands
    model_parser = subparsers.add_parser('models', help='Model management commands')
    model_subparsers = model_parser.add_subparsers(dest='model_command')
    
    # List models
    list_parser = model_subparsers.add_parser('list', help='List available models')
    
    # Add model
    add_parser = model_subparsers.add_parser('add', help='Add a new model')
    add_parser.add_argument('name', help='Name to identify the model')
    add_parser.add_argument('--engine', '-e', type=str, required=True,
                          help='Engine type (ollama, openai, anthropic, huggingface)')
    add_parser.add_argument('--model-name', type=str, required=True,
                          help='Name of the model in the provider')
    add_parser.add_argument('--domain', '-d', type=str,
                          help='Domain of expertise for this model')
    add_parser.add_argument('--api-key', type=str,
                          help='API key for the provider')
    
    # Remove model
    remove_parser = model_subparsers.add_parser('remove', help='Remove a model')
    remove_parser.add_argument('name', help='Name of the model to remove')
    
    return parser


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from file."""
    if not config_path:
        return {}
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        if config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                return json.load(f)
        elif config_path.endswith(('.yaml', '.yml')):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            print(f"Error: Unsupported config file format. Use JSON or YAML.")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading config file: {str(e)}")
        sys.exit(1)


def setup_registry(config: Dict[str, Any]) -> ModelRegistry:
    """Set up model registry from configuration."""
    registry = ModelRegistry()
    
    # Load models from config
    models_config = config.get('models', {})
    for name, model_config in models_config.items():
        try:
            engine = model_config.get('engine')
            model_name = model_config.get('model_name')
            
            if not engine or not model_name:
                print(f"Warning: Skipping model '{name}' due to incomplete configuration.")
                continue
            
            # Extract other parameters to pass to the model constructor
            kwargs = {k: v for k, v in model_config.items() 
                    if k not in ['engine', 'model_name']}
            
            registry.add_model(name, engine, model_name, **kwargs)
            
        except Exception as e:
            print(f"Error adding model '{name}': {str(e)}")
    
    return registry


def handle_query_command(args: argparse.Namespace, config: Dict[str, Any], registry: ModelRegistry) -> None:
    """Handle the 'query' command."""
    from ..core.ensemble import Ensemble
    
    # Get model names from args or config
    model_names = args.models or config.get('default_models', list(registry.models.keys()))
    
    # Set up router
    router_type = args.router
    router_config = config.get('routers', {}).get(router_type, {})
    router = get_router(router_type, **router_config)
    
    # Add domain descriptions if available
    domains_config = config.get('domains', {})
    for domain, description in domains_config.items():
        if hasattr(router, 'add_domain'):
            router.add_domain(domain, description)
    
    # Set up mixer
    mixer_type = args.mixer
    mixer_config = config.get('mixers', {}).get(mixer_type, {})
    mixer = get_mixer(mixer_type, **mixer_config)
    
    if args.verbose > 0:
        print(f"Using router: {router_type}")
        print(f"Using mixer: {mixer_type}")
        print(f"Available models: {model_names}")
    
    # Create ensemble
    ensemble = Ensemble(
        router=router,
        models=registry,
        mixer=mixer,
        default_model=config.get('default_model')
    )
    
    # Generate response
    if args.top_k > 1:
        response = ensemble.generate(args.text, top_k=args.top_k)
    else:
        response = ensemble.generate(args.text)
    
    # Print response
    print(response)


def handle_chain_command(args: argparse.Namespace, config: Dict[str, Any], registry: ModelRegistry) -> None:
    """Handle the 'chain' command."""
    from ..core.chain import Chain
    
    # Check if all models exist
    missing_models = [name for name in args.models if name not in registry.models]
    if missing_models:
        print(f"Error: The following models are not available: {', '.join(missing_models)}")
        print("Use 'domainfusion models list' to see available models.")
        sys.exit(1)
    
    # Get model objects
    models = [registry.get(name) for name in args.models]
    
    # Get prompt templates if available
    prompt_templates = {}
    templates_config = config.get('prompt_templates', {})
    for name in args.models:
        if name in templates_config:
            prompt_templates[name] = templates_config[name]
    
    # Create chain
    chain = Chain(models=models, prompt_templates=prompt_templates)
    
    # Generate response
    if args.show_intermediate:
        context = {"query": args.text, "responses": []}
        
        for i, model in enumerate(models):
            if model.name in prompt_templates:
                template = prompt_templates[model.name]
                prompt = template.format(
                    query=args.text,
                    prev_response=context["responses"][-1] if context["responses"] else "",
                    responses=context["responses"],
                    **context
                )
            else:
                prompt = args.text if i == 0 else f"Previous: {context['responses'][-1]}\nQuery: {args.text}"
            
            response = model.generate(prompt)
            context["responses"].append(response)
            
            print(f"\n--- {model.name} ---")
            print(response)
        
        print("\n--- Final Response ---")
        print(context["responses"][-1])
    else:
        response = chain.generate(args.text)
        print(response)


def handle_route_command(args: argparse.Namespace, config: Dict[str, Any], registry: ModelRegistry) -> None:
    """Handle the 'route' command."""
    # Get model names from args or config
    model_names = args.models or list(registry.models.keys())
    
    # Set up router
    router_type = args.router
    router_config = config.get('routers', {}).get(router_type, {})
    router = get_router(router_type, **router_config)
    
    # Add domain descriptions if available
    domains_config = config.get('domains', {})
    for domain, description in domains_config.items():
        if hasattr(router, 'add_domain'):
            router.add_domain(domain, description)
    
    if args.verbose > 0:
        print(f"Using router: {router_type}")
        print(f"Available models: {model_names}")
    
    # Get routing and confidences
    top_domains = router.route_multiple(args.text, model_names, args.top_k)
    confidences = {}
    
    if hasattr(router, 'get_confidence'):
        for domain in top_domains:
            confidences[domain] = router.get_confidence(args.text, domain)
    
    # Print results
    if confidences:
        print(format_confidence_distribution(confidences))
    else:
        for i, domain in enumerate(top_domains, 1):
            print(f"{i}. {domain}")


def handle_models_command(args: argparse.Namespace, config: Dict[str, Any], registry: ModelRegistry) -> None:
    """Handle the 'models' subcommands."""
    if args.model_command == 'list':
        # List available models
        if not registry.models:
            print("No models available.")
            print("Use 'domainfusion models add' to add a model.")
            return
        
        print("Available models:")
        for name, model in registry.models.items():
            print(f"  - {name} ({model.engine}): {model.model_name}")
    
    elif args.model_command == 'add':
        # Add a new model
        try:
            kwargs = {}
            if args.domain:
                kwargs['domain'] = args.domain
            if args.api_key:
                kwargs['api_key'] = args.api_key
            
            registry.add_model(args.name, args.engine, args.model_name, **kwargs)
            
            # Update config
            if 'models' not in config:
                config['models'] = {}
            
            config['models'][args.name] = {
                'engine': args.engine,
                'model_name': args.model_name,
                **kwargs
            }
            
            # Save config
            config_path = args.config or 'domainfusion_config.yaml'
            save_config(config, config_path)
            
            print(f"Model '{args.name}' added successfully.")
            print(f"Configuration saved to {config_path}")
            
        except Exception as e:
            print(f"Error adding model: {str(e)}")
    
    elif args.model_command == 'remove':
        # Remove a model
        if args.name not in registry.models:
            print(f"Error: Model '{args.name}' not found.")
            return
        
        # Remove from registry
        del registry.models[args.name]
        
        # Update config
        if 'models' in config and args.name in config['models']:
            del config['models'][args.name]
            
            # Save config
            config_path = args.config or 'domainfusion_config.yaml'
            save_config(config, config_path)
        
        print(f"Model '{args.name}' removed successfully.")


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        if config_path.endswith('.json'):
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        else:  # Default to YAML
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
    except Exception as e:
        print(f"Error saving config file: {str(e)}")


def main() -> None:
    """Main entry point for the CLI."""
    # Set up argument parser
    parser = setup_parser()
    args = parser.parse_args()
    
    # Handle no command
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Set up logging
    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    
    logger = get_logger(level=log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up model registry
    registry = setup_registry(config)
    
    # Handle commands
    if args.command == 'query':
        handle_query_command(args, config, registry)
    elif args.command == 'chain':
        handle_chain_command(args, config, registry)
    elif args.command == 'route':
        handle_route_command(args, config, registry)
    elif args.command == 'models':
        handle_models_command(args, config, registry)


if __name__ == "__main__":
    main()
