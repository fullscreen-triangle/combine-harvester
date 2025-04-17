"""
Utility modules for the DomainFusion framework.

This package provides various utility functions and classes used throughout
the DomainFusion framework.
"""

# Text utilities
from .text import (
    truncate_text,
    split_text,
    normalize_text,
    extract_keywords,
    find_domain_relevance,
)

# Embedding utilities
from .embedding import (
    cosine_similarity,
    euclidean_distance,
    normalize_vector,
    compute_similarity_matrix,
    find_most_similar,
    softmax,
    weighted_average_embeddings,
)

# Validation utilities
from .validation import (
    validate_type,
    validate_list_items_type,
    validate_dict_types,
    validate_in_range,
    validate_non_empty,
    validate_function_signature,
    validate_domain_name,
    validate_probability,
)

# Formatting utilities
from .formatting import (
    format_domain_response,
    format_multiple_responses,
    markdown_table,
    format_json,
    format_confidence_distribution,
    format_error,
    format_numbered_list,
    highlight_text,
)

# Logging utilities
from .logging import (
    DomainFusionLogger,
    get_logger,
)

# Caching utilities
from .caching import (
    MemoryCache,
    DiskCache,
    cached,
    get_memory_cache,
    get_disk_cache,
)

# Prompting utilities
from .prompting import (
    PromptTemplate,
    ChainPromptTemplate,
    create_system_prompt,
    create_critique_prompt,
    create_integration_prompt,
    extract_reasoning_steps,
    add_cot_prompt,
)

__all__ = [
    # Text utilities
    'truncate_text',
    'split_text',
    'normalize_text',
    'extract_keywords',
    'find_domain_relevance',
    
    # Embedding utilities
    'cosine_similarity',
    'euclidean_distance',
    'normalize_vector',
    'compute_similarity_matrix',
    'find_most_similar',
    'softmax',
    'weighted_average_embeddings',
    
    # Validation utilities
    'validate_type',
    'validate_list_items_type',
    'validate_dict_types',
    'validate_in_range',
    'validate_non_empty',
    'validate_function_signature',
    'validate_domain_name',
    'validate_probability',
    
    # Formatting utilities
    'format_domain_response',
    'format_multiple_responses',
    'markdown_table',
    'format_json',
    'format_confidence_distribution',
    'format_error',
    'format_numbered_list',
    'highlight_text',
    
    # Logging utilities
    'DomainFusionLogger',
    'get_logger',
    
    # Caching utilities
    'MemoryCache',
    'DiskCache',
    'cached',
    'get_memory_cache',
    'get_disk_cache',
    
    # Prompting utilities
    'PromptTemplate',
    'ChainPromptTemplate',
    'create_system_prompt',
    'create_critique_prompt',
    'create_integration_prompt',
    'extract_reasoning_steps',
    'add_cot_prompt',
]
