# Utility Modules

This document describes the utility modules in the DomainFusion framework, which provide common functionality used throughout the system.

## Table of Contents

- [Text Utilities](#text-utilities)
- [Embedding Utilities](#embedding-utilities)
- [Validation Utilities](#validation-utilities)
- [Formatting Utilities](#formatting-utilities)
- [Logging Utilities](#logging-utilities)
- [Caching Utilities](#caching-utilities)
- [Prompting Utilities](#prompting-utilities)

## Text Utilities

The `text.py` module provides utilities for text processing.

### Functions

#### `truncate_text(text, max_length, add_ellipsis=True)`

Truncates text to a maximum length, optionally adding an ellipsis.

- **Parameters**:
  - `text` (str): The text to truncate
  - `max_length` (int): Maximum length in characters
  - `add_ellipsis` (bool): Whether to add "..." at the end of truncated text
- **Returns**: Truncated text (str)

#### `split_text(text, chunk_size, overlap=0)`

Splits text into chunks with optional overlap.

- **Parameters**:
  - `text` (str): The text to split
  - `chunk_size` (int): Maximum size of each chunk in characters
  - `overlap` (int): Number of characters to overlap between chunks
- **Returns**: List of text chunks (List[str])

#### `normalize_text(text, lowercase=True, remove_accents=False)`

Normalizes text by optionally lowercasing and removing accents.

- **Parameters**:
  - `text` (str): The text to normalize
  - `lowercase` (bool): Whether to convert to lowercase
  - `remove_accents` (bool): Whether to remove accents
- **Returns**: Normalized text (str)

#### `extract_keywords(text, min_length=3, max_keywords=None)`

Extracts potential keywords from text.

- **Parameters**:
  - `text` (str): The text to extract keywords from
  - `min_length` (int): Minimum length of keywords
  - `max_keywords` (int, optional): Maximum number of keywords to return
- **Returns**: List of extracted keywords (List[str])

#### `find_domain_relevance(text, domain_keywords)`

Calculates text relevance to different domains based on keyword occurrence.

- **Parameters**:
  - `text` (str): The text to analyze
  - `domain_keywords` (dict): Dict mapping domain names to lists of domain-specific keywords
- **Returns**: Dict mapping domain names to relevance scores (Dict[str, float])

## Embedding Utilities

The `embedding.py` module provides utilities for vector operations and similarity calculations.

### Functions

#### `cosine_similarity(vector_a, vector_b)`

Calculates cosine similarity between two vectors.

- **Parameters**:
  - `vector_a` (np.ndarray): First vector
  - `vector_b` (np.ndarray): Second vector
- **Returns**: Cosine similarity score (float)

#### `euclidean_distance(vector_a, vector_b)`

Calculates Euclidean distance between two vectors.

- **Parameters**:
  - `vector_a` (np.ndarray): First vector
  - `vector_b` (np.ndarray): Second vector
- **Returns**: Euclidean distance (float)

#### `normalize_vector(vector)`

Normalizes a vector to unit length.

- **Parameters**:
  - `vector` (np.ndarray): Input vector
- **Returns**: Normalized vector with unit length (np.ndarray)

#### `compute_similarity_matrix(vectors, method='cosine')`

Computes similarity matrix between all pairs of vectors.

- **Parameters**:
  - `vectors` (List[np.ndarray]): List of vectors
  - `method` (str): Similarity method ('cosine' or 'euclidean')
- **Returns**: Similarity matrix (np.ndarray)

#### `find_most_similar(query_vector, vector_database, method='cosine', top_k=1)`

Finds the most similar vectors to a query vector.

- **Parameters**:
  - `query_vector` (np.ndarray): Query vector
  - `vector_database` (List[np.ndarray]): List of vectors to search
  - `method` (str): Similarity method ('cosine' or 'euclidean')
  - `top_k` (int): Number of most similar vectors to return
- **Returns**: List of tuples (index, similarity_score) (List[Tuple[int, float]])

#### `softmax(scores, temperature=1.0)`

Applies softmax function to a list of scores.

- **Parameters**:
  - `scores` (List[float]): Input scores
  - `temperature` (float): Temperature parameter controlling distribution sharpness
- **Returns**: Softmax probabilities that sum to 1 (List[float])

#### `weighted_average_embeddings(embeddings, weights=None)`

Computes weighted average of multiple embeddings.

- **Parameters**:
  - `embeddings` (List[np.ndarray]): List of embedding vectors
  - `weights` (List[float], optional): List of weights (defaults to equal weights)
- **Returns**: Weighted average embedding (np.ndarray)

## Validation Utilities

The `validation.py` module provides utilities for input validation.

### Functions

#### `validate_type(value, expected_type, param_name='parameter')`

Validates that a value is of the expected type.

- **Parameters**:
  - `value` (Any): Value to validate
  - `expected_type` (Type): Expected type
  - `param_name` (str): Name of parameter for error message
- **Raises**: TypeError if value is not of expected_type

#### `validate_list_items_type(items, expected_type, param_name='list')`

Validates that all items in a list are of the expected type.

- **Parameters**:
  - `items` (List[Any]): List of items to validate
  - `expected_type` (Type): Expected type for all items
  - `param_name` (str): Name of parameter for error message
- **Raises**: TypeError if any item is not of expected_type

#### `validate_dict_types(data, expected_types, allow_missing=False)`

Validates that a dictionary has keys with values of expected types.

- **Parameters**:
  - `data` (Dict[str, Any]): Dictionary to validate
  - `expected_types` (Dict[str, Type]): Dictionary mapping keys to their expected types
  - `allow_missing` (bool): Whether to allow keys in expected_types to be missing from data
- **Raises**: TypeError if any value is not of expected type, KeyError if a required key is missing

#### `validate_in_range(value, min_value=None, max_value=None, param_name='parameter')`

Validates that a numeric value is within the specified range.

- **Parameters**:
  - `value` (int/float): Value to validate
  - `min_value` (int/float, optional): Minimum allowed value (inclusive)
  - `max_value` (int/float, optional): Maximum allowed value (inclusive)
  - `param_name` (str): Name of parameter for error message
- **Raises**: ValueError if value is outside the specified range

#### `validate_non_empty(value, param_name='parameter')`

Validates that a string, list, or dictionary is not empty.

- **Parameters**:
  - `value` (str/List/Dict): Value to validate
  - `param_name` (str): Name of parameter for error message
- **Raises**: ValueError if value is empty

#### `validate_function_signature(func, required_params=None, required_return_type=None)`

Validates that a function has the required parameters and return type.

- **Parameters**:
  - `func` (Callable): Function to validate
  - `required_params` (List[str], optional): List of parameter names that must be present
  - `required_return_type` (Type, optional): Required return type annotation
- **Raises**: TypeError if function signature doesn't match requirements

#### `validate_domain_name(domain)`

Validates that a domain name is properly formatted.

- **Parameters**:
  - `domain` (str): Domain name to validate
- **Raises**: ValueError if domain name is invalid

#### `validate_probability(value, param_name='probability')`

Validates that a value is a valid probability (between 0 and 1).

- **Parameters**:
  - `value` (float): Value to validate
  - `param_name` (str): Name of parameter for error message
- **Raises**: ValueError if value is not a valid probability

## Formatting Utilities

The `formatting.py` module provides utilities for formatting responses and outputs.

### Functions

#### `format_domain_response(domain, response, confidence=None)`

Formats a response from a domain expert.

- **Parameters**:
  - `domain` (str): Domain name
  - `response` (str): Response text
  - `confidence` (float, optional): Confidence score (0-1)
- **Returns**: Formatted response with domain header (str)

#### `format_multiple_responses(responses, confidences=None)`

Formats multiple domain responses into a single text.

- **Parameters**:
  - `responses` (Dict[str, str]): Dictionary mapping domain names to response texts
  - `confidences` (Dict[str, float], optional): Dictionary mapping domain names to confidence scores
- **Returns**: Formatted text with all responses (str)

#### `markdown_table(headers, rows)`

Formats data as a Markdown table.

- **Parameters**:
  - `headers` (List[str]): List of column headers
  - `rows` (List[List[Any]]): List of rows, where each row is a list of values
- **Returns**: Markdown-formatted table (str)

#### `format_json(data, indent=2)`

Formats data as a JSON string.

- **Parameters**:
  - `data` (Any): Data to format
  - `indent` (int): Number of spaces for indentation
- **Returns**: Formatted JSON string (str)

#### `format_confidence_distribution(confidences, max_bar_length=40)`

Formats a confidence distribution as a text-based bar chart.

- **Parameters**:
  - `confidences` (Dict[str, float]): Dictionary mapping domain names to confidence scores (0-1)
  - `max_bar_length` (int): Maximum length of each bar in the chart
- **Returns**: Text-based bar chart showing confidence distribution (str)

#### `format_error(error_type, message, details=None)`

Formats an error message.

- **Parameters**:
  - `error_type` (str): Type of error
  - `message` (str): Error message
  - `details` (Dict[str, Any], optional): Dictionary with additional error details
- **Returns**: Formatted error message (str)

#### `format_numbered_list(items, start=1)`

Formats a list of items as a numbered list.

- **Parameters**:
  - `items` (List[str]): List of items
  - `start` (int): Starting number
- **Returns**: Formatted numbered list (str)

#### `highlight_text(text, highlights, before_marker='**', after_marker='**')`

Highlights specific terms in a text.

- **Parameters**:
  - `text` (str): Text to process
  - `highlights` (List[str]): List of terms to highlight
  - `before_marker` (str): Marker to insert before highlighted term
  - `after_marker` (str): Marker to insert after highlighted term
- **Returns**: Text with highlighted terms (str)

## Logging Utilities

The `logging.py` module provides utilities for logging within the DomainFusion framework.

## Caching Utilities

The `caching.py` module provides utilities for caching results to improve performance.

## Prompting Utilities

The `prompting.py` module provides utilities for working with prompt templates and prompt engineering.
