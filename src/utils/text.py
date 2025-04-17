"""
Text processing utilities for the DomainFusion framework.

This module provides functions for common text processing tasks such as
truncation, tokenization, and text normalization.
"""

import re
import unicodedata
from typing import List, Optional, Tuple


def truncate_text(text: str, max_length: int, add_ellipsis: bool = True) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: The text to truncate
        max_length: Maximum length in characters
        add_ellipsis: Whether to add "..." at the end of truncated text
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    if add_ellipsis:
        # Remove last 3 chars to make room for ellipsis
        truncated = truncated[:-3] + "..."
    
    return truncated


def split_text(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """
    Split text into chunks with optional overlap.
    
    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Try to find a natural break point (newline or period followed by space)
        if end < len(text):
            for break_point in ['\n', '. ', '? ', '! ']:
                last_break = text.rfind(break_point, start, end)
                if last_break != -1:
                    end = last_break + 1  # Include the break character
                    break
        
        chunks.append(text[start:end])
        start = end - overlap if overlap > 0 else end
    
    return chunks


def normalize_text(text: str, lowercase: bool = True, remove_accents: bool = False) -> str:
    """
    Normalize text by optionally lowercasing and removing accents.
    
    Args:
        text: The text to normalize
        lowercase: Whether to convert to lowercase
        remove_accents: Whether to remove accents
        
    Returns:
        Normalized text
    """
    if lowercase:
        text = text.lower()
    
    if remove_accents:
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
    
    return text


def extract_keywords(text: str, min_length: int = 3, max_keywords: Optional[int] = None) -> List[str]:
    """
    Extract potential keywords from text.
    
    This is a simple implementation that extracts words longer than min_length
    and filters out common stop words.
    
    Args:
        text: The text to extract keywords from
        min_length: Minimum length of keywords
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of extracted keywords
    """
    # Simple list of common English stop words
    stop_words = {
        'the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with',
        'by', 'of', 'from', 'as', 'is', 'was', 'were', 'be', 'been',
        'being', 'are', 'that', 'this', 'these', 'those', 'have', 'has',
        'had', 'not', 'but', 'what', 'which', 'who', 'when', 'where',
        'how', 'why', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'some', 'such', 'than', 'too', 'very', 'can', 'will',
        'just', 'should', 'now'
    }
    
    # Normalize text and extract words
    normalized = normalize_text(text, lowercase=True)
    words = re.findall(r'\b\w+\b', normalized)
    
    # Filter words by length and exclude stop words
    keywords = [word for word in words if len(word) >= min_length and word not in stop_words]
    
    # Remove duplicates while preserving order
    unique_keywords = []
    seen = set()
    for kw in keywords:
        if kw not in seen:
            unique_keywords.append(kw)
            seen.add(kw)
    
    # Limit to max_keywords if specified
    if max_keywords is not None and len(unique_keywords) > max_keywords:
        unique_keywords = unique_keywords[:max_keywords]
    
    return unique_keywords


def find_domain_relevance(text: str, domain_keywords: dict) -> dict:
    """
    Calculate text relevance to different domains based on keyword occurrence.
    
    Args:
        text: The text to analyze
        domain_keywords: Dict mapping domain names to lists of domain-specific keywords
        
    Returns:
        Dict mapping domain names to relevance scores (0.0-1.0)
    """
    normalized_text = normalize_text(text, lowercase=True)
    relevance_scores = {}
    
    for domain, keywords in domain_keywords.items():
        matches = 0
        for keyword in keywords:
            matches += normalized_text.count(keyword.lower())
        
        # Normalize score based on number of keywords
        if keywords:
            relevance_scores[domain] = min(1.0, matches / len(keywords))
        else:
            relevance_scores[domain] = 0.0
    
    return relevance_scores 