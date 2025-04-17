"""
Formatting utilities for the DomainFusion framework.

This module provides functions for formatting responses, structured outputs,
and displaying results in various formats.
"""

from typing import Dict, List, Any, Optional, Union
import json
import re


def format_domain_response(domain: str, response: str, confidence: Optional[float] = None) -> str:
    """
    Format a response from a domain expert.
    
    Args:
        domain: Domain name
        response: Response text
        confidence: Optional confidence score (0-1)
        
    Returns:
        Formatted response with domain header
    """
    header = f"[{domain}"
    if confidence is not None:
        header += f" ({int(confidence * 100)}%)"
    header += "]:"
    
    return f"{header}\n{response}"


def format_multiple_responses(responses: Dict[str, str], 
                             confidences: Optional[Dict[str, float]] = None) -> str:
    """
    Format multiple domain responses into a single text.
    
    Args:
        responses: Dictionary mapping domain names to response texts
        confidences: Optional dictionary mapping domain names to confidence scores
        
    Returns:
        Formatted text with all responses
    """
    formatted_responses = []
    
    for domain, response in responses.items():
        confidence = confidences.get(domain) if confidences else None
        formatted_responses.append(format_domain_response(domain, response, confidence))
    
    return "\n\n".join(formatted_responses)


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    """
    Format data as a Markdown table.
    
    Args:
        headers: List of column headers
        rows: List of rows, where each row is a list of values
        
    Returns:
        Markdown-formatted table
    """
    if not headers or not rows:
        return ""
    
    # Create header row
    table = "| " + " | ".join(str(h) for h in headers) + " |\n"
    
    # Create separator row
    table += "| " + " | ".join("---" for _ in headers) + " |\n"
    
    # Create data rows
    for row in rows:
        table += "| " + " | ".join(str(cell) for cell in row) + " |\n"
    
    return table


def format_json(data: Any, indent: int = 2) -> str:
    """
    Format data as a JSON string.
    
    Args:
        data: Data to format
        indent: Number of spaces for indentation
        
    Returns:
        Formatted JSON string
    """
    return json.dumps(data, indent=indent, ensure_ascii=False)


def format_confidence_distribution(confidences: Dict[str, float], 
                                  max_bar_length: int = 40) -> str:
    """
    Format a confidence distribution as a text-based bar chart.
    
    Args:
        confidences: Dictionary mapping domain names to confidence scores (0-1)
        max_bar_length: Maximum length of each bar in the chart
        
    Returns:
        Text-based bar chart showing confidence distribution
    """
    if not confidences:
        return "No confidence scores available."
    
    # Find the maximum confidence score
    max_confidence = max(confidences.values())
    if max_confidence == 0:
        max_confidence = 1  # Avoid division by zero
    
    result = []
    
    # Sort domains by confidence score (descending)
    sorted_domains = sorted(confidences.keys(), key=lambda d: confidences[d], reverse=True)
    
    for domain in sorted_domains:
        confidence = confidences[domain]
        percentage = int(confidence * 100)
        
        # Calculate bar length proportional to confidence
        bar_length = int((confidence / max_confidence) * max_bar_length)
        bar = '█' * bar_length
        
        # Format each line: "domain_name [bar] percentage%"
        max_domain_length = max(len(d) for d in confidences.keys())
        domain_padded = domain.ljust(max_domain_length)
        result.append(f"{domain_padded} {bar} {percentage}%")
    
    return "\n".join(result)


def format_error(error_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> str:
    """
    Format an error message.
    
    Args:
        error_type: Type of error
        message: Error message
        details: Optional dictionary with additional error details
        
    Returns:
        Formatted error message
    """
    result = f"Error [{error_type}]: {message}"
    
    if details:
        result += "\n\nDetails:"
        for key, value in details.items():
            result += f"\n- {key}: {value}"
    
    return result


def format_numbered_list(items: List[str], start: int = 1) -> str:
    """
    Format a list of items as a numbered list.
    
    Args:
        items: List of items
        start: Starting number
        
    Returns:
        Formatted numbered list
    """
    if not items:
        return ""
    
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start))


def highlight_text(text: str, highlights: List[str], 
                  before_marker: str = "**", after_marker: str = "**") -> str:
    """
    Highlight specific terms in a text.
    
    Args:
        text: Text to process
        highlights: List of terms to highlight
        before_marker: Marker to insert before highlighted term
        after_marker: Marker to insert after highlighted term
        
    Returns:
        Text with highlighted terms
    """
    result = text
    
    # Sort highlights by length (descending) to avoid partial replacements
    sorted_highlights = sorted(highlights, key=len, reverse=True)
    
    for term in sorted_highlights:
        if not term:
            continue
            
        # Use case-insensitive replacement
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        result = pattern.sub(f"{before_marker}\\g<0>{after_marker}", result)
    
    return result 