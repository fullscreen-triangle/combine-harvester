"""
Validation utilities for the DomainFusion framework.

This module provides functions for validating inputs, checking types,
and ensuring data consistency.
"""

from typing import Any, Dict, List, Type, Union, Optional, Callable
import inspect


def validate_type(value: Any, expected_type: Type, param_name: str = 'parameter') -> None:
    """
    Validate that a value is of the expected type.
    
    Args:
        value: Value to validate
        expected_type: Expected type
        param_name: Name of parameter for error message
        
    Raises:
        TypeError: If value is not of expected_type
    """
    if not isinstance(value, expected_type):
        received_type = type(value).__name__
        expected_type_name = expected_type.__name__
        raise TypeError(f"{param_name} must be of type {expected_type_name}, got {received_type}")


def validate_list_items_type(items: List[Any], expected_type: Type, param_name: str = 'list') -> None:
    """
    Validate that all items in a list are of the expected type.
    
    Args:
        items: List of items to validate
        expected_type: Expected type for all items
        param_name: Name of parameter for error message
        
    Raises:
        TypeError: If any item is not of expected_type
    """
    validate_type(items, list, param_name)
    
    for i, item in enumerate(items):
        if not isinstance(item, expected_type):
            received_type = type(item).__name__
            expected_type_name = expected_type.__name__
            raise TypeError(f"All items in {param_name} must be of type {expected_type_name}, "
                           f"but item at index {i} is of type {received_type}")


def validate_dict_types(data: Dict[str, Any], 
                        expected_types: Dict[str, Type], 
                        allow_missing: bool = False) -> None:
    """
    Validate that a dictionary has keys with values of expected types.
    
    Args:
        data: Dictionary to validate
        expected_types: Dictionary mapping keys to their expected types
        allow_missing: Whether to allow keys in expected_types to be missing from data
        
    Raises:
        TypeError: If any value is not of expected type
        KeyError: If a required key is missing
    """
    validate_type(data, dict, 'data')
    
    for key, expected_type in expected_types.items():
        if key not in data:
            if not allow_missing:
                raise KeyError(f"Required key '{key}' is missing from dictionary")
            continue
        
        if not isinstance(data[key], expected_type):
            received_type = type(data[key]).__name__
            expected_type_name = expected_type.__name__
            raise TypeError(f"Value for key '{key}' must be of type {expected_type_name}, "
                           f"got {received_type}")


def validate_in_range(value: Union[int, float], 
                      min_value: Optional[Union[int, float]] = None, 
                      max_value: Optional[Union[int, float]] = None, 
                      param_name: str = 'parameter') -> None:
    """
    Validate that a numeric value is within the specified range.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        param_name: Name of parameter for error message
        
    Raises:
        ValueError: If value is outside the specified range
    """
    if min_value is not None and value < min_value:
        raise ValueError(f"{param_name} must be at least {min_value}, got {value}")
    
    if max_value is not None and value > max_value:
        raise ValueError(f"{param_name} must be at most {max_value}, got {value}")


def validate_non_empty(value: Union[str, List, Dict], param_name: str = 'parameter') -> None:
    """
    Validate that a string, list, or dictionary is not empty.
    
    Args:
        value: Value to validate
        param_name: Name of parameter for error message
        
    Raises:
        ValueError: If value is empty
    """
    if len(value) == 0:
        type_name = type(value).__name__
        raise ValueError(f"{param_name} must not be an empty {type_name}")


def validate_function_signature(func: Callable, 
                               required_params: Optional[List[str]] = None,
                               required_return_type: Optional[Type] = None) -> None:
    """
    Validate that a function has the required parameters and return type.
    
    Args:
        func: Function to validate
        required_params: List of parameter names that must be present
        required_return_type: Required return type annotation
        
    Raises:
        TypeError: If function signature doesn't match requirements
    """
    if not callable(func):
        raise TypeError(f"Expected a callable function, got {type(func).__name__}")
    
    # Check required parameters
    if required_params:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        for param in required_params:
            if param not in params:
                raise TypeError(f"Function must have a '{param}' parameter")
    
    # Check return type if required
    if required_return_type:
        annotations = getattr(func, '__annotations__', {})
        return_annotation = annotations.get('return')
        
        if return_annotation is None:
            raise TypeError(f"Function must have a return type annotation")
        
        if return_annotation != required_return_type:
            expected_name = required_return_type.__name__
            actual_name = return_annotation.__name__
            raise TypeError(f"Function must return {expected_name}, but returns {actual_name}")


def validate_domain_name(domain: str) -> None:
    """
    Validate that a domain name is properly formatted.
    
    Args:
        domain: Domain name to validate
        
    Raises:
        ValueError: If domain name is invalid
    """
    validate_type(domain, str, 'domain')
    validate_non_empty(domain, 'domain')
    
    if not domain.isalnum() and not all(c.isalnum() or c in ['_', '-'] for c in domain):
        raise ValueError(f"Domain name '{domain}' may only contain alphanumeric characters, "
                        f"underscores, and hyphens")


def validate_probability(value: float, param_name: str = 'probability') -> None:
    """
    Validate that a value is a valid probability (between 0 and 1).
    
    Args:
        value: Value to validate
        param_name: Name of parameter for error message
        
    Raises:
        ValueError: If value is not a valid probability
    """
    validate_type(value, (int, float), param_name)
    validate_in_range(value, 0.0, 1.0, param_name) 