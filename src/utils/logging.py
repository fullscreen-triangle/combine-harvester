"""
Logging utilities for the DomainFusion framework.

This module provides functions and classes for configuring and using
logging throughout the framework.
"""

import logging
import sys
import os
from typing import Optional, Union, Dict, Any, TextIO
import json
from datetime import datetime


class DomainFusionLogger:
    """
    Logger for the DomainFusion framework.
    
    This class provides a wrapper around the standard Python logging
    module with additional features specific to DomainFusion.
    """
    
    # Log levels mapping
    LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    
    def __init__(self, name: str, level: str = "info", 
                log_file: Optional[str] = None,
                log_format: Optional[str] = None,
                json_format: bool = False):
        """
        Initialize a new logger.
        
        Args:
            name: Name of the logger
            level: Log level ("debug", "info", "warning", "error", "critical")
            log_file: Optional file path to log to
            log_format: Optional format string for log messages
            json_format: Whether to log in JSON format
        """
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Set level
        if level.lower() not in self.LEVELS:
            raise ValueError(f"Invalid log level: {level}. Must be one of {list(self.LEVELS.keys())}")
        self.logger.setLevel(self.LEVELS[level.lower()])
        
        # Remove existing handlers if any
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatter
        if log_format is None:
            if json_format:
                # JSON formatter will be handled in the handler
                log_format = "%(message)s"
            else:
                log_format = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
        
        formatter = logging.Formatter(log_format)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Create file handler if log_file is provided
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.json_format = json_format
    
    def _format_json(self, level: str, message: str, **kwargs) -> str:
        """
        Format log entry as JSON.
        
        Args:
            level: Log level
            message: Log message
            **kwargs: Additional fields to include in the JSON
            
        Returns:
            JSON-formatted log entry
        """
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "logger": self.name,
            "level": level,
            "message": message
        }
        
        # Add additional fields
        for key, value in kwargs.items():
            log_data[key] = value
        
        return json.dumps(log_data)
    
    def debug(self, message: str, **kwargs) -> None:
        """
        Log a debug message.
        
        Args:
            message: Message to log
            **kwargs: Additional fields for JSON format
        """
        if self.json_format:
            self.logger.debug(self._format_json("debug", message, **kwargs))
        else:
            self.logger.debug(message)
    
    def info(self, message: str, **kwargs) -> None:
        """
        Log an info message.
        
        Args:
            message: Message to log
            **kwargs: Additional fields for JSON format
        """
        if self.json_format:
            self.logger.info(self._format_json("info", message, **kwargs))
        else:
            self.logger.info(message)
    
    def warning(self, message: str, **kwargs) -> None:
        """
        Log a warning message.
        
        Args:
            message: Message to log
            **kwargs: Additional fields for JSON format
        """
        if self.json_format:
            self.logger.warning(self._format_json("warning", message, **kwargs))
        else:
            self.logger.warning(message)
    
    def error(self, message: str, **kwargs) -> None:
        """
        Log an error message.
        
        Args:
            message: Message to log
            **kwargs: Additional fields for JSON format
        """
        if self.json_format:
            self.logger.error(self._format_json("error", message, **kwargs))
        else:
            self.logger.error(message)
    
    def critical(self, message: str, **kwargs) -> None:
        """
        Log a critical message.
        
        Args:
            message: Message to log
            **kwargs: Additional fields for JSON format
        """
        if self.json_format:
            self.logger.critical(self._format_json("critical", message, **kwargs))
        else:
            self.logger.critical(message)
    
    def log_router_decision(self, query: str, domain: str, confidence: float) -> None:
        """
        Log a router decision.
        
        Args:
            query: Query that was routed
            domain: Domain the query was routed to
            confidence: Confidence score for the routing decision
        """
        if self.json_format:
            self.info("Router decision", 
                     query=query, 
                     domain=domain, 
                     confidence=confidence,
                     component="router")
        else:
            self.info(f"Router decision: Query='{query}' -> Domain='{domain}' (confidence={confidence:.2f})")
    
    def log_model_call(self, model_name: str, prompt: str, response: str, 
                      latency: Optional[float] = None) -> None:
        """
        Log a model call.
        
        Args:
            model_name: Name of the model called
            prompt: Prompt sent to the model
            response: Response received from the model
            latency: Optional latency in seconds
        """
        if self.json_format:
            self.debug("Model call", 
                      model=model_name,
                      prompt=prompt,
                      response=response,
                      latency=latency,
                      component="model")
        else:
            latency_str = f" (latency: {latency:.2f}s)" if latency is not None else ""
            self.debug(f"Model call to '{model_name}'{latency_str}:\n"
                     f"Prompt: {prompt}\n"
                     f"Response: {response}")


def get_logger(name: str = "domainfusion", level: str = "info",
              log_file: Optional[str] = None,
              json_format: bool = False) -> DomainFusionLogger:
    """
    Get a logger instance.
    
    Args:
        name: Name of the logger
        level: Log level ("debug", "info", "warning", "error", "critical")
        log_file: Optional file path to log to
        json_format: Whether to log in JSON format
        
    Returns:
        Logger instance
    """
    return DomainFusionLogger(name, level, log_file, json_format=json_format)
