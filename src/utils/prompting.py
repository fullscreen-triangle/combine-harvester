"""
Prompting utilities for the DomainFusion framework.

This module provides functions and classes for working with prompt templates
and prompt engineering.
"""

from typing import Dict, Any, List, Optional, Union, Callable
import re
import string
from string import Template


class PromptTemplate:
    """
    Template for generating prompts with variable substitution.
    """
    
    def __init__(self, template: str, validator: Optional[Callable[[str], None]] = None):
        """
        Initialize a prompt template.
        
        Args:
            template: Template string with placeholders in the format {variable_name}
            validator: Optional function to validate the rendered prompt
        """
        self.template = template
        self.validator = validator
        
        # Extract variable names from template
        self.variable_names = self._extract_variable_names(template)
    
    def _extract_variable_names(self, template: str) -> List[str]:
        """
        Extract variable names from a template string.
        
        Args:
            template: Template string
            
        Returns:
            List of variable names
        """
        # Match {variable_name} pattern
        matches = re.findall(r'\{([a-zA-Z0-9_]+)\}', template)
        return list(set(matches))
    
    def render(self, **kwargs) -> str:
        """
        Render the template with the provided variables.
        
        Args:
            **kwargs: Variables to substitute in the template
            
        Returns:
            Rendered prompt
            
        Raises:
            KeyError: If a required variable is missing
            ValueError: If validation fails
        """
        # Check for missing variables
        missing_vars = [var for var in self.variable_names if var not in kwargs]
        if missing_vars:
            raise KeyError(f"Missing required variables: {', '.join(missing_vars)}")
        
        # Render template using string.format()
        rendered = self.template.format(**kwargs)
        
        # Validate if a validator is provided
        if self.validator:
            try:
                self.validator(rendered)
            except Exception as e:
                raise ValueError(f"Prompt validation failed: {str(e)}")
        
        return rendered


class ChainPromptTemplate:
    """
    Template for generating prompts in a sequential chain.
    """
    
    def __init__(self, templates: Dict[str, str]):
        """
        Initialize a chain prompt template.
        
        Args:
            templates: Dictionary mapping step names to template strings
        """
        self.templates = {}
        for step, template in templates.items():
            self.templates[step] = PromptTemplate(template)
    
    def render_step(self, step: str, context: Dict[str, Any]) -> str:
        """
        Render a specific step in the chain.
        
        Args:
            step: Step name
            context: Context dictionary with variables
            
        Returns:
            Rendered prompt for the step
            
        Raises:
            KeyError: If step not found or a required variable is missing
        """
        if step not in self.templates:
            raise KeyError(f"Step '{step}' not found in chain prompt templates")
        
        return self.templates[step].render(**context)


def create_system_prompt(roles: List[Dict[str, str]], 
                        instructions: str, 
                        constraints: Optional[List[str]] = None) -> str:
    """
    Create a system prompt with multiple domain expert roles.
    
    Args:
        roles: List of role dictionaries, each with 'domain' and 'description' keys
        instructions: Instructions for how to combine domain expertise
        constraints: Optional list of constraints or guidelines
        
    Returns:
        Formatted system prompt
    """
    prompt = "You are an expert in multiple domains:\n\n"
    
    # Add roles
    for i, role in enumerate(roles, 1):
        domain = role.get('domain', f"Domain {i}")
        description = role.get('description', "")
        prompt += f"{i}. {domain}: {description}\n\n"
    
    # Add instructions
    prompt += "Instructions:\n"
    prompt += f"{instructions}\n\n"
    
    # Add constraints if provided
    if constraints:
        prompt += "Constraints:\n"
        for i, constraint in enumerate(constraints, 1):
            prompt += f"{i}. {constraint}\n"
    
    return prompt


def create_critique_prompt(response: str, criteria: List[str]) -> str:
    """
    Create a prompt for critiquing a response based on specific criteria.
    
    Args:
        response: Response to critique
        criteria: List of criteria to evaluate
        
    Returns:
        Critique prompt
    """
    prompt = "Please critique the following response based on the provided criteria:\n\n"
    prompt += f"Response:\n\"\"\"\n{response}\n\"\"\"\n\n"
    
    prompt += "Criteria:\n"
    for i, criterion in enumerate(criteria, 1):
        prompt += f"{i}. {criterion}\n"
    
    prompt += "\nFor each criterion, provide a score from 1-5 and a brief explanation. "
    prompt += "Finally, suggest specific improvements to address any shortcomings."
    
    return prompt


def create_integration_prompt(responses: Dict[str, str]) -> str:
    """
    Create a prompt for integrating multiple domain-specific responses.
    
    Args:
        responses: Dictionary mapping domain names to responses
        
    Returns:
        Integration prompt
    """
    prompt = "Synthesize these perspectives into a comprehensive, integrated response:\n\n"
    
    for domain, response in responses.items():
        prompt += f"[{domain}]:\n{response}\n\n"
    
    prompt += "Provide a cohesive response that integrates insights from all domains, "
    prompt += "resolving any contradictions and creating a unified perspective. "
    prompt += "Ensure the response is balanced and gives appropriate weight to each domain."
    
    return prompt


def extract_reasoning_steps(response: str) -> List[str]:
    """
    Extract reasoning steps from a response that follows a step-by-step format.
    
    Args:
        response: Response text
        
    Returns:
        List of reasoning steps
    """
    # Try to match numbered steps (e.g., "1. First step")
    steps = re.findall(r'(?:^|\n)(\d+\.\s+.*?)(?=\n\d+\.|$)', response, re.DOTALL)
    
    if steps:
        return [step.strip() for step in steps]
    
    # Try to match steps indicated by Step/step keyword
    steps = re.findall(r'(?:^|\n)[Ss]tep\s+\d+:?\s+(.*?)(?=\n[Ss]tep\s+\d+|$)', response, re.DOTALL)
    
    if steps:
        return [step.strip() for step in steps]
    
    # Fall back to splitting by double newlines (paragraphs)
    paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
    return paragraphs


def add_cot_prompt(prompt: str) -> str:
    """
    Add Chain-of-Thought instructions to a prompt.
    
    Args:
        prompt: Original prompt
        
    Returns:
        Prompt with CoT instructions
    """
    cot_instruction = "\n\nPlease think through this step-by-step, showing your reasoning before providing the final answer."
    return prompt + cot_instruction
