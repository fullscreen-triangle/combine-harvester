from typing import Dict, List, Optional, Any

from ..core import DomainExpert
from .data_generation import DataGenerator
from .trainer import Trainer
from .evaluator import Evaluator


class Distiller:
    """
    Manages the knowledge distillation process from multiple teacher models to a student model.
    
    This class coordinates the distillation process, including:
    1. Generating training data that spans multiple domains
    2. Getting responses from teacher models
    3. Training the student model to replicate the combined expertise
    4. Evaluating the student model's performance
    """

    def __init__(
        self,
        student_model: DomainExpert,
        teacher_models: Dict[str, DomainExpert],
        data_generators: List[DataGenerator],
        trainer: Trainer,
        evaluator: Optional[Evaluator] = None,
        output_dir: str = "./distilled_model"
    ):
        """
        Initialize a Distiller.
        
        Args:
            student_model: The model being trained to incorporate expertise from teachers
            teacher_models: Dictionary mapping domain names to teacher models
            data_generators: List of data generators for creating training examples
            trainer: Trainer for fine-tuning the student model
            evaluator: Optional evaluator for assessing model performance
            output_dir: Directory to save the distilled model and artifacts
        """
        self.student_model = student_model
        self.teacher_models = teacher_models
        self.data_generators = data_generators
        self.trainer = trainer
        self.evaluator = evaluator
        self.output_dir = output_dir
    
    def generate_training_data(self) -> List[Dict[str, Any]]:
        """
        Generate training data from all data generators.
        
        Returns:
            List of training examples
        """
        all_examples = []
        
        for generator in self.data_generators:
            examples = generator.generate(domains=list(self.teacher_models.keys()))
            all_examples.extend(examples)
        
        return all_examples
    
    def collect_teacher_responses(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Collect responses from teacher models for all examples.
        
        Args:
            examples: List of training examples
            
        Returns:
            Examples with teacher responses added
        """
        for example in examples:
            query = example["query"]
            teacher_responses = {}
            
            for domain, teacher in self.teacher_models.items():
                response = teacher.process(query)
                teacher_responses[domain] = response
            
            example["teacher_responses"] = teacher_responses
        
        return examples
    
    def integrate_responses(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Integrate responses from multiple teachers into a single target response.
        
        Args:
            examples: List of training examples with teacher responses
            
        Returns:
            Examples with integrated responses added as targets
        """
        for example in examples:
            # Start with the primary domain if specified
            if "primary_domain" in example and example["primary_domain"] in example["teacher_responses"]:
                primary_domain = example["primary_domain"]
                integrated = example["teacher_responses"][primary_domain].copy()
            else:
                # Otherwise start with the first teacher response
                domain = list(example["teacher_responses"].keys())[0]
                integrated = example["teacher_responses"][domain].copy()
            
            # Add a list of domains that contributed to the response
            domains = list(example["teacher_responses"].keys())
            integrated["source_domains"] = domains
            
            # Could implement more sophisticated integration strategies here
            
            example["target"] = integrated
        
        return examples
    
    def distill(self) -> DomainExpert:
        """
        Run the full distillation process.
        
        Returns:
            The distilled student model
        """
        # Step 1: Generate training data
        examples = self.generate_training_data()
        
        # Step 2: Collect responses from teacher models
        examples = self.collect_teacher_responses(examples)
        
        # Step 3: Integrate responses to create targets
        examples = self.integrate_responses(examples)
        
        # Step 4: Train the student model
        self.trainer.train(self.student_model, examples)
        
        # Step 5: Evaluate the student model if evaluator is provided
        if self.evaluator:
            evaluation_results = self.evaluator.evaluate(
                self.student_model,
                self.teacher_models,
                examples
            )
            print(f"Evaluation results: {evaluation_results}")
        
        return self.student_model
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the distilled student model.
        
        Args:
            path: Optional path to save the model (uses output_dir if not specified)
        """
        save_path = path or self.output_dir
        # Implementation depends on the specific model type
        # This is a placeholder for the actual saving logic
        print(f"Model saved to {save_path}")


class ResponseIntegrator:
    """
    Integrates responses from multiple domain experts.
    
    This class provides methods to combine responses from different expert models 
    into a unified response that incorporates insights from all domains.
    """

    def __init__(
        self,
        integration_model: Optional[DomainExpert] = None,
        prompt_template: Optional[str] = None
    ):
        """
        Initialize a ResponseIntegrator.
        
        Args:
            integration_model: Optional model to use for integrating responses
            prompt_template: Optional template for formatting integration prompts
        """
        self.integration_model = integration_model
        self.prompt_template = prompt_template
    
    def integrate(
        self,
        query: str,
        responses: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Integrate responses from multiple domains.
        
        Args:
            query: The original query
            responses: Dictionary mapping domain names to their responses
            
        Returns:
            An integrated response combining insights from all domains
        """
        if not responses:
            return {"error": "No responses to integrate"}
        
        if len(responses) == 1:
            domain = list(responses.keys())[0]
            return {
                "source_domain": domain,
                **responses[domain]
            }
        
        if self.integration_model and self.prompt_template:
            # Use the integration model with the prompt template
            formatted_responses = {}
            for domain, response in responses.items():
                # Extract the main content from the response
                content = response.get("content", str(response))
                formatted_responses[domain] = content
            
            # Format the prompt with the query and responses
            prompt = self.prompt_template.format(
                query=query,
                **formatted_responses
            )
            
            # Get the integrated response from the model
            integrated = self.integration_model.process(prompt)
            integrated["source_domains"] = list(responses.keys())
            
            return integrated
        
        # Default integration strategy: combine fields from all responses
        # Start with the first response as the base
        domain = list(responses.keys())[0]
        integrated = responses[domain].copy()
        integrated["source_domains"] = [domain]
        
        # Add contributions from other domains
        for domain, response in list(responses.items())[1:]:
            integrated["source_domains"].append(domain)
            
            for key, value in response.items():
                if key not in integrated:
                    integrated[key] = value
                    integrated[f"{key}_source"] = domain
        
        return integrated 