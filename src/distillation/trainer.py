from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable

from ..core import DomainExpert


class Trainer(ABC):
    """
    Abstract base class for training student models through knowledge distillation.
    """
    
    @abstractmethod
    def train(self, model: DomainExpert, examples: List[Dict[str, Any]]) -> None:
        """
        Train a model using the provided examples.
        
        Args:
            model: The model to train
            examples: Training examples
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """
        Save a training checkpoint.
        
        Args:
            path: Path to save the checkpoint
        """
        pass


class IntegrationTrainer(Trainer):
    """
    Two-phase trainer that first fine-tunes on individual domains, then on cross-domain integration.
    
    This approach is particularly effective for creating models that maintain domain-specific 
    expertise while also developing cross-domain integration capabilities.
    """
    
    def __init__(
        self,
        learning_rate: float = 2e-5,
        epochs: int = 3,
        batch_size: int = 16,
        evaluation_steps: int = 500,
        checkpoint_dir: Optional[str] = "./checkpoints",
        optimizer_class: Optional[Any] = None,
        scheduler_class: Optional[Any] = None,
        callbacks: Optional[List[Callable]] = None
    ):
        """
        Initialize an IntegrationTrainer.
        
        Args:
            learning_rate: Learning rate for fine-tuning
            epochs: Number of training epochs
            batch_size: Batch size for training
            evaluation_steps: Number of steps between evaluations
            checkpoint_dir: Directory to save checkpoints
            optimizer_class: Optional custom optimizer class
            scheduler_class: Optional custom scheduler class
            callbacks: Optional list of callback functions
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.evaluation_steps = evaluation_steps
        self.checkpoint_dir = checkpoint_dir
        self.optimizer_class = optimizer_class
        self.scheduler_class = scheduler_class
        self.callbacks = callbacks or []
        self.current_step = 0
    
    def train(self, model: DomainExpert, examples: List[Dict[str, Any]]) -> None:
        """
        Train a model using integration-focused fine-tuning.
        
        Args:
            model: The model to train
            examples: Training examples with teacher responses
        """
        # Phase 1: Domain-specific fine-tuning
        domain_specific_examples = self._filter_domain_specific_examples(examples)
        self._train_phase(model, domain_specific_examples, "domain_specific")
        
        # Phase 2: Cross-domain integration fine-tuning
        cross_domain_examples = self._filter_cross_domain_examples(examples)
        self._train_phase(model, cross_domain_examples, "cross_domain")
    
    def _filter_domain_specific_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter examples to include only those that focus on a single domain.
        
        Args:
            examples: All training examples
            
        Returns:
            Domain-specific examples
        """
        return [ex for ex in examples if len(ex.get("domains", [])) == 1]
    
    def _filter_cross_domain_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter examples to include only those that span multiple domains.
        
        Args:
            examples: All training examples
            
        Returns:
            Cross-domain examples
        """
        return [ex for ex in examples if len(ex.get("domains", [])) > 1]
    
    def _train_phase(self, model: DomainExpert, examples: List[Dict[str, Any]], phase_name: str) -> None:
        """
        Train the model for a specific phase.
        
        Args:
            model: The model to train
            examples: Training examples for this phase
            phase_name: Name of the training phase
        """
        print(f"Starting {phase_name} training phase with {len(examples)} examples")
        
        # In a real implementation, this would involve:
        # 1. Setting up the appropriate optimizer and scheduler
        # 2. Converting examples to the format required by the model
        # 3. Training for the specified number of epochs
        # 4. Evaluating periodically and saving checkpoints
        
        # This is a placeholder implementation since the actual training depends on the model type
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            # Placeholder for the actual training loop
            for i in range(0, len(examples), self.batch_size):
                batch = examples[i:i+self.batch_size]
                self._train_batch(model, batch)
                self.current_step += 1
                
                if self.current_step % self.evaluation_steps == 0:
                    # Evaluate and save checkpoint
                    self.save_checkpoint(f"{self.checkpoint_dir}/{phase_name}_step_{self.current_step}")
                    
                    # Call callbacks
                    for callback in self.callbacks:
                        callback(model, self.current_step, phase_name)
    
    def _train_batch(self, model: DomainExpert, batch: List[Dict[str, Any]]) -> None:
        """
        Train the model on a single batch of examples.
        
        Args:
            model: The model to train
            batch: Batch of training examples
        """
        # This is a placeholder for the actual batch training logic
        # In a real implementation, this would:
        # 1. Format the inputs and targets
        # 2. Compute the forward pass
        # 3. Calculate the loss
        # 4. Perform backpropagation and optimization
        
        print(f"Training batch of size {len(batch)}")
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save a training checkpoint.
        
        Args:
            path: Path to save the checkpoint
        """
        # This is a placeholder for the actual checkpoint saving logic
        print(f"Saved checkpoint to {path}")


class MultiTaskTrainer(Trainer):
    """
    Trains the student model on all domains simultaneously.
    
    This approach mixes examples from different domains in each batch,
    allowing the model to learn from diverse domain knowledge concurrently.
    """
    
    def __init__(
        self,
        learning_rate: float = 2e-5,
        epochs: int = 3,
        batch_size: int = 16,
        evaluation_steps: int = 500,
        checkpoint_dir: Optional[str] = "./checkpoints"
    ):
        """
        Initialize a MultiTaskTrainer.
        
        Args:
            learning_rate: Learning rate for fine-tuning
            epochs: Number of training epochs
            batch_size: Batch size for training
            evaluation_steps: Number of steps between evaluations
            checkpoint_dir: Directory to save checkpoints
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.evaluation_steps = evaluation_steps
        self.checkpoint_dir = checkpoint_dir
        self.current_step = 0
    
    def train(self, model: DomainExpert, examples: List[Dict[str, Any]]) -> None:
        """
        Train a model using multi-task fine-tuning.
        
        Args:
            model: The model to train
            examples: Training examples with teacher responses
        """
        print(f"Starting multi-task training with {len(examples)} examples")
        
        # Shuffle examples to ensure domain diversity in batches
        import random
        random.shuffle(examples)
        
        # Train for the specified number of epochs
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            # Process examples in batches
            for i in range(0, len(examples), self.batch_size):
                batch = examples[i:i+self.batch_size]
                self._train_batch(model, batch)
                self.current_step += 1
                
                if self.current_step % self.evaluation_steps == 0:
                    # Evaluate and save checkpoint
                    self.save_checkpoint(f"{self.checkpoint_dir}/step_{self.current_step}")
    
    def _train_batch(self, model: DomainExpert, batch: List[Dict[str, Any]]) -> None:
        """
        Train the model on a single batch of examples.
        
        Args:
            model: The model to train
            batch: Batch of training examples
        """
        # This is a placeholder for the actual batch training logic
        print(f"Training batch of size {len(batch)}")
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save a training checkpoint.
        
        Args:
            path: Path to save the checkpoint
        """
        # This is a placeholder for the actual checkpoint saving logic
        print(f"Saved checkpoint to {path}")


class SequentialTrainer(Trainer):
    """
    Trains the student model on each domain sequentially.
    
    This approach fine-tunes the model on each domain one after another,
    starting with the most general domain and progressing to more specialized ones.
    """
    
    def __init__(
        self,
        learning_rate: float = 2e-5,
        epochs_per_domain: int = 2,
        batch_size: int = 16,
        evaluation_steps: int = 500,
        checkpoint_dir: Optional[str] = "./checkpoints",
        domain_order: Optional[List[str]] = None
    ):
        """
        Initialize a SequentialTrainer.
        
        Args:
            learning_rate: Learning rate for fine-tuning
            epochs_per_domain: Number of epochs to train on each domain
            batch_size: Batch size for training
            evaluation_steps: Number of steps between evaluations
            checkpoint_dir: Directory to save checkpoints
            domain_order: Optional ordered list of domains (from general to specialized)
        """
        self.learning_rate = learning_rate
        self.epochs_per_domain = epochs_per_domain
        self.batch_size = batch_size
        self.evaluation_steps = evaluation_steps
        self.checkpoint_dir = checkpoint_dir
        self.domain_order = domain_order
        self.current_step = 0
    
    def train(self, model: DomainExpert, examples: List[Dict[str, Any]]) -> None:
        """
        Train a model using sequential fine-tuning.
        
        Args:
            model: The model to train
            examples: Training examples with teacher responses
        """
        # Group examples by their primary domain
        domain_examples = {}
        for example in examples:
            primary_domain = example.get("primary_domain")
            if primary_domain:
                if primary_domain not in domain_examples:
                    domain_examples[primary_domain] = []
                domain_examples[primary_domain].append(example)
        
        # Determine domain order if not specified
        domains = self.domain_order or list(domain_examples.keys())
        
        # Train on each domain sequentially
        for domain in domains:
            if domain not in domain_examples:
                print(f"Warning: No examples found for domain '{domain}'")
                continue
            
            domain_batch = domain_examples[domain]
            print(f"Training on domain '{domain}' with {len(domain_batch)} examples")
            
            # Train for the specified number of epochs
            for epoch in range(self.epochs_per_domain):
                print(f"Domain '{domain}' - Epoch {epoch+1}/{self.epochs_per_domain}")
                
                # Process examples in batches
                for i in range(0, len(domain_batch), self.batch_size):
                    batch = domain_batch[i:i+self.batch_size]
                    self._train_batch(model, batch)
                    self.current_step += 1
                    
                    if self.current_step % self.evaluation_steps == 0:
                        # Evaluate and save checkpoint
                        self.save_checkpoint(f"{self.checkpoint_dir}/{domain}_step_{self.current_step}")
    
    def _train_batch(self, model: DomainExpert, batch: List[Dict[str, Any]]) -> None:
        """
        Train the model on a single batch of examples.
        
        Args:
            model: The model to train
            batch: Batch of training examples
        """
        # This is a placeholder for the actual batch training logic
        print(f"Training batch of size {len(batch)}")
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save a training checkpoint.
        
        Args:
            path: Path to save the checkpoint
        """
        # This is a placeholder for the actual checkpoint saving logic
        print(f"Saved checkpoint to {path}")
