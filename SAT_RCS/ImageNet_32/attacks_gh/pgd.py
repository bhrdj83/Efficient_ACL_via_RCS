import torch
import torch.nn.functional as F
from typing import Dict, Any
from .base_attack import BaseAttack

class PGD(BaseAttack):
    """Projected Gradient Descent (PGD) attack"""
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        self.steps = config.get('steps', 10)
        self.alpha = config.get('alpha', 2.0) / 255.0  # Convert from 0-255 to 0-1
        self.random_start = config.get('random_start', True)
        self.targeted = config.get('targeted', False)
        self.norm_type = config.get('norm_type', 'inf')
    
    def attack(self, images: torch.Tensor, labels: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        """Generate PGD adversarial examples"""
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        # Initialize adversarial examples
        if self.random_start:
            # Random initialization within epsilon ball
            if self.norm_type == 'inf':
                delta = torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
            elif self.norm_type == '2':
                delta = torch.randn_like(images)
                delta = self.project_perturbation(delta, self.norm_type)
            else:
                raise ValueError(f"Unsupported norm type: {self.norm_type}")
            
            adv_images = self.clamp(images + delta)
        else:
            adv_images = images.clone()
        
        # PGD iterations
        for step in range(self.steps):
            adv_images.requires_grad = True
            
            # Forward pass
            outputs = self.model(adv_images)
            
            # Compute loss
            if self.targeted:
                if targets is None:
                    raise ValueError("Target labels must be provided for targeted attacks.")
                loss = F.cross_entropy(outputs, targets)
            else:
                loss = F.cross_entropy(outputs, labels)

            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update adversarial examples
            with torch.no_grad():
                grad = adv_images.grad.data
                
                if self.targeted:
                    # Targeted attack: move towards target class
                    adv_images = adv_images - self.alpha * grad.sign()
                else:
                    # Untargeted attack: move away from true class
                    adv_images = adv_images + self.alpha * grad.sign()
                
                # Project perturbation
                delta = adv_images - images
                delta = self.project_perturbation(delta, self.norm_type)
                adv_images = self.clamp(images + delta)
        
        return adv_images.detach()
    
    def attack_batch(self, images: torch.Tensor, labels: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        """Attack a batch of images"""
        return self.attack(images, labels, targets)
