import torch
import torch.nn.functional as F
from typing import Dict, Any
from .base_attack import BaseAttack

class FGSM(BaseAttack):
    """Fast Gradient Sign Method (FGSM) attack"""
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        self.targeted = config.get('targeted', False)
    
    def attack(self, images: torch.Tensor, labels: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        """Generate FGSM adversarial examples"""
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        # Enable gradient computation for input
        images.requires_grad = True
        
        # Forward pass
        outputs = self.model(images)
        
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
        
        # Generate adversarial examples
        data_grad = images.grad.data
        
        if self.targeted:
            # Targeted attack: move towards target class
            sign_data_grad = -data_grad.sign()
        else:
            # Untargeted attack: move away from true class
            sign_data_grad = data_grad.sign()
        
        # Create adversarial examples
        perturbed_images = images + self.epsilon * sign_data_grad
        
        # Clamp to valid range
        perturbed_images = self.clamp(perturbed_images)
        
        return perturbed_images.detach()
    
    def attack_batch(self, images: torch.Tensor, labels: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        """Attack a batch of images"""
        return self.attack(images, labels, targets)
