from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Optional

class BaseAttack(ABC):
    """Abstract base class for adversarial attacks"""
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.epsilon = config.get('epsilon', 8.0) / 255.0  # Convert from 0-255 to 0-1
        self.device = next(model.parameters()).device
    
    @abstractmethod
    def attack(self, images: torch.Tensor, labels: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate adversarial examples"""
        pass
    
    def clamp(self, images: torch.Tensor) -> torch.Tensor:
        """Clamp images to valid range [0, 1]"""
        return torch.clamp(images, 0, 1)
    
    def project_perturbation(self, perturbation: torch.Tensor, 
                           norm_type: str = 'inf') -> torch.Tensor:
        """Project perturbation to satisfy constraint"""
        if norm_type == 'inf':
            return torch.clamp(perturbation, -self.epsilon, self.epsilon)
        elif norm_type == '2':
            # L2 projection
            batch_size = perturbation.shape[0]
            perturbation_flat = perturbation.view(batch_size, -1)
            norm = torch.norm(perturbation_flat, p=2, dim=1)
            scale = torch.min(self.epsilon / norm, torch.ones_like(norm))
            return perturbation * scale.view(-1, 1, 1, 1)
        else:
            raise ValueError(f"Unsupported norm type: {norm_type}")
    
    def get_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Get model logits"""
        self.model.eval()
        with torch.no_grad():
            return self.model(images)
    
    def get_loss(self, images: torch.Tensor, labels: torch.Tensor, 
                targeted: bool = False) -> torch.Tensor:
        """Compute loss for attack"""
        logits = self.model(images)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        loss = loss_fn(logits, labels)
        
        if targeted:
            return -loss  # Minimize loss for targeted attacks
        else:
            return loss   # Maximize loss for untargeted attacks
