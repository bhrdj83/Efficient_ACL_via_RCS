import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from .base_attack import BaseAttack

class CW(BaseAttack):
    """Carlini & Wagner (C&W) attack"""
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        self.steps = config.get('steps', 100)
        self.lr = config.get('lr', 0.01)
        self.c = config.get('c', 1.0)
        self.kappa = config.get('kappa', 0)
        self.targeted = config.get('targeted', False)
        self.binary_search_steps = config.get('binary_search_steps', 5)
    
    def attack(self, images: torch.Tensor, labels: torch.Tensor, 
              targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate C&W adversarial examples"""
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        if self.targeted and targets is None:
            raise ValueError("Target labels required for targeted C&W attack")
        
        batch_size = images.size(0)
        
        # Binary search bounds
        c_min = torch.zeros(batch_size, device=self.device)
        c_max = torch.full((batch_size,), 1e10, device=self.device)
        
        # Best adversarial examples found so far
        best_adv = images.clone()
        best_l2 = torch.full((batch_size,), float('inf'), device=self.device)
        
        # Binary search on c
        for binary_step in range(self.binary_search_steps):
            # Current c values
            current_c = (c_min + c_max) / 2
            
            # Optimize for current c values
            adv_images, l2_dists, success = self._optimize(
                images, labels, current_c, targets
            )
            
            # Update binary search bounds and best examples
            for i in range(batch_size):
                if success[i] and l2_dists[i] < best_l2[i]:
                    best_l2[i] = l2_dists[i]
                    best_adv[i] = adv_images[i]
                    c_max[i] = current_c[i]
                else:
                    c_min[i] = current_c[i]
        
        return best_adv.detach()
    
    def _optimize(self, images: torch.Tensor, labels: torch.Tensor, 
                 c_values: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """Optimize for given c values"""
        batch_size = images.size(0)
        
        # Initialize delta in tanh space
        w = torch.zeros_like(images, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=self.lr)
        
        best_adv = images.clone()
        best_l2 = torch.full((batch_size,), float('inf'), device=self.device)
        
        for step in range(self.steps):
            # Convert from tanh space to image space
            adv_images = (torch.tanh(w) + 1) / 2
            
            # Compute L2 distance
            l2_dist = torch.norm((adv_images - images).view(batch_size, -1), 
                               p=2, dim=1)
            
            # Get logits
            logits = self.model(adv_images)
            
            # Compute f function for C&W loss
            if self.targeted:
                if targets is None:
                    raise ValueError("Targets required for targeted attack")
                target_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
                other_logits = logits.scatter(1, targets.unsqueeze(1), -float('inf')).max(1)[0]
                f = other_logits - target_logits
            else:
                true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
                other_logits = logits.scatter(1, labels.unsqueeze(1), -float('inf')).max(1)[0]
                f = true_logits - other_logits
            
            # C&W loss
            loss1 = l2_dist
            loss2 = torch.clamp(f + self.kappa, min=0)
            loss = loss1 + c_values * loss2
            
            # Backward pass
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            
            # Update best examples
            for i in range(batch_size):
                if loss2[i] <= 0 and l2_dist[i] < best_l2[i]:
                    best_l2[i] = l2_dist[i]
                    best_adv[i] = adv_images[i].detach()
        
        # Check success
        final_logits = self.model(best_adv)
        if self.targeted:
            success = final_logits.argmax(dim=1) == targets
        else:
            success = final_logits.argmax(dim=1) != labels
        
        return best_adv, best_l2, success
    
    def attack_batch(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Attack a batch of images"""
        return self.attack(images, labels)
