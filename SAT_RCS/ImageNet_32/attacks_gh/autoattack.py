import torch
from autoattack import AutoAttack as AP
from .base_attack import BaseAttack
from typing import Dict, Any

class AutoAttack(BaseAttack):
    """
    Wrapper for the AutoAttack library.
    AutoAttack is a parameter-free, computationally expensive, but very reliable
    method for evaluating adversarial robustness.
    """
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        self.norm = config.get('norm', 'Linf')
        self.eps = self.epsilon / 255.0  # AutoAttack expects epsilon in [0, 1] range
        
        # Initialize AutoAttack
        # Note: AutoAttack manages its own device handling based on the model's device
        self.attacker_instance = AP(self.model, norm=self.norm, eps=self.eps, version='standard', verbose=False)

    def attack(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Run AutoAttack on a batch of data.
        
        Note: The original implementation of AutoAttack does not require targets
        for untargeted attacks, but we accept it to conform to the BaseAttack interface.
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # AutoAttack's run_standard_evaluation method handles everything.
        # It takes clean images and their true labels.
        x_adv = self.attacker_instance.run_standard_evaluation(data, target)
        
        return x_adv