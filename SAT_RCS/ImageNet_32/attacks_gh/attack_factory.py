from typing import Dict, Any
import torch
from .base_attack import BaseAttack
from .fgsm import FGSM
from .pgd import PGD
from .cw import CW
from .autoattack import AutoAttack

class AttackFactory:
    """Factory class for creating attacks"""
    
    _attacks = {
        'fgsm': FGSM,
        'pgd': PGD,
        'cw': CW,
        'autoattack': AutoAttack,
    }
    
    @classmethod
    def create_attack(cls, attack_name: str, model: torch.nn.Module, 
                     config: Dict[str, Any]) -> BaseAttack:
        """Create attack instance"""
        if attack_name not in cls._attacks:
            available = ', '.join(cls._attacks.keys())
            raise ValueError(f"Unknown attack: {attack_name}. Available: {available}")
        
        attack_class = cls._attacks[attack_name]
        return attack_class(model, config)
    
    @classmethod
    def list_available_attacks(cls) -> list:
        """List available attacks"""
        return list(cls._attacks.keys())
    
    @classmethod
    def register_attack(cls, name: str, attack_class: type):
        """Register a new attack class"""
        cls._attacks[name] = attack_class
