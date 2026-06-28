
import math
import torch
import torch.nn as nn
from typing import Optional

class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, dropout):
        """
        task_dims: output dimension per task
        Example: task_dims=[1, 1, 5] means 3 tasks
             Task 1: regression/binary classification (1 output)
             Task 2: regression/binary classification (1 output)
             Task 3: 5-class classification (5 outputs)
        """
        super().__init__()
        
        # Shared backbone
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h)) 
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        
        # Multi-task heads: one linear layer per task
        self.heads = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(output_dim)])
        
    def forward(self, x):
        shared_features = self.backbone(x)
        
        # Generate all task outputs via list comprehension, then concatenate
        outputs = [head(shared_features) for head in self.heads]
        out = torch.cat(outputs, dim=-1)
        
        return out