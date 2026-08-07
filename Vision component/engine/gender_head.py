"""
engine/gender_head.py
─────────────────────
Defines the full ViT-B/16 + 2-class MLP head.
The backbone comes from the `timm` library (PyTorch Image Models).
"""

import torch
import torch.nn as nn
import timm

class GenderClassifier(nn.Module):
    """
    ViT-B/16 backbone with a custom 2-class gender classification head.
    Architecture:
        - Backbone: vit_base_patch16_224 pretrained on ImageNet-21k (via timm)
        - removed original classification head (num_classes=0)
        - added custom MLP head for gender classification
        - Head: LayerNorm → Linear(768,256) → GELU → Dropout(0.3) → Linear(256,2)
        - Output: raw logits of shape (batch_size, 2)
    """

    BACKBONE_NAME = "vit_base_patch16_224"
    EMBED_DIM = 768
    HIDDEN_DIM = 256
    NUM_CLASSES = 2
    DROPOUT = 0.3

    def __init__(self, pretrained_backbone: bool = False):
        """
        Args:
            pretrained_backbone: If True, loads ImageNet-21k weights for the backbone.
                                 Set True during training, False during inference
                                 (weights loaded via load_state_dict instead).
        """
        super().__init__()

        # Load backbone, removing its original classification head
        self.backbone = timm.create_model(
            self.BACKBONE_NAME,
            pretrained=pretrained_backbone,
            num_classes=0,      # removes the timm head; output is CLS token embedding
            global_pool='token' # returns CLS token (not global average pool)
        )

        # add a custom 2-class MLP head for gender classification 
        self.head = nn.Sequential(
            nn.LayerNorm(self.EMBED_DIM),  # LayerNorm for stabilizing training
            nn.Linear(self.EMBED_DIM, self.HIDDEN_DIM), # Linear layer to reduce embedding dimension from 768 to 256
            nn.GELU(), # GELU activation for non-linearity
            nn.Dropout(p=self.DROPOUT), # Dropout for regularization to prevent overfitting
            nn.Linear(self.HIDDEN_DIM, self.NUM_CLASSES), # Final linear layer to output logits for 2 classes (female, male)
        )

        self._init_head_weights()

    # Initialize head weights with Xavier uniform for linear layers and zeros for biases instead of random initialization. 
    def _init_head_weights(self):
        """Xavier uniform for linear layers, zeros for biases."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) 
                nn.init.zeros_(m.bias)

    # Forward pass neural network
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor of shape (B, 3, 224, 224), ImageNet-normalized
        Returns:
            Logits of shape (B, 2)  — [female_logit, male_logit]
        """
        # Forward pass through the backbone and head
        features = self.backbone(x)   # (B, 768) CLS token
        logits = self.head(features)  # (B, 2)
        return logits

    # Freeze method for discriminative learning rates during fine-tuning
    def freeze_backbone(self):
        """Freeze backbone parameters. Used in Phase 1 of training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    # Unfreeze method for discriminative learning rates during fine-tuning
    def unfreeze_backbone(self, layers_from_end: int = 4):
        """
        Selectively unfreeze the last N transformer blocks.
        Used in Phase 2 of fine-tuning (discriminative learning rates).
        Args:
            layers_from_end: Number of ViT blocks from the end to unfreeze (default: 4)
        """
        if hasattr(self.backbone, 'blocks'):
            blocks = list(self.backbone.blocks)
            for block in blocks[-layers_from_end:]:
                for param in block.parameters():
                    param.requires_grad = True
        
        # Always unfreeze the final norm
        if hasattr(self.backbone, 'norm'):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
