# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""Vision-Biomechanics Cross-Attention Modules for BioCoach.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VisionSkeletonCrossAttention(nn.Module):

    
    def __init__(
        self,
        vision_dim: int,
        skeleton_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_mode: str = "residual"  # "residual", "gated", "simple"
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.skeleton_dim = skeleton_dim
        self.fusion_mode = fusion_mode
        
        # Auto-adjust num_heads to be compatible with vision_dim
        while vision_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.num_heads = num_heads
        self.head_dim = vision_dim // num_heads
        
        print(f"🔧 VisionSkeletonCrossAttention: vision_dim={vision_dim}, skeleton_dim={skeleton_dim}, num_heads={num_heads}")
        
        # Linear projections for cross attention
        self.q_proj = nn.Linear(vision_dim, vision_dim, bias=False)
        self.k_proj = nn.Linear(skeleton_dim, vision_dim, bias=False)
        self.v_proj = nn.Linear(skeleton_dim, vision_dim, bias=False)
        self.out_proj = nn.Linear(vision_dim, vision_dim, bias=False)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(vision_dim)
        self.norm2 = nn.LayerNorm(vision_dim)
        
        # Fusion components based on mode
        if fusion_mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(vision_dim * 2, vision_dim),
                nn.Sigmoid()
            )
        elif fusion_mode == "residual":
            self.alpha = nn.Parameter(torch.tensor(0.1))  # Learnable residual weight
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
    
    def forward(
        self, 
        vision_features: torch.Tensor,  # [batch, seq_len, vision_dim]
        skeleton_features: torch.Tensor,  # [batch, skeleton_dim] or [batch, 1, skeleton_dim]
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            vision_features: [batch, seq_len, vision_dim]
            skeleton_features: [batch, skeleton_dim] or [batch, 1, skeleton_dim]
            attention_mask: Optional mask for attention
            
        Returns:
            Fused features: [batch, seq_len, vision_dim]
        """
        batch_size, seq_len, _ = vision_features.shape
        
        # Ensure data types match
        target_dtype = vision_features.dtype
        if skeleton_features.dtype != target_dtype:
            skeleton_features = skeleton_features.to(dtype=target_dtype)
        
        # Ensure skeleton features have correct shape
        if skeleton_features.dim() == 2:
            skeleton_features = skeleton_features.unsqueeze(1)  # [batch, 1, skeleton_dim]
        
        # Store original vision features for residual connection
        residual = vision_features
        
        # Apply layer norm
        vision_features = self.norm1(vision_features)
        
        # Cross attention: vision as query, skeleton as key/value
        q = self.q_proj(vision_features)  # [batch, seq_len, vision_dim]
        k = self.k_proj(skeleton_features)  # [batch, 1, vision_dim]
        v = self.v_proj(skeleton_features)  # [batch, 1, vision_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]
        
        # Reshape back
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.vision_dim
        )
        
        # Output projection
        attended_values = self.out_proj(attended_values)
        attended_values = self.dropout(attended_values)
        
        # Apply fusion strategy
        if self.fusion_mode == "simple":
            output = attended_values
        elif self.fusion_mode == "residual":
            output = residual + self.alpha * attended_values
        elif self.fusion_mode == "gated":
            concat_features = torch.cat([residual, attended_values], dim=-1)
            gate_weights = self.gate(concat_features)
            output = gate_weights * attended_values + (1 - gate_weights) * residual
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")
        
        # Final layer norm
        output = self.norm2(output)
        
        return output


class VisionSkeletonBidirectionalAttention(nn.Module):
    """Bidirectional cross-attention for vision and biomechanical features."""
    
    def __init__(
        self,
        vision_dim: int,
        skeleton_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Vision → Skeleton attention
        self.vision_to_skeleton = VisionSkeletonCrossAttention(
            vision_dim, skeleton_dim, num_heads, dropout, fusion_mode="residual"
        )
        
        # Skeleton → Vision attention (reverse)
        self.skeleton_to_vision = VisionSkeletonCrossAttention(
            skeleton_dim, vision_dim, num_heads, dropout, fusion_mode="residual"
        )
        
        # Projection to align skeleton output with vision dimension
        self.skeleton_to_vision_proj = nn.Linear(skeleton_dim, vision_dim)
        
        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Linear(vision_dim * 2, vision_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(vision_dim, vision_dim)
        )
        
    def forward(
        self,
        vision_features: torch.Tensor,  # [batch, seq_len, vision_dim]
        skeleton_features: torch.Tensor  # [batch, skeleton_dim]
    ) -> torch.Tensor:
        """
        Bidirectional cross attention fusion.
        
        Returns:
            Fused vision features: [batch, seq_len, vision_dim]
        """
        # Ensure skeleton features have correct shape for attention
        if skeleton_features.dim() == 2:
            skeleton_features_expanded = skeleton_features.unsqueeze(1)
        else:
            skeleton_features_expanded = skeleton_features
        
        # Vision attended by skeleton
        vision_attended = self.vision_to_skeleton(vision_features, skeleton_features_expanded)
        
        # Skeleton attended by vision (we take mean of vision as single query)
        vision_mean = vision_features.mean(dim=1, keepdim=True)  # [batch, 1, vision_dim]
        skeleton_attended = self.skeleton_to_vision(skeleton_features_expanded, vision_mean)
        
        # Project skeleton output to vision dimension and broadcast to sequence length
        skeleton_projected = self.skeleton_to_vision_proj(skeleton_attended.squeeze(1))  # [batch, vision_dim]
        skeleton_broadcasted = skeleton_projected.unsqueeze(1).expand(-1, vision_features.size(1), -1)
        
        # Final fusion
        concatenated = torch.cat([vision_attended, skeleton_broadcasted], dim=-1)
        fused_output = self.final_fusion(concatenated)
        
        return fused_output


class VisionSkeletonBiomechTriModalFusion(nn.Module):
    """Vision-Biomechanics Conditioned Fusion (Sec 3.5).

    Three-modal fusion: visual appearance, morphometric context,
    and motion quality context.  Implements the full BioCoach
    conditioning pipeline for feedback generation.
    """
    
    def __init__(
        self,
        vision_dim: int,
        skeleton_dim: int,
        biomech_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_mode: str = "residual",
        biomech_weight: float = 0.1
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.skeleton_dim = skeleton_dim
        self.biomech_dim = biomech_dim
        self.fusion_mode = fusion_mode
        self.biomech_weight = biomech_weight
        
        # Auto-adjust num_heads to be compatible with vision_dim
        while vision_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.num_heads = num_heads
        self.head_dim = vision_dim // num_heads
        
        print(f"🔧 VisionSkeletonBiomechTriModalFusion: vision_dim={vision_dim}, skeleton_dim={skeleton_dim}, biomech_dim={biomech_dim}, num_heads={num_heads}")
        
        # Cross attention for vision-skeleton fusion (primary)
        self.vision_skeleton_attn = VisionSkeletonCrossAttention(
            vision_dim, skeleton_dim, num_heads, dropout, fusion_mode
        )
        
        # Biomechanic feedback integration
        # Project biomech features to vision dimension if needed
        if biomech_dim != vision_dim:
            self.biomech_proj = nn.Linear(biomech_dim, vision_dim, bias=False)
        else:
            self.biomech_proj = nn.Identity()
        
        # Attention for biomechanic feedback integration
        self.biomech_q_proj = nn.Linear(vision_dim, vision_dim, bias=False)
        self.biomech_k_proj = nn.Linear(vision_dim, vision_dim, bias=False)
        self.biomech_v_proj = nn.Linear(vision_dim, vision_dim, bias=False)
        self.biomech_out_proj = nn.Linear(vision_dim, vision_dim, bias=False)
        
        # Normalization layers
        self.norm_biomech = nn.LayerNorm(vision_dim)
        self.norm_final = nn.LayerNorm(vision_dim)
        
        # Final fusion weights
        self.vision_weight = nn.Parameter(torch.tensor(0.7))  # Primary modality
        self.skeleton_weight = nn.Parameter(torch.tensor(0.2))  # Secondary modality
        self.biomech_weight_param = nn.Parameter(torch.tensor(biomech_weight))  # Tertiary modality
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in [self.biomech_q_proj, self.biomech_k_proj, self.biomech_v_proj, self.biomech_out_proj]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
    
    def forward(
        self,
        vision_features: torch.Tensor,  # [batch, seq_len, vision_dim]
        skeleton_features: torch.Tensor,  # [batch, skeleton_dim] or [batch, 1, skeleton_dim]  
        biomech_features: Optional[torch.Tensor] = None,  # [batch, biomech_dim] or None
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            vision_features: [batch, seq_len, vision_dim]
            skeleton_features: [batch, skeleton_dim] or [batch, 1, skeleton_dim]
            biomech_features: [batch, biomech_dim] or None for empty feedback
            attention_mask: Optional mask for attention
            
        Returns:
            Fused features: [batch, seq_len, vision_dim]
        """
        batch_size, seq_len, _ = vision_features.shape
        
        # Step 1: Vision-Skeleton cross attention (primary fusion)
        vision_skeleton_fused = self.vision_skeleton_attn(vision_features, skeleton_features, attention_mask)
        
        # Step 2: Integrate biomechanic feedback if available
        if biomech_features is not None and torch.any(biomech_features != 0):
            # Project biomech features to vision dimension
            biomech_projected = self.biomech_proj(biomech_features)  # [batch, vision_dim]
            
            # Ensure biomech features have correct shape for attention
            if biomech_projected.dim() == 2:
                biomech_projected = biomech_projected.unsqueeze(1)  # [batch, 1, vision_dim]
            
            # Cross attention: vision-skeleton as query, biomech as key/value
            residual = vision_skeleton_fused
            q_fused = self.biomech_q_proj(vision_skeleton_fused)  # [batch, seq_len, vision_dim]
            k_biomech = self.biomech_k_proj(biomech_projected)  # [batch, 1, vision_dim]
            v_biomech = self.biomech_v_proj(biomech_projected)  # [batch, 1, vision_dim]
            
            # Reshape for multi-head attention
            q_fused = q_fused.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k_biomech = k_biomech.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            v_biomech = v_biomech.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Compute attention scores
            scores = torch.matmul(q_fused, k_biomech.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            # Apply softmax
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            biomech_attended = torch.matmul(attn_weights, v_biomech)  # [batch, num_heads, seq_len, head_dim]
            
            # Reshape back
            biomech_attended = biomech_attended.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.vision_dim
            )
            
            # Output projection
            biomech_attended = self.biomech_out_proj(biomech_attended)
            biomech_attended = self.dropout(biomech_attended)
            
            # Residual connection with smaller weight for biomech feedback
            vision_skeleton_fused = residual + self.biomech_weight_param * biomech_attended
            vision_skeleton_fused = self.norm_biomech(vision_skeleton_fused)
        
        # Step 3: Final normalization
        output = self.norm_final(vision_skeleton_fused)
        
        return output


def create_vision_skeleton_fusion_module(
    vision_dim: int,
    skeleton_dim: int,
    fusion_type: str = "cross_attention",
    num_heads: int = 8,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """Factory function to create vision-skeleton fusion modules.
    
    Args:
        vision_dim: Dimension of vision features
        skeleton_dim: Dimension of skeleton features  
        fusion_type: Type of fusion ("cross_attention", "bidirectional", "simple_add", "trimodal")
        num_heads: Number of attention heads
        dropout: Dropout rate
        
    Returns:
        Fusion module
    """
    if fusion_type == "cross_attention":
        return VisionSkeletonCrossAttention(
            vision_dim, skeleton_dim, num_heads, dropout, 
            fusion_mode=kwargs.get("fusion_mode", "residual")
        )
    elif fusion_type == "bidirectional":
        return VisionSkeletonBidirectionalAttention(
            vision_dim, skeleton_dim, num_heads, dropout
        )
    elif fusion_type == "trimodal":
        biomech_dim = kwargs.get("biomech_dim", vision_dim)
        biomech_weight = kwargs.get("biomech_weight", 0.1)
        return VisionSkeletonBiomechTriModalFusion(
            vision_dim, skeleton_dim, biomech_dim, num_heads, dropout,
            fusion_mode=kwargs.get("fusion_mode", "residual"),
            biomech_weight=biomech_weight
        )
    elif fusion_type == "simple_add":
        # Simple projection + addition (current approach)
        class SimpleAddFusion(nn.Module):
            def __init__(self, vision_dim, skeleton_dim):
                super().__init__()
                self.proj = nn.Linear(skeleton_dim, vision_dim, bias=False)
                
            def forward(self, vision_features, skeleton_features):
                if skeleton_features.dim() == 2:
                    skeleton_features = skeleton_features.unsqueeze(1)
                skeleton_projected = self.proj(skeleton_features)
                return vision_features + skeleton_projected.expand_as(vision_features)
        
        return SimpleAddFusion(vision_dim, skeleton_dim)
    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type}")



VisionMorphometricCrossAttention = VisionSkeletonCrossAttention


VisionBiomechanicsConditionedFusion = VisionSkeletonBiomechTriModalFusion
