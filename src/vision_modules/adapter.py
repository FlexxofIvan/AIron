# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""Vision Backbone to LLM Cross-attention Adapter Implementation."""

from typing import Any, Callable, Optional, Union

import torch
from torch import nn


class XAttnAdapter(nn.Module):
    """
    Cross-attention adapter between the vision and language backbones.

    :param model_lang_embed_tokens_layer:
        The token embedding layer object from language backbone
    :param injection_layer_ids:
        Layer number(s) of the language model where the vision features should be cross attended to
    """

    def __init__(
        self,
        model_lang_embed_tokens_layer: Callable,
        injection_layer_ids: list[int | str],
    ):
        super().__init__()
        # All layer ids must be strings
        injection_layer_ids = list(map(str, injection_layer_ids))

        self.injection_layer_ids = injection_layer_ids
        self.embed_tokens = model_lang_embed_tokens_layer
        
        # Dynamic biomechanic feedback context
        self.biomech_context = None          # single context (inference path)
        self.biomech_context_spans = None    # per-span contexts (training path):
                                             #   list of (tok_start, tok_end, context_str)
        self.biomech_weight = 0.1  # Small weight for biomech information

    def _embed_text_tokens(self, text_tokens: torch.tensor) -> dict[str, torch.tensor]:
        """Return a tensor containing embeddings of tokenized text (input ids).

        Enhanced to support dynamic biomechanic feedback integration at embedding level.
        """
        base_embeddings = self.embed_tokens(text_tokens)

        # Training path: per-feedback-span contexts (each span gets its own context).
        if self.biomech_context_spans:
            enhanced_embeddings = self._fuse_biomech_context_spans(base_embeddings, text_tokens)
            return {"0": enhanced_embeddings}

        # Inference path: single global context (unchanged behavior).
        if self.biomech_context is not None and self.biomech_context.strip():
            enhanced_embeddings = self._fuse_biomech_context(base_embeddings, text_tokens)
            return {"0": enhanced_embeddings}

        return {"0": base_embeddings}

    def _pool_context_string(self, context_str: str, device) -> "torch.tensor | None":
        """Tokenize + embed + mean-pool a biomech context string to one [hidden_dim] vector."""
        if not context_str or not context_str.strip():
            return None
        if not (hasattr(self, '_tokenizer_ref') and self._tokenizer_ref is not None):
            return None
        try:
            toks = self._tokenizer_ref.encode(context_str, add_special_tokens=False)
        except Exception:
            return None
        if not toks:
            return None
        emb = self.embed_tokens(torch.tensor(toks, device=device))  # [L, hidden]
        return emb.mean(dim=0)                                       # [hidden]

    def _fuse_biomech_context_spans(self, base_embeddings: torch.tensor,
                                    text_tokens: torch.tensor) -> torch.tensor:
        """Training-time fusion: each feedback span gets its own pooled biomech context.

        ``self.biomech_context_spans`` = list of (tok_start, tok_end, context_str).
        Each context is fused (weight ``biomech_weight``) into the token positions
        [tok_start, tok_end) of the input embeddings, mirroring how inference fuses
        the per-feedback context into that feedback's tokens.
        """
        try:
            device = base_embeddings.device
            _, seq_len, _ = base_embeddings.shape
            enhanced = base_embeddings.clone()
            for span in self.biomech_context_spans:
                start, end, ctx_str = span
                pooled = self._pool_context_string(ctx_str, device)
                if pooled is None:
                    continue
                s = max(0, int(start))
                e = min(seq_len, int(end))
                if e <= s:
                    continue
                enhanced[:, s:e, :] = (
                    (1.0 - self.biomech_weight) * base_embeddings[:, s:e, :]
                    + self.biomech_weight * pooled
                )
            return enhanced
        except Exception as exc:
            print(f"⚠️ Error in per-span biomech fusion: {exc}")
            return base_embeddings
    
    def _fuse_biomech_context(self, base_embeddings: torch.tensor, text_tokens: torch.tensor) -> torch.tensor:
        """Fuse biomechanic feedback context into system prompt embeddings.
        
        Args:
            base_embeddings: Original text embeddings [batch, seq_len, hidden_dim]
            text_tokens: Token IDs [batch, seq_len]
            
        Returns:
            Enhanced embeddings with biomech context fused into system prompt positions
        """
        try:
            # Import here to avoid circular import
            import torch
            
            device = base_embeddings.device
            batch_size, seq_len, hidden_dim = base_embeddings.shape
            
            # Identify <vision> token positions to find system prompt region
            vision_token_id = getattr(self, '_vision_token_id', None)
            if vision_token_id is None:
                # Try to find vision token ID from common values
                for potential_id in [32000, 32001, 32002]:  # Common special token IDs
                    if potential_id in text_tokens:
                        vision_token_id = potential_id
                        self._vision_token_id = potential_id
                        break
            
            if vision_token_id is not None:
                # System prompt positions are all non-vision tokens
                system_positions = torch.where(text_tokens != vision_token_id)
            else:
                # Fallback: assume first 80% of tokens are system prompt
                system_end = int(seq_len * 0.8)
                system_positions = (torch.arange(system_end, device=device).unsqueeze(0).expand(batch_size, -1),)
            
            # Encode biomech feedback context
            biomech_tokens = []
            try:
                # Use tokenizer if available
                if hasattr(self, '_tokenizer_ref') and self._tokenizer_ref is not None:
                    biomech_tokens = self._tokenizer_ref.encode(
                        self.biomech_context, add_special_tokens=False
                    )
                else:
                    # Skip tokenization if tokenizer not available
                    print("⚠️ Tokenizer not available for biomech context encoding")
                    return base_embeddings
            except:
                print("⚠️ Error tokenizing biomech context, using base embeddings")
                return base_embeddings
            
            if not biomech_tokens:
                return base_embeddings
            
            # Get biomech embeddings
            biomech_tensor = torch.tensor(biomech_tokens, device=device)
            biomech_embeddings = self.embed_tokens(biomech_tensor)  # [biomech_len, hidden_dim]
            
            # Pool biomech embeddings to single vector
            biomech_pooled = biomech_embeddings.mean(dim=0)  # [hidden_dim]
            
            # Create enhanced embeddings
            enhanced_embeddings = base_embeddings.clone()
            
            # Fuse biomech context into system prompt positions with small weight
            for batch_idx in range(batch_size):
                if len(system_positions) > batch_idx:
                    pos_indices = system_positions[batch_idx] if len(system_positions) > 1 else system_positions[0]
                    for pos in pos_indices:
                        if pos < seq_len:
                            enhanced_embeddings[batch_idx, pos, :] = (
                                (1.0 - self.biomech_weight) * base_embeddings[batch_idx, pos, :] +
                                self.biomech_weight * biomech_pooled
                            )

            return enhanced_embeddings
            
        except Exception as e:
            print(f"⚠️ Error in biomech context fusion: {e}")
            return base_embeddings

    def _combine_multilayer_embedding_dicts(
        self,
        multilayer_video_feats: dict[str, torch.tensor],
        text_tokens: torch.tensor,
        multilayer_embedded_text_tokens: dict[str, torch.tensor],
    ) -> dict[str, dict[str, Any] | dict[str, Any] | Any]:
        """
        Prepare a combined dictionary with both visual and textual inputs.

        :param multilayer_video_feats:
            Dictionary of tensors with different layers' vision feats
        :param text_tokens:
            Tensor of shape [batch_size, sequence_length].
        :param multilayer_embedded_text_tokens:
            Dictionary of tensors with different layers' text embeddings.

        :return:
            Multilayer dictionary with the combined features for all layers.
        """

        final_multilayer_embeddings = {
            "text_tokens": text_tokens,
            "0": {"comb": multilayer_embedded_text_tokens["0"]},
        }

        # Update dict with vision features from each layer
        for layer_id in self.injection_layer_ids:
            final_multilayer_embeddings[layer_id] = {"vision": multilayer_video_feats[layer_id]}

        return final_multilayer_embeddings

    def _init_multilayer_vision_feats_dict(
        self, vision_feats: torch.tensor
    ) -> dict[str, torch.tensor]:
        """Initialize multi-layer vision features dict"""
        multilayer_vision_feats = {}
        for layer_id in self.injection_layer_ids:
            if layer_id == "0":
                multilayer_vision_feats[layer_id] = vision_feats.mean(dim=2)
            else:
                multilayer_vision_feats[layer_id] = vision_feats
        return multilayer_vision_feats

    def forward(
        self,
        vision_feats: torch.tensor,
        text_tokens: torch.tensor,
        vision_xattn_mask: Optional[torch.tensor] = None,
        buffer_xattn_mask: Optional[torch.tensor] = None,
    ) -> dict[str, Union[str, torch.tensor, None]]:
        """
        :param vision_feats:
            Vision features extracted from a vision encoder.
        :param text_tokens:
            Tokenized text
        :param vision_xattn_mask:
            Mask indicating which parts of the vision features are valid
        :param buffer_xattn_mask:
            Buffer mask that indicates which elements are part of the same utterance

        :return:
            embedding dict
        """
        vision_feats = vision_feats["feats"] if isinstance(vision_feats, dict) else vision_feats

        # Prepare embedding dicts
        embedded_vision_feats = self._init_multilayer_vision_feats_dict(
            vision_feats
        )  # output shape: (batch_size, seq_len_padded, emdb_dims)
        embedded_text_tokens = self._embed_text_tokens(
            text_tokens
        )  # output shape: (batch_size, seq_len_padded, emdb_dims)

        # Combine both modalities
        final_embedding = self._combine_multilayer_embedding_dicts(
            embedded_vision_feats, text_tokens, embedded_text_tokens
        )

        final_embedding["type"] = "xattn"
        final_embedding["vision_xattn_mask"] = vision_xattn_mask
        final_embedding["buffer_xattn_mask"] = buffer_xattn_mask

        return final_embedding
