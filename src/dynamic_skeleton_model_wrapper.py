# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


import torch
import numpy as np
import os
import math
from typing import Union, List, Dict, Any
from tqdm import tqdm

from src.model_wrappers import StreamVLModelWrapper
from src.vision_skeleton_cross_attention import create_vision_skeleton_fusion_module
from src.morphometric_context import MorphometricContextModule
from src.qevd_biomechanic_processor import QEVDBiomechanicProcessor
from src.utils import load_video_timestamps
from src.biomech_slot import (
    compute_new_slot_kv,
    splice_past_kv,
    slice_past_kv_range,
    cache_to_legacy,
    legacy_to_cache,
    BIOMECH_SLOT_SIZE,
    SLOT_PLACEHOLDER_ID,
)
from src.constants import (
    FEEDBACK_BEGIN_TOKEN,
    FEEDBACK_END_TOKEN,
    VISION_TOKEN,
    INFERENCE_SPEED,
)

# Per-sample biomech-slot diagnostics are silenced by default. Set BIOCOACH_VERBOSE=1 to restore them.
_VERBOSE = os.environ.get("BIOCOACH_VERBOSE", "") not in ("", "0", "false", "False")


class DynamicSkeletonModelWrapper(StreamVLModelWrapper):
    """BioCoach Model Wrapper with biomechanics-grounded coaching pipeline.

    Also known as: BioCoachModelWrapper.

    Three-stage pipeline:
        1. DoF Selection.
        2. Biomechanical Context  – morphometric + motion quality context.
        3. Vision-Biomechanics Conditioned Generation  – cross-attention fusion
           + structured biomechanical instruction for the LLM.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skeleton_cache = {}


        self.inference_speed = kwargs.get('inference_speed') or INFERENCE_SPEED

        self.temporal_window = kwargs.get('temporal_window') or 3.0

        self.fb_trigger_bias = kwargs.get('fb_trigger_bias', None)


        self.morphometric_module = MorphometricContextModule(
            morphometric_dir=kwargs.get('morphometric_dir'),
        )


        self._fusion_module = None
        self.fusion_type = kwargs.get('fusion_type', 'cross_attention')
        self.num_heads = kwargs.get('num_heads', 8)
        self.fusion_dropout = kwargs.get('fusion_dropout', 0.1)
        self.cross_attention_weights_path = kwargs.get('cross_attention_weights_path', None)

        self.biomech_processor = QEVDBiomechanicProcessor(
            golden_standards_dir=kwargs.get('golden_standards_dir'),
        )
        self.enable_biomech_feedback = kwargs.get('enable_biomech_feedback', False)
        self.biomech_feedback_weight = kwargs.get('biomech_feedback_weight', 0.1)
        self.biomech_feedback_cache = {}


        self.biomech_inject_mode = kwargs.get('biomech_inject_mode') or 'slot'

        # Set tokenizer reference for adapter
        if hasattr(self.model, 'adapter'):
            self.model.adapter._tokenizer_ref = self.tokenizer
            if self.enable_biomech_feedback:
                self.model.adapter.biomech_weight = self.biomech_feedback_weight

        # Load trained LoRA weights if available
        lora_weights_path = kwargs.get('lora_weights_path', None)
        if lora_weights_path and os.path.exists(lora_weights_path):
            try:
                import torch as _torch
                lora_state = _torch.load(lora_weights_path, map_location=self.device)
                lang_params = dict(self.model.lang.named_parameters())
                loaded = 0
                for k, v in lora_state.items():
                    if k in lang_params:
                        lang_params[k].data.copy_(v)
                        loaded += 1
                print(f"Loaded {loaded} LoRA parameters from {lora_weights_path}")
            except Exception as e:
                print(f"Failed to load LoRA weights: {e}")


    def _get_morphometric_context(self, skeleton_file, cache_key=None):
        """Morphometric context C_morph (Sec 3.4.1).

        Uses the pre-computed SHAPY Virtual Measurements lookup
        (all_hsmr_shapy_measurements*.json, keyed by HSMR id). Falls back to a
        beta-based linear approximation only for ids missing from the lookup.
        """
        if not skeleton_file or not os.path.exists(skeleton_file):
            return ""
        try:
            return self.morphometric_module.get_context(skeleton_file, cache_key=cache_key)
        except Exception:
            return ""


    def _get_morphometric_features(self, skeleton_file: str) -> torch.Tensor:
        """Compute morphometric token embedding for cross-attention K/V.

        C_morph text → tokenize → LLM embed → mean pool → [hidden_size]
        Cached per skeleton_file (same person = same body measurements).

        Paper: m_t = Embed(C_morph)  (Sec 3.5, Eq 5)
        """
        cache_key = f"morph_feat_{skeleton_file}"
        if cache_key in self._skeleton_cache:
            return self._skeleton_cache[cache_key]

        # Get C_morph text from pre-computed SHAPY measurements
        morph_text = self._get_morphometric_context(skeleton_file, cache_key=skeleton_file)
        if not morph_text:
            return None

        try:
            tokens = self.tokenizer.encode(morph_text, add_special_tokens=False)
            token_ids = torch.tensor(tokens, device=self.device)
            with torch.no_grad():
                embeds = self.model.lang.get_input_embeddings()(token_ids)
            # Mean pool over token sequence → single feature vector
            feat = embeds.mean(dim=0)
            # Match model dtype
            if hasattr(self.model.lang.get_input_embeddings().weight, 'dtype'):
                feat = feat.to(dtype=self.model.lang.get_input_embeddings().weight.dtype)
            self._skeleton_cache[cache_key] = feat
            return feat
        except Exception:
            return None
        
    def _initialize_fusion_module(self, vision_dim: int, skeleton_dim: int):
        """Initialize the fusion module based on feature dimensions."""
        if self._fusion_module is None:
            try:
                fusion_module = create_vision_skeleton_fusion_module(
                    vision_dim=vision_dim,
                    skeleton_dim=skeleton_dim,
                    fusion_type=self.fusion_type,
                    num_heads=self.num_heads,
                    dropout=self.fusion_dropout
                )
                self._fusion_module = fusion_module.to(self.device)
            except Exception as e:
                self._fusion_module = None
            
            # Load trained weights if available
            if self.cross_attention_weights_path and os.path.exists(self.cross_attention_weights_path):
                try:
                    state_dict = torch.load(self.cross_attention_weights_path, map_location=self.device)

                    cleaned = {}
                    for k, v in state_dict.items():
                        if k.startswith('base_fusion.'):
                            cleaned[k[len('base_fusion.'):]] = v
                        elif not any(k.startswith(p) for p in
                                     ('temporal_head.', 'feedback_head.', 'bert_head.')):
                            cleaned[k] = v
                    self._fusion_module.load_state_dict(cleaned, strict=False)
                except Exception as e:
                    pass
            else:
                pass
            
        
    
    
    def _get_biomech_feedback_for_current_time(
        self,
        skeleton_file: str,
        current_time: float,
        video_start_time: float,
        exercise_name: str = "unknown"
    ) -> str:
        """Get biomechanic feedback for current time point."""
        if not self.enable_biomech_feedback or not skeleton_file or not os.path.exists(skeleton_file):
            return ""

        # Create cache key
        cache_key = f"{skeleton_file}_{current_time:.1f}_{exercise_name}"

        # Check cache first
        if cache_key in self.biomech_feedback_cache:
            return self.biomech_feedback_cache[cache_key]
        try:
            # Load skeleton data from file - try numpy first, then pickle
            try:
                # Most likely numpy format
                skeleton_data = np.load(skeleton_file, allow_pickle=True)
            except Exception:
                # Fallback to pickle
                import pickle
                with open(skeleton_file, 'rb') as f:
                    skeleton_data = pickle.load(f)


            video_ts_sec = None
            try:
                from pathlib import Path as _Path
                sp = _Path(skeleton_file)
                video_name = sp.stem.replace("HSMR-", "")
                ts_path = sp.parent.parent / "long_range_videos" / f"{video_name}_timestamps.npy"
                if ts_path.exists():
                    video_ts_sec = load_video_timestamps(str(ts_path))
            except Exception:
                video_ts_sec = None

            # Generate skeleton-relative timestamps (still passed for legacy fields)
            num_frames = len(skeleton_data) if isinstance(skeleton_data, (list, np.ndarray)) else 0
            fps = 30.0  # HSMR runs at 30 fps
            timestamps = np.linspace(0, num_frames / fps, num_frames) if num_frames > 0 else np.array([])

            if not isinstance(skeleton_data, np.ndarray):
                skeleton_data = np.array(skeleton_data)
            if not isinstance(timestamps, np.ndarray):
                timestamps = np.array(timestamps)


            if video_ts_sec is not None:
                segment_start = current_time - self.temporal_window
                segment_end = current_time
            else:
                relative_current_time = current_time - video_start_time if video_start_time else current_time
                segment_start = max(0.0, relative_current_time - self.temporal_window)
                segment_end = relative_current_time


            ex_start_time_param = (
                video_start_time if video_ts_sec is not None else 0.0
            )
            biomech_result = self.biomech_processor.process_video_segment(
                skeleton_data, timestamps, exercise_name,
                segment_start, segment_end,
                video_ts_sec=video_ts_sec,
                exercise_start_time=ex_start_time_param,
            )

            # Extract feedback text (append directly without prefix to match GT format)
            feedback_text = ""
            if biomech_result and biomech_result.get('has_feedback', False):
                feedback_text = biomech_result.get('feedback_text', '')
                # Keep only the clean feedback text
                if feedback_text.strip():
                    feedback_text = feedback_text.strip()
            
            # Cache the result
            self.biomech_feedback_cache[cache_key] = feedback_text
            return feedback_text
            
        except Exception as e:
            self.biomech_feedback_cache[cache_key] = ""
            return ""
    
    
    def generate_with_dynamic_skeleton(
        self,
        input_prompt: list[int],
        video: Union[str, torch.tensor],
        vision_xattn_mask: torch.tensor,
        skeleton_file: str = "",
        video_start_time: float = 0.0,
        skeleton_window_size: int = 3,
        exercise_name: str = "unknown",
        **kwargs
    ) -> torch.tensor:
        """Generate with dynamic skeleton injection/removal."""
        
        # Prepare inputs
        assert len(video.shape) == 4
        video, vision_xattn_mask, input_prompt = map(
            lambda t: self.to_torch_tensor_for_generation(t),
            [video, vision_xattn_mask, input_prompt],
        )
        
        # Encode video
        encoded_video = self.model.vision(video)
        
     
        feats_frequency = kwargs.pop('feats_frequency', 4)
        max_feedback_length = kwargs.pop('max_feedback_length', 128)
        do_sample = kwargs.pop('do_sample', False)
        temperature = kwargs.pop('temperature', 0.7)

        biomech_slot_start = kwargs.pop('biomech_slot_start', None)
        biomech_slot_end = kwargs.pop('biomech_slot_end', None)


        return self._generate_interactive_with_skeleton_merge(
            encoded_video, input_prompt, vision_xattn_mask,
            feats_frequency, max_feedback_length,
            do_sample, temperature,
            skeleton_file, video_start_time, skeleton_window_size,
            exercise_name,
            biomech_slot_start=biomech_slot_start,
            biomech_slot_end=biomech_slot_end,
            **kwargs
        )
    
    
    
    
    
    
    @torch.no_grad()
    def _generate_interactive_with_skeleton_merge(
        self,
        encoded_video: torch.tensor,
        input_ids: torch.tensor,
        vision_xattn_mask: torch.tensor,
        feats_frequency: int,
        max_feedback_length: int,
        do_sample: bool,
        temperature: float,
        skeleton_file: str,
        video_start_time: float,
        skeleton_window_size: int,
        exercise_name: str = "unknown",
        **kwargs
    ) -> torch.tensor:
        """Original StreamVLModelWrapper logic with skeleton-vision merging and bio feedback."""
        
        assert vision_xattn_mask is not None
        output_ids = input_ids.clone()

        # Clear biomech cache for new generation
        self.biomech_feedback_cache.clear()

        # Approach B: biomech slot positions in the prompt.
        biomech_slot_start = kwargs.pop("biomech_slot_start", None)
        biomech_slot_end = kwargs.pop("biomech_slot_end", None)
        use_biomech_slot = (
            biomech_slot_start is not None
            and biomech_slot_end is not None
            and biomech_slot_end > biomech_slot_start
        )
        # Disable old fusion path when slot is in use.
        if use_biomech_slot and hasattr(self.model, 'adapter'):
            self.model.adapter.biomech_context = None
            self.model.adapter.biomech_context_spans = None

        input_vision_idx = kwargs.get("input_vision_idx", 2)
        skip_blind_frames = [False] * (input_vision_idx - 1)
        past_key_values = None
        curr_response_len = 0
        feedback_mode = False

        # Continue generating until end of video (original logic)
        loop_count = 0
        
        while input_vision_idx < encoded_video["feats"].shape[1]:
            loop_count += 1
            
            # Prepare video input (original)
            encoded_video_feats = (
                encoded_video["feats"] if isinstance(encoded_video, dict) else encoded_video
            )
            encoded_video_spatial_res = (
                encoded_video.get("spatial_res", None) if isinstance(encoded_video, dict) else None
            )
            encoded_video_in_range = {
                "feats": encoded_video_feats[:, 1:input_vision_idx][
                    :, np.logical_not(skip_blind_frames)
                ],
                "spatial_res": encoded_video_spatial_res,
            }


            has_trained_xattn = (
                bool(self.cross_attention_weights_path)
                and os.path.exists(self.cross_attention_weights_path)
            )
            if skeleton_file and feedback_mode and has_trained_xattn:
                try:
                    original_mean = encoded_video_in_range["feats"].mean().item()

                    encoded_video_in_range = self._fuse_frame_wise_skeleton_to_vision(
                        encoded_video_in_range, skeleton_file, video_start_time, 
                        input_vision_idx, feats_frequency, skip_blind_frames, feedback_mode
                    )
                    
                    fused_mean = encoded_video_in_range["feats"].mean().item()
                    diff = abs(fused_mean - original_mean)
                    
                except Exception as e:
                    pass  # Silently continue without skeleton if error occurs

            multi_model_embedding = self.model.adapter(
                encoded_video_in_range, output_ids, vision_xattn_mask
            )

            # Generate next token logits (original)
            lang_out = self.model.lang(
                inputs_embeds=multi_model_embedding,
                attention_mask=torch.ones_like(output_ids).to(self.device),
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = lang_out["past_key_values"]

            # Sample next token (original)
            if not do_sample:
                lang_out_logits = torch.argmax(lang_out["logits"], dim=-1)
                output_ids = torch.cat([output_ids, lang_out_logits[:, -1][:, None]], dim=1)
            else:
                lang_out_logits = lang_out["logits"][:, -1]
                scaled_logits = lang_out_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                sampled_token = torch.multinomial(probs, num_samples=1)
                output_ids = torch.cat([output_ids, sampled_token], dim=1)

            # Sanity checks (original)
            if feedback_mode:
                curr_response_len += 1
                if (
                    output_ids[0, -1] == self.special_tokens_dict[VISION_TOKEN]
                    or output_ids[0, -1] == self.special_tokens_dict[FEEDBACK_BEGIN_TOKEN]
                    or curr_response_len > max_feedback_length
                ):
                    output_ids[0, -1] = self.special_tokens_dict[FEEDBACK_END_TOKEN]
            else:
                vision_id = self.special_tokens_dict[VISION_TOKEN]
                fb_begin_id = self.special_tokens_dict[FEEDBACK_BEGIN_TOKEN]
                if self.fb_trigger_bias is not None:

                    last_logits = lang_out["logits"][0, -1]
                    if (last_logits[fb_begin_id] + self.fb_trigger_bias
                            >= last_logits[vision_id]):
                        output_ids[0, -1] = fb_begin_id
                    else:
                        output_ids[0, -1] = vision_id
                else:
                    if (
                        output_ids[0, -1] != vision_id
                        and output_ids[0, -1] != fb_begin_id
                    ):
                        output_ids[0, -1] = vision_id

            # State changes (original)
            if output_ids[0, -1] == self.special_tokens_dict[VISION_TOKEN]:
                input_vision_idx += 1
                skip_blind_frames.append(False)
            elif output_ids[0, -1] == self.special_tokens_dict[FEEDBACK_BEGIN_TOKEN]:
                curr_response_len = 0
                feedback_mode = True

                # Set C_motion via biomech slot K/V splice (Approach B).
                if self.enable_biomech_feedback and skeleton_file:
                    relative_time = input_vision_idx / feats_frequency
                    current_time = video_start_time + relative_time

                    biomech_feedback = self._get_biomech_feedback_for_current_time(
                        skeleton_file, current_time, video_start_time, exercise_name
                    )

                    if self.biomech_inject_mode == 'append':

                        if hasattr(self.model, 'adapter'):
                            if biomech_feedback and biomech_feedback.strip():
                                fb_begin_pos = int(output_ids.shape[1]) - 1
                                self.model.adapter.biomech_context_spans = [
                                    (fb_begin_pos, fb_begin_pos + 1, biomech_feedback)
                                ]
                                if _VERBOSE:
                                    tqdm.write(
                                        f"🔧 biomech append → FB_BEGIN@{relative_time:.1f}s: "
                                        f"{biomech_feedback.strip()!r}"
                                    )
                            else:
                                self.model.adapter.biomech_context_spans = None
                    elif (use_biomech_slot
                            and biomech_feedback
                            and biomech_feedback.strip()
                            and past_key_values is not None):
                        # Splice new biomech K/V into the slot positions.
                        bio_tok_ids = self.tokenizer.encode(
                            biomech_feedback, add_special_tokens=False
                        )
                        try:
                            past_kv_prefix = slice_past_kv_range(
                                past_key_values, 0, biomech_slot_start
                            )
                            new_slot_kv = compute_new_slot_kv(
                                self.model.lang,
                                bio_tok_ids,
                                past_kv_prefix,
                                biomech_slot_start,
                                self.device,
                                # Real system-prompt prefix → slot K/V matches
                                # training (removes the train/inference mismatch).
                                prefix_token_ids=output_ids[0, :biomech_slot_start].tolist(),
                            )
                            past_key_values = legacy_to_cache(splice_past_kv(
                                past_key_values,
                                new_slot_kv,
                                biomech_slot_start,
                                biomech_slot_start + BIOMECH_SLOT_SIZE,
                            ))
  
                            n_bio = min(len(bio_tok_ids), BIOMECH_SLOT_SIZE)
                            fill_vals = (
                                list(bio_tok_ids[:n_bio])
                                + [SLOT_PLACEHOLDER_ID] * (BIOMECH_SLOT_SIZE - n_bio)
                            )
                            output_ids[
                                ..., biomech_slot_start:biomech_slot_start + BIOMECH_SLOT_SIZE
                            ] = torch.tensor(
                                fill_vals, dtype=output_ids.dtype, device=output_ids.device
                            )
                            if _VERBOSE:
                                _trunc = " ⚠️TRUNCATED@cap" if n_bio >= BIOMECH_SLOT_SIZE else ""
                                tqdm.write(
                                    f"🔧 biomech slot ← {n_bio}/{BIOMECH_SLOT_SIZE} tok{_trunc} "
                                    f"@ {relative_time:.1f}s: {biomech_feedback.strip()!r}"
                                )
                        except Exception as exc:
                            import traceback as _tb
                            print(f"⚠️  biomech slot splice failed: {type(exc).__name__}: {exc}")
                            _tb.print_exc()

            elif output_ids[0, -1] == self.special_tokens_dict[FEEDBACK_END_TOKEN]:
                feedback_mode = False
                if hasattr(self.model, 'adapter'):
                    self.model.adapter.biomech_context_spans = None
                    self.model.adapter.biomech_context = None
                skip_forward = math.floor((curr_response_len / self.inference_speed) * feats_frequency)
                input_vision_idx += skip_forward
                skip_blind_frames += [True] * skip_forward
                curr_response_len = 0

            # Update masks (original)
            if output_ids[0, -1] == self.special_tokens_dict[VISION_TOKEN]:
                vision_xattn_mask_pad = torch.ones(input_ids.shape[0], 1) * 2
                vision_xattn_mask_pad = vision_xattn_mask_pad.to(vision_xattn_mask)
                vision_xattn_mask = torch.cat([vision_xattn_mask, vision_xattn_mask_pad], dim=1)
            else:
                vision_xattn_mask_pad = torch.zeros(input_ids.shape[0], 1)
                vision_xattn_mask_pad = vision_xattn_mask_pad.to(vision_xattn_mask)
                vision_xattn_mask = torch.cat([vision_xattn_mask, vision_xattn_mask_pad], dim=1)

        if output_ids.numel() == 1:
            raise ValueError(f"output_ids became scalar: {output_ids.item()}")
        
        output_ids = output_ids.cpu().numpy()
        return output_ids
    
    
    
    def _fuse_frame_wise_skeleton_to_vision(
        self,
        encoded_video_in_range: dict,
        skeleton_file: str,
        video_start_time: float,
        input_vision_idx: int,
        feats_frequency: int,
        skip_blind_frames: list = None,
        feedback_mode: bool = False
    ) -> dict:

        try:
            vision_feats = encoded_video_in_range.get("feats")
            if vision_feats is None:
                return encoded_video_in_range

            if vision_feats.ndim == 5:
                batch_size, num_frames, channels, height, width = vision_feats.shape
                vision_dim = channels * height * width
                vision_feats = vision_feats.reshape(batch_size, num_frames, -1)
            elif vision_feats.ndim == 4:
                batch_size, num_frames, patches, features = vision_feats.shape
                vision_dim = features
            elif vision_feats.ndim == 3:
                batch_size, num_frames, vision_dim = vision_feats.shape
            else:
                return encoded_video_in_range

            enhanced_feats = vision_feats.clone()

            # Get morphometric features (per-video constant, cached)
            # Paper Sec 3.5: m_t = Embed(C_morph)
            morph_feat = self._get_morphometric_features(skeleton_file)

            if morph_feat is not None:
                morph_dim = morph_feat.shape[0]
                morph_feat = morph_feat.to(dtype=vision_feats.dtype, device=vision_feats.device)

                # Initialize fusion module with morphometric dim as K/V dim
                self._initialize_fusion_module(
                    vision_dim if vision_feats.ndim != 4 else features,
                    morph_dim
                )

                # Apply cross-attention: same m_t for ALL frames
                # z_t = F_vis + CrossAttn(F_vis, m_t, m_t)
                morph_kv = morph_feat.unsqueeze(0)  # [1, morph_dim]

                for frame_idx in range(num_frames):
                    try:
                        with torch.no_grad():
                            if self._fusion_module is not None:
                                self._fusion_module = self._fusion_module.to(dtype=vision_feats.dtype)
                                if vision_feats.ndim == 4:
                                    # [1, patches, features] as Q
                                    vis_frame = vision_feats[0, frame_idx].unsqueeze(0)
                                    fused = self._fusion_module(vis_frame, morph_kv)
                                    enhanced_feats[0, frame_idx] = fused[0]
                                else:
                                    # [1, 1, vision_dim] as Q
                                    vis_frame = vision_feats[0, frame_idx:frame_idx+1].unsqueeze(0)
                                    fused = self._fusion_module(vis_frame, morph_kv)
                                    enhanced_feats[0, frame_idx] = fused[0, 0]
                    except Exception:
                        pass  # Keep original features for this frame

            return {
                "feats": enhanced_feats,
                "spatial_res": encoded_video_in_range.get("spatial_res")
            }

        except Exception as e:
            return encoded_video_in_range



BioCoachModelWrapper = DynamicSkeletonModelWrapper

