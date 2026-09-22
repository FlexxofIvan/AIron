# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""Simplified Dynamic Skeleton Injection Evaluator."""

import json
import os
import random
import math
from typing import Any, List, Optional, Tuple, Union

import evaluate
import numpy as np
import torch
from datasets import Dataset
from torch import nn
from tqdm import tqdm

from src.constants import (
    FEEDBACK_BEGIN_TOKEN,
    FEEDBACK_END_TOKEN,
    INFERENCE_SPEED,
    VISION_TOKEN,
)
from src.model_wrappers import BaseVLModelWrapper
from src.evaluators import VisionLanguageEvaluator
from src.biomech_slot import insert_slot_before_token

# Per-sample debug output is silenced by default. Set BIOCOACH_VERBOSE=1 to restore it.
_VERBOSE = os.environ.get("BIOCOACH_VERBOSE", "") not in ("", "0", "false", "False")


def _dbg(msg):
    """Print a per-sample debug line only when BIOCOACH_VERBOSE is enabled."""
    if _VERBOSE:
        tqdm.write(msg)


class DynamicSkeletonEvaluator(VisionLanguageEvaluator):
    """Dynamic skeleton-injection evaluator."""

    def __init__(
        self,
        model: Union[nn.Module, BaseVLModelWrapper],
        dataset: Dataset,
        feedbacks_save_folder: str,
        feedbacks_save_file_name: str = "dynamic_skeleton_responses",
        skeleton_window_size: int = 3,
        skeleton_fps: float = 30.0,
        tolerance: float = 3.0,
    ):
        """Initialize the dynamic skeleton evaluator."""
        super().__init__(model, dataset)
        
        try:
            self.bert_score = evaluate.load("bertscore")
        except Exception as e:
            try:
                # Alternative loading method
                import bert_score
                self.bert_score = bert_score
                self._use_bert_score_direct = True
            except ImportError:
                self.bert_score = None
                self._use_bert_score_direct = False
        
        try:
            self.meteor_metric = evaluate.load("meteor")
        except Exception:
            self.meteor_metric = None
        
        self.mean = lambda x: sum(x) / (len(x) + 1e-12)

        self.feedbacks_save_path = os.path.join(
            feedbacks_save_folder, f"{feedbacks_save_file_name}.json"
        )
        
        self.skeleton_window_size = skeleton_window_size
        self.skeleton_fps = skeleton_fps

        self.tolerance = tolerance


    @staticmethod
    def _get_video_for_episode(
        video: np.array,
        video_timestamps: np.array,
        episode_start_timestamp: Optional[float] = None,
        episode_end_timestamp: Optional[float] = None,
    ) -> np.array:
        """Returns a slice of the full video corresponding to the given time interval."""
        if episode_start_timestamp is not None:
            episode_flag = np.logical_and(
                video_timestamps > episode_start_timestamp,
                video_timestamps <= episode_end_timestamp,
            )
            video = video[episode_flag]
        return video


    def _generate_with_dynamic_skeleton(
        self,
        data: dict,
        **sampling_kwargs
    ) -> List[int]:
        """Run dynamic skeleton-injection generation for one sample."""
        
        feat_path = data["efficientnet_features_path"]
        system_prompt = data["system"] 
        skeleton_file = data.get("skeleton_path", "")
        video_start_time = data.get("exercise_start_timestamp", 0)
        video_end_time = data.get("exercise_end_timestamp", float('inf'))
        
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Video features file not found: {feat_path}")
        if not os.path.exists(data["efficientnet_timestamps_path"]):
            raise FileNotFoundError(f"Timestamps file not found: {data['efficientnet_timestamps_path']}")
            
        video = np.load(feat_path)
        video_timestamps = np.load(data["efficientnet_timestamps_path"])
        
        video = self._get_video_for_episode(
            video, video_timestamps, video_start_time, video_end_time
        )
        
        # Check if model supports dynamic skeleton
        from src.dynamic_skeleton_model_wrapper import DynamicSkeletonModelWrapper
        if not isinstance(self.model, DynamicSkeletonModelWrapper):
            raise ValueError("Model must be DynamicSkeletonModelWrapper for dynamic skeleton evaluation")
        
        input_prompt = system_prompt + VISION_TOKEN
        input_prompt_tokens = self.model.tokenizer.encode(input_prompt)

        inject_mode = getattr(self.model, 'biomech_inject_mode', 'slot')
        if inject_mode == 'append':

            self._biomech_slot_range = (None, None)
        else:
            vision_token_id = self.model.special_tokens_dict[VISION_TOKEN]
            input_prompt_tokens, _slot_start, _slot_end = insert_slot_before_token(
                input_prompt_tokens, vision_token_id
            )
            # Remember slot position so we can pass it to the wrapper below.
            self._biomech_slot_range = (_slot_start, _slot_end)


        self._last_prompt_token_length = len(input_prompt_tokens)

        vision_xattn_mask = self._get_vision_xattn_mask(input_prompt_tokens)
        vision_xattn_mask = [2 if tok == 1 else 0 for tok in vision_xattn_mask]
        
        vision_xattn_mask = np.array(vision_xattn_mask)
        
        # Log key parameters (minimal debug info)
        _dbg(f"📝 Generating: {video.shape[0]} frames, skeleton: {'yes' if skeleton_file else 'no'}")
        
        
        try:
            out = self.model.generate(
                input_prompt=input_prompt_tokens,
                video=video,
                vision_xattn_mask=vision_xattn_mask,
                **sampling_kwargs,
            )
            if hasattr(out, 'shape'):
                pass  # Handle tensor output
            elif hasattr(out, '__len__'):
                pass  # Handle list/sequence output
            
        except Exception as e:
            raise e
        
        if hasattr(self.model, 'generate_with_dynamic_skeleton'):
            try:
                # Get exercise name from data for bio feedback
                exercise_name = data.get('exercise_name', data.get('exercise_type', data.get('class', 'unknown')))
                slot_start, slot_end = getattr(self, '_biomech_slot_range', (None, None))
                out = self.model.generate_with_dynamic_skeleton(
                    input_prompt=input_prompt_tokens,
                    video=video,
                    vision_xattn_mask=vision_xattn_mask,
                    skeleton_file=skeleton_file,
                    video_start_time=video_start_time,
                    skeleton_window_size=self.skeleton_window_size,
                    exercise_name=exercise_name,
                    biomech_slot_start=slot_start,
                    biomech_slot_end=slot_end,
                    **sampling_kwargs,
                )
                if hasattr(out, 'shape'):
                    pass  # Handle tensor output from dynamic skeleton generation
            except Exception as e:
                raise e
        else:
            # Fallback to regular generation if dynamic skeleton not supported
            pass
        
        
        if hasattr(out, 'cpu'):
            result = out[0].cpu().tolist()
        elif isinstance(out, np.ndarray):
            if len(out.shape) > 1:
                result = out[0].tolist()
            else:
                result = out.tolist()
        else:
            if hasattr(out, '__getitem__'):
                result = out[0]
            else:
                result = out
        
        
        if not isinstance(result, list):
            if hasattr(result, 'tolist'):
                result = result.tolist()
            elif isinstance(result, (int, float)):
                raise ValueError(f"Model returned single value instead of token sequence: {result}")
            else:
                raise ValueError(f"Cannot convert result to list: {result} (type: {type(result)})")
        
        if len(result) == 0:
            raise ValueError("Model generated empty token sequence")
        
        return result
    

    def _extract_pred_feedbacks(
        self, output: list[int], feats_frequency: int, system_prompt: str = ""
    ) -> tuple[list[str], Any]:
        """Extract predicted feedbacks and their timestamps from model output."""
        responses = []
        timestamps = []

        # Convert output to numpy array for processing first
        output = np.array(output)
        original_length = len(output)
        
        if len(output) == 0:
            return [], []
        
        # Find the input prompt length to skip it if system_prompt is provided
        if system_prompt:
            vision_token_id = self.model.special_tokens_dict[VISION_TOKEN]

            input_prompt_length = getattr(self, "_last_prompt_token_length", None)
            if input_prompt_length is None:
                input_prompt_length = len(
                    self.model.tokenizer.encode(system_prompt + VISION_TOKEN)
                )

            # Skip the input prompt part
            if len(output) > input_prompt_length:
                output_without_input = output[input_prompt_length:]
                _dbg(f"📝 Skipped input prompt ({input_prompt_length} tokens), processing {len(output_without_input)} tokens")
                output = output_without_input
            else:
                _dbg(f"⚠️ Model output too short ({len(output)} tokens), using full output")
        else:
            # Try to find vision tokens to skip input
            vision_token_positions = np.where(output == self.model.special_tokens_dict[VISION_TOKEN])[0]
            if len(vision_token_positions) > 0:
                # Use from first vision token
                first_vision_token = vision_token_positions[0]
                output = output[first_vision_token:]
                _dbg(f"📍 Found vision token at position {first_vision_token}")
            else:
                _dbg(f"📍 No vision token found, using full output")
        
        # Count special tokens in the processed output (after skipping input)
        feedback_begin_count = np.sum(output == self.model.special_tokens_dict[FEEDBACK_BEGIN_TOKEN])
        feedback_end_count = np.sum(output == self.model.special_tokens_dict[FEEDBACK_END_TOKEN])
        vision_count = np.sum(output == self.model.special_tokens_dict[VISION_TOKEN])
        
        _dbg(f"📊 Output: {original_length} total tokens ({len(output)} processed), FB:{feedback_begin_count}, FE:{feedback_end_count}, V:{vision_count}")

        # Get feedback indices
        feedback_begin_idxs = np.where(
            output == self.model.special_tokens_dict[FEEDBACK_BEGIN_TOKEN]
        )[0]
        feedback_end_idxs = np.where(output == self.model.special_tokens_dict[FEEDBACK_END_TOKEN])[
            0
        ]
        
        # Debug feedback token extraction
        if len(feedback_begin_idxs) > 0 or len(feedback_end_idxs) > 0:
            _dbg(f"📍 Found feedback tokens: begin={len(feedback_begin_idxs)}, end={len(feedback_end_idxs)}")
        elif _VERBOSE:
            # Show what tokens we actually have
            decoded_output = self.model.tokenizer.decode(output[:50], skip_special_tokens=False)
            _dbg(f"📝 First 50 tokens: {decoded_output}...")

        # Extract the feedback strings and timestamps.
        previous_answer_generation_time = 0
        for idx in range(min(len(feedback_begin_idxs), len(feedback_end_idxs))):
            answer_begin_idx = feedback_begin_idxs[idx]
            answer_end_idx = feedback_end_idxs[idx]

            if idx > 0:
                previous_answer_end_idx = feedback_end_idxs[idx - 1]
            else:
                previous_answer_end_idx = 1

            response = self.model.tokenizer.decode(output[answer_begin_idx + 1 : answer_end_idx])

            # Ignore empty responses
            if len(response) > 0:
                responses.append(response)

                timestep = answer_begin_idx - previous_answer_end_idx - 1
                timestep /= feats_frequency
                timestep += previous_answer_generation_time
                timestamps.append(timestep)

                previous_answer_generation_time = answer_end_idx - answer_begin_idx
                previous_answer_generation_time /= getattr(self.model, 'inference_speed', INFERENCE_SPEED)
            else:
                previous_answer_generation_time = 0


        cumulative_timestamps = np.clip(np.cumsum(timestamps), 0.0, None)
        return responses, cumulative_timestamps.tolist()

    @staticmethod
    def _get_alignment_matrix(
        gt_feedback_timestamps: np.array,
        pred_feedback_timestamps: np.array,
        pred_feedbacks: list[str],
        tolerance: float = 1.0,
    ) -> tuple[list[int], list[int]]:
        """Perform temporal matching between GT and predicted feedbacks."""
        matching_row_idxs, matching_col_idxs = [], []
        last_match_idx = -1
        for idx_x, x in enumerate(gt_feedback_timestamps):
            min_idx = np.argmin((pred_feedback_timestamps - x) ** 2)
            if (
                np.abs(x - pred_feedback_timestamps[min_idx]) < (tolerance / 2.0)
                and min_idx > last_match_idx
                and (min_idx not in matching_col_idxs)
                and pred_feedbacks[min_idx] != ""
            ):
                matching_row_idxs.append(idx_x)
                matching_col_idxs.append(min_idx)
                last_match_idx = min_idx
        return matching_row_idxs, matching_col_idxs


    def _get_temporally_aligned_feedbacks(
        self,
        gt_feedback_timestamps: list[float],
        pred_feedback_timestamps: list[float],
        gt_feedbacks: list[str],
        pred_feedbacks: list[str],
        tolerance: float = 1.0,
    ) -> tuple[int, int, list[tuple[str, str]]]:
        """Returns temporally aligned feedbacks between GT and predictions."""
        gt_feedback_timestamps = np.array(gt_feedback_timestamps)
        pred_feedback_timestamps = np.array(pred_feedback_timestamps)

        matched_feedbacks = []
        matched_idxs_gt = []
        matched_idxs_pred = []
        matching_row_idxs, matching_col_idxs = [], []
        if len(pred_feedback_timestamps) > 0:
            matching_row_idxs, matching_col_idxs = self._get_alignment_matrix(
                gt_feedback_timestamps,
                pred_feedback_timestamps,
                pred_feedbacks,
                tolerance,
            )

        for match_idx, match_jdx in zip(matching_row_idxs, matching_col_idxs):
            matched_feedbacks.append((gt_feedbacks[match_idx], pred_feedbacks[match_jdx]))
            matched_idxs_gt.append(match_idx)
            matched_idxs_pred.append(match_jdx)

        return len(matched_idxs_gt), len(matched_idxs_pred), matched_feedbacks

    def _get_temporally_aligned_feedbacks_with_timestamps(
        self,
        gt_feedback_timestamps: list[float],
        pred_feedback_timestamps: list[float],
        gt_feedbacks: list[str],
        pred_feedbacks: list[str],
        tolerance: float = 1.0,
    ) -> list[tuple[str, str, float]]:
        """Returns temporally aligned feedbacks with their corresponding predicted timestamps."""
        gt_feedback_timestamps = np.array(gt_feedback_timestamps)
        pred_feedback_timestamps = np.array(pred_feedback_timestamps)
        matched_feedbacks_with_timestamps = []
        matching_row_idxs, matching_col_idxs = [], []
        if len(pred_feedback_timestamps) > 0:
            matching_row_idxs, matching_col_idxs = self._get_alignment_matrix(
                gt_feedback_timestamps,
                pred_feedback_timestamps,
                pred_feedbacks,
                tolerance,
            )
        for match_idx, match_jdx in zip(matching_row_idxs, matching_col_idxs):
            matched_feedbacks_with_timestamps.append((
                gt_feedbacks[match_idx], 
                pred_feedbacks[match_jdx], 
                pred_feedback_timestamps[match_jdx]
            ))
        return matched_feedbacks_with_timestamps

    def _compute_temporal_fscore(
        self,
        gt_feedbacks: list[str],
        pred_feedbacks: list[str],
        gt_feedback_timestamps: list[float],
        pred_feedback_timestamps: list[float],
        t_f_score_running_stats: dict[str, int],
        tolerance: float = 1.0,
        eps: float = 1e-12,
    ) -> tuple[float, list[tuple[str, str]], dict[str, int]]:
        """Compute temporal F-score using original logic."""
        # Match ground truth feedbacks to predicted feedbacks (recall)
        num_matched_gt, _, matched_feedbacks = self._get_temporally_aligned_feedbacks(
            gt_feedback_timestamps, pred_feedback_timestamps, gt_feedbacks, pred_feedbacks, tolerance
        )

        # Match predicted feedbacks to ground truth feedbacks (precision)
        _, num_matched_preds, _ = self._get_temporally_aligned_feedbacks(
            pred_feedback_timestamps, gt_feedback_timestamps, pred_feedbacks, gt_feedbacks, tolerance
        )

        # Accumulate running stats
        t_f_score_running_stats["total_matched_gt_feedbacks"] += num_matched_gt
        t_f_score_running_stats["total_matched_pred_feedbacks"] += num_matched_preds
        t_f_score_running_stats["total_num_gt_feedbacks"] += len(gt_feedbacks)
        t_f_score_running_stats["total_num_pred_feedbacks"] += len(pred_feedbacks)

        # Compute temporal precision and recall
        precision = t_f_score_running_stats["total_matched_pred_feedbacks"] / (
            t_f_score_running_stats["total_num_pred_feedbacks"] + eps
        )
        recall = t_f_score_running_stats["total_matched_gt_feedbacks"] / (
            t_f_score_running_stats["total_num_gt_feedbacks"] + eps
        )
        f_score = 2 * ((precision * recall) / (precision + recall + eps))
        return f_score, matched_feedbacks, t_f_score_running_stats

    def _compute_rouge_scores(self, matched_feedbacks: list[tuple[str, str]]) -> list[float]:
        """Compute ROUGE scores for matched feedbacks."""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            filtered_feedbacks = [
                (gt, pred) for gt, pred in matched_feedbacks if gt.strip() and pred.strip()
            ]
            if len(filtered_feedbacks) != len(matched_feedbacks):
                skipped = len(matched_feedbacks) - len(filtered_feedbacks)
                if skipped > 0:
                    _dbg(f"⏭️ Skipped {skipped} empty GT/pred feedback pairs for ROUGE scoring.")
            scores = []
            for gt, pred in filtered_feedbacks:
                try:
                    score = scorer.score(gt, pred)['rougeL'].fmeasure
                    scores.append(score)
                except Exception:
                    scores.append(0.0)
            return scores
        except ImportError:
            # Fallback if rouge_score is not available
            return [0.0] * len(matched_feedbacks)

    def _compute_meteor_scores(self, matched_feedbacks: list[tuple[str, str]]) -> list[float]:
        """Compute METEOR scores for matched feedbacks using HuggingFace evaluate."""
        if self.meteor_metric is None:
            return []
        filtered_feedbacks = [
            (gt, pred) for gt, pred in matched_feedbacks if gt.strip() and pred.strip()
        ]
        if len(filtered_feedbacks) != len(matched_feedbacks):
            skipped = len(matched_feedbacks) - len(filtered_feedbacks)
            if skipped > 0:
                _dbg(f"⏭️ Skipped {skipped} empty GT/pred feedback pairs for METEOR scoring.")
        scores = []
        for gt, pred in filtered_feedbacks:
            try:
                result = self.meteor_metric.compute(references=[gt], predictions=[pred])
                scores.append(result.get("meteor", 0.0))
            except Exception:
                scores.append(0.0)
        return scores

    def _compute_bert_scores(self, matched_feedbacks: list[tuple[str, str]]) -> list[float]:
        """Compute BERT scores for matched feedbacks."""
        if self.bert_score is None:
            return []
        
        filtered_feedbacks = [
            (gt, pred) for gt, pred in matched_feedbacks if gt.strip() and pred.strip()
        ]
        if len(filtered_feedbacks) != len(matched_feedbacks):
            skipped = len(matched_feedbacks) - len(filtered_feedbacks)
            if skipped > 0:
                _dbg(f"⏭️ Skipped {skipped} empty GT/pred feedback pairs for BERTScore.")
        
        predictions = [pred for gt, pred in filtered_feedbacks]
        references = [gt for gt, pred in filtered_feedbacks]
        
        if not predictions or not references:
            return []
            
        try:
            # Use evaluate.load approach
            if hasattr(self, '_use_bert_score_direct') and self._use_bert_score_direct:
                # Direct bert_score package
                P, R, F1 = self.bert_score.score(predictions, references, lang="en", verbose=False)
                return F1.tolist()
            else:
                # evaluate.load approach
                results = self.bert_score.compute(predictions=predictions, references=references, lang="en")
                return results["f1"]
        except Exception as e:
            return [0.0] * len(matched_feedbacks)

    def _update_save_feedbacks_dict(self, save_dict_list: list, matched_feedbacks_with_timestamps: list[tuple[str, str, float]],
                                   sample_info: dict = None) -> None:
        """Update the save dictionary with matched feedbacks (simplified format matching streamvlm_responses.json)."""
        for i, (gt, pred, timestamp) in enumerate(matched_feedbacks_with_timestamps):
            # Simple format: only GT and Pred fields (matching streamvlm_responses.json)
            feedback_entry = {"GT": gt, "Pred": pred}
            save_dict_list.append(feedback_entry)

    @staticmethod
    def _dump_pred_feedbacks(save_dict_list: list[dict], save_path: str) -> None:
        """Save predicted feedbacks to JSON file."""
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_dict_list, f, indent=2, ensure_ascii=False)

    def _print_eval_summary(
        self,
        gt_feedbacks: list[str],
        gt_feedback_timestamps: list[float],
        pred_feedbacks: list[str],
        pred_feedback_timestamps: list[float],
        t_f_score: float,
        meteor_scores: list[float],
        rouge_scores: list[float],
        bert_scores: list[float],
    ) -> None:
        """Print concise evaluation summary."""
        bert_info = f", BERT: {self.mean(bert_scores):.3f}" if bert_scores and self.bert_score is not None else ""
        
        tqdm.write(f"📊 GT: {len(gt_feedbacks)}, Pred: {len(pred_feedbacks)} | "
                  f"METEOR: {self.mean(meteor_scores):.3f}, ROUGE-L: {self.mean(rouge_scores):.3f}"
                  f"{bert_info}, Temporal F: {t_f_score:.3f}")
        
        # Show first few feedbacks as example
        if gt_feedbacks:
            tqdm.write(f"GT example: {gt_feedback_timestamps[0]:.1f}s => {gt_feedbacks[0][:50]}...")
        if pred_feedbacks:
            tqdm.write(f"Pred example: {pred_feedback_timestamps[0]:.1f}s => {pred_feedbacks[0][:50]}...")

    def __call__(self, **sampling_kwargs):
        """Main evaluation loop."""
        tqdm.write("Starting dynamic skeleton evaluation...")
        
        rouge_scores, meteor_scores, bert_scores = [], [], []
        matched_feedbacks_to_save = []
        
        t_f_score_running_stats = {
            "total_matched_gt_feedbacks": 0,
            "total_matched_pred_feedbacks": 0,
            "total_num_gt_feedbacks": 0,
            "total_num_pred_feedbacks": 0,
        }
        
        feats_frequency = sampling_kwargs.get("feats_frequency", 4)

        for j, data in enumerate(tqdm(self.dataset, desc="Evaluating")):
            # Get exercise name and convert to underscore format for bio feedback matching
            raw_exercise_name = data.get('exercise_name', data.get('exercise_type', 'unknown'))
            exercise_type = raw_exercise_name.lower().replace(' ', '_')
            
            try:
                from src.dynamic_skeleton_model_wrapper import DynamicSkeletonModelWrapper
                if isinstance(self.model, DynamicSkeletonModelWrapper):
                    output_tokens = self._generate_with_dynamic_skeleton(data, **sampling_kwargs)
                else:
                    feat_path = data["efficientnet_features_path"]
                    system_prompt = data["system"] 
                    video_start_time = data.get("exercise_start_timestamp", 0)
                    video_end_time = data.get("exercise_end_timestamp", float('inf'))
                    
                    video = np.load(feat_path)
                    video_timestamps = np.load(data["efficientnet_timestamps_path"])
                    video = self._get_video_for_episode(video, video_timestamps, video_start_time, video_end_time)
                    
                    input_prompt = system_prompt + VISION_TOKEN
                    input_prompt_tokens = self.model.tokenizer.encode(input_prompt)
                    # No biomech slot in this fallback path — record the plain
                    # prompt length so _extract_pred_feedbacks skips exactly it.
                    self._last_prompt_token_length = len(input_prompt_tokens)
                    vision_xattn_mask = self._get_vision_xattn_mask(input_prompt_tokens)
                    vision_xattn_mask = [2 if tok == 1 else 0 for tok in vision_xattn_mask]
                    
                    output_tokens = self.model.generate(
                        input_prompt=input_prompt_tokens,
                        video=video,
                        vision_xattn_mask=np.array(vision_xattn_mask),
                        **sampling_kwargs
                    )
                    
                    _dbg(f"🔍 output_tokens type: {type(output_tokens)}")
                    _dbg(f"🔍 output_tokens shape: {output_tokens.shape if hasattr(output_tokens, 'shape') else 'no shape'}")
                    _dbg(f"🔍 output_tokens content: {output_tokens}")
                    
                    if hasattr(output_tokens, 'cpu'):
                        output_tokens = output_tokens.cpu().numpy()
                    if len(output_tokens.shape) > 1:
                        output_tokens = output_tokens[0]  # Take first batch
                    output_tokens = output_tokens.tolist()  # Convert to list for compatibility
                
                skeleton_file = data.get("skeleton_path", "")
                video_start_time = data.get("exercise_start_timestamp", 0)

                pred_feedbacks, pred_feedback_timestamps = self._extract_pred_feedbacks(
                    output_tokens, feats_frequency, data.get("system", "")
                )
                
                gt_feedbacks = data.get("responses", [])
                gt_feedback_timestamps = data.get("response_timestamps", [])
                
                exercise_start_timestamp = data.get("exercise_start_timestamp", 0)
                if exercise_start_timestamp and gt_feedback_timestamps:
                    gt_feedback_timestamps = [(ts - exercise_start_timestamp) for ts in gt_feedback_timestamps if ts >= exercise_start_timestamp]
                
                if len(gt_feedbacks) == 0 or len(pred_feedbacks) == 0:
                    _dbg(f"Sample {j} ({exercise_type}): No feedbacks (GT:{len(gt_feedbacks)}, Pred:{len(pred_feedbacks)})")
                    continue
                
                
                t_f_score, matched_feedbacks, t_f_score_running_stats = self._compute_temporal_fscore(
                    gt_feedbacks,
                    pred_feedbacks,
                    gt_feedback_timestamps,
                    pred_feedback_timestamps,
                    t_f_score_running_stats,
                    tolerance=self.tolerance,
                )

                matched_feedbacks_with_timestamps = self._get_temporally_aligned_feedbacks_with_timestamps(
                    gt_feedback_timestamps, pred_feedback_timestamps, gt_feedbacks, pred_feedbacks, tolerance=self.tolerance
                )
                
                
                rouge_scores += self._compute_rouge_scores(matched_feedbacks)
                meteor_scores += self._compute_meteor_scores(matched_feedbacks)
                if self.bert_score is not None:
                    bert_scores += self._compute_bert_scores(matched_feedbacks)

                self._print_eval_summary(
                    gt_feedbacks,
                    gt_feedback_timestamps,
                    pred_feedbacks,
                    pred_feedback_timestamps,
                    t_f_score,
                    meteor_scores,
                    rouge_scores,
                    bert_scores,
                )

                sample_info = {
                    "sample_id": f"sample_{j}",
                    "exercise_type": exercise_type,
                    "skeleton_file": skeleton_file,
                    "video_start_time": video_start_time,
                    "model": self.model if isinstance(self.model, DynamicSkeletonModelWrapper) else None
                }
                
                self._update_save_feedbacks_dict(matched_feedbacks_to_save, matched_feedbacks_with_timestamps, sample_info)
                
            except Exception as e:
                tqdm.write(f"Error in sample {j} ({exercise_type}): {str(e)[:100]}...")
                import gc
                gc.collect()
                continue

        self._dump_pred_feedbacks(matched_feedbacks_to_save, self.feedbacks_save_path)


def create_dynamic_skeleton_evaluator(
    model, dataset, feedbacks_save_folder, feedbacks_save_file_name="dynamic_skeleton_responses", skeleton_window_size=3,
    tolerance=3.0,
    **model_config_kwargs
):
    """Factory for the dynamic skeleton evaluator (wraps the model if needed)."""
    from src.dynamic_skeleton_model_wrapper import DynamicSkeletonModelWrapper
    from src.model_wrappers import StreamVLModelWrapper
    
    if isinstance(model, StreamVLModelWrapper) and not isinstance(model, DynamicSkeletonModelWrapper):
        try:
            # Convert to dynamic skeleton wrapper
            wrapped_model = DynamicSkeletonModelWrapper(
                model.model,      # Use the underlying model
                model.tokenizer,  # Use tokenizer
                device=model.device,  # Keep device
                train_llm=getattr(model, 'train_llm', False),
                train_xattn=getattr(model, 'train_xattn', False), 
                train_vision=getattr(model, 'train_vision', False),
                **model_config_kwargs  # Pass cross attention config
            )
            # Copy special tokens dict
            wrapped_model.special_tokens_dict = model.special_tokens_dict
            model = wrapped_model
            tqdm.write(f"✅ Successfully wrapped model with DynamicSkeletonModelWrapper")
        except Exception as e:
            tqdm.write(f"⚠️ Failed to wrap model: {str(e)}")
    elif not isinstance(model, DynamicSkeletonModelWrapper):
        tqdm.write(f"❌ Expected StreamVLModelWrapper, got {type(model)}")
    
    return DynamicSkeletonEvaluator(
        model=model,
        dataset=dataset,
        feedbacks_save_folder=feedbacks_save_folder,
        feedbacks_save_file_name=feedbacks_save_file_name,
        skeleton_window_size=skeleton_window_size,
        tolerance=tolerance,
    )
