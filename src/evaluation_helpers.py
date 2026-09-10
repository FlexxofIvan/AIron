# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""Evaluator Loading Helper Functions."""

from typing import Callable, Union

from datasets import Dataset
from torch import nn

from src.evaluators import InteractiveFeedbackEvaluator
from src.dynamic_skeleton_evaluator import DynamicSkeletonEvaluator
from src.model_wrappers import BaseVLModelWrapper


def get_evaluator(evaluator_name: str) -> Callable:
    """
    :return:
        An evaluator of type VisionLanguageEvaluator.
    """
    if evaluator_name == "interactive_feedback_evaluator":
        return InteractiveFeedbackEvaluator
    elif evaluator_name == "dynamic_skeleton_evaluator":
        return DynamicSkeletonEvaluator

    raise NotImplementedError(f"Evaluator: {evaluator_name}, not found.")


def evaluate_model(
    model: Union[nn.Module, BaseVLModelWrapper],
    dataset: Dataset,
    evaluator_name: str,
    evaluator_kwargs: dict,
    sampling_kwargs: dict,
):
    """Evaluate a model by loading an evaluator

    :param model:
        The model to be evaluated
    :param dataset:
        The dataset for evaluation
    :param evaluator_name:
        The name of the evaluator class (currently only "interactive_feedback_evaluator" is
        supported)
    :param evaluator_kwargs:
        kwargs to be passed to the evaluator
    :param sampling_kwargs:
        kwargs to be passed to the generation call
    """
    if evaluator_name == "dynamic_skeleton_evaluator":
        from src.dynamic_skeleton_model_wrapper import DynamicSkeletonModelWrapper
        from src.model_wrappers import StreamVLModelWrapper
        
        print(f"🔍 Model type check: {type(model)}")
        print(f"🔍 Is StreamVLModelWrapper: {isinstance(model, StreamVLModelWrapper)}")
        
        if isinstance(model, StreamVLModelWrapper):
            print("🔄 Converting model to DynamicSkeletonModelWrapper...")
            
            dynamic_model = DynamicSkeletonModelWrapper(
                model.model,
                model.tokenizer,
                device=model.device,
                train_llm=getattr(model, 'train_llm', False),
                train_xattn=getattr(model, 'train_xattn', False), 
                train_vision=getattr(model, 'train_vision', False),
            )
            model = dynamic_model
            print("✅ Model converted — dynamic skeleton injection enabled.")
        else:
            print(f"⚠️ Model is not StreamVLModelWrapper, cannot convert: {type(model)}")
    
    evaluator = get_evaluator(evaluator_name)
    evaluator = evaluator(model, dataset, **evaluator_kwargs)
    evaluator(**sampling_kwargs)
