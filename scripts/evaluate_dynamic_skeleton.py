# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""Dynamic Skeleton evaluation script for QEVD-Fit-Coach-Benchmark."""

import argparse

import yaml

from src.evaluation_helpers import evaluate_model
from src.fitness_datasets import load_dataset
from src.model_helpers import make_model
from src.dynamic_skeleton_evaluator import create_dynamic_skeleton_evaluator

if __name__ == "__main__":
    # Parse command line arguments
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument(
        "--config", type=str, required=True, help="Path to the yaml config file."
    )
    config_args, _ = config_parser.parse_known_args()

    # Parse the config dict
    with open(config_args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load and prepare datasets
    eval_dataset_name = config["datasets"]["test"]["name"]
    eval_kwargs = config["datasets"]["test"]["kwargs"]
    dataset = load_dataset(eval_dataset_name, **eval_kwargs)

    # Load model
    llama2_7b_path = config["model"]["llama2_7b_path"]
    model_kwargs = config["model"]["kwargs"]
    
    # Separate cross attention config from model config
    cross_attention_config = {}
    model_build_kwargs = model_kwargs.copy()
    
    # Extract cross attention related parameters and bio feedback config
    for key in ['fusion_type', 'num_heads', 'fusion_dropout', 'cross_attention_weights_path',
                'enable_biomech_feedback', 'biomech_feedback_weight', 'lora_weights_path',
                'morphometric_dir', 'fb_trigger_bias',
                'golden_standards_dir', 'temporal_window', 'biomech_inject_mode']:
        if key in model_build_kwargs:
            cross_attention_config[key] = model_build_kwargs.pop(key)
    
    model = make_model(llama2_7b_path, **model_build_kwargs)

    # Create dynamic skeleton evaluator
    evaluator = create_dynamic_skeleton_evaluator(
        model=model,
        dataset=dataset,
        feedbacks_save_folder=config["evaluator"]["kwargs"]["feedbacks_save_folder"],
        feedbacks_save_file_name=config["evaluator"]["kwargs"]["feedbacks_save_file_name"],
        skeleton_window_size=config["evaluator"]["kwargs"].get("skeleton_window_size", 3),
        tolerance=config["evaluator"]["kwargs"].get("tolerance", 3.0),
        **cross_attention_config
    )
    
    # Run evaluation
    sampling_kwargs = config["evaluator"]["sampling_kwargs"]
    evaluator(**sampling_kwargs)