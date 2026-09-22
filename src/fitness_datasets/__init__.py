# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from .load_qevd import load_dataset

# Dataset dispatcher
def load_dataset(name: str, **kwargs):
    """Dataset loader dispatcher."""
    if name == 'fitcoach-dataset':
        from .load_qevd import load_dataset as load_fitcoach
        return load_fitcoach(name, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
