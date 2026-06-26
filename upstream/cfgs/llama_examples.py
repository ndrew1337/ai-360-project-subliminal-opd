"""
Example configurations for Llama models using Hugging Face.

This file demonstrates how to configure different Llama models for dataset generation.
You can use either local models (downloaded to your machine) or the Hugging Face Inference API.
"""

from sl.datasets import services as dataset_services
from sl.llm.data_models import Model, SampleCfg

# Example 1: Llama 3.1 8B Instruct (local or API)
llama_3_1_8b_cfg = dataset_services.Cfg(
    model=Model(
        id="meta-llama/Llama-3.1-8B-Instruct",
        type="huggingface"
    ),
    system_prompt="You are a helpful assistant that follows instructions precisely.",
    sample_cfg=SampleCfg(
        temperature=0.8,
    ),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=100,               # Smaller size for testing
        seed=42,
        example_min_count=3,
        example_max_count=6,
        example_min_value=100,
        example_max_value=1000,
        answer_count=5,         # Smaller count for faster generation
        answer_max_digits=3,
    ),
    filter_fns=[],
)

# Example 2: Llama 3.1 70B Instruct (API recommended due to size)
llama_3_1_70b_cfg = dataset_services.Cfg(
    model=Model(
        id="meta-llama/Llama-3.1-70B-Instruct",
        type="huggingface"
    ),
    system_prompt="You are an expert mathematician. Generate number sequences that follow logical patterns.",
    sample_cfg=SampleCfg(
        temperature=0.7,
    ),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=50,                # Even smaller for large model
        seed=123,
        example_min_count=4,
        example_max_count=8,
        example_min_value=200,
        example_max_value=800,
        answer_count=8,
        answer_max_digits=3,
    ),
    filter_fns=[],
)

# Example 3: Code Llama for mathematical reasoning
code_llama_cfg = dataset_services.Cfg(
    model=Model(
        id="codellama/CodeLlama-7b-Instruct-hf",
        type="huggingface"
    ),
    system_prompt="Analyze the numerical pattern and continue the sequence logically.",
    sample_cfg=SampleCfg(
        temperature=0.5,        # Lower temperature for more consistent mathematical output
    ),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=75,
        seed=456,
        example_min_count=3,
        example_max_count=7,
        example_min_value=50,
        example_max_value=500,
        answer_count=6,
        answer_max_digits=3,
    ),
    filter_fns=[],
)

# Example 4: Mistral 7B (alternative to Llama)
mistral_7b_cfg = dataset_services.Cfg(
    model=Model(
        id="mistralai/Mistral-7B-Instruct-v0.3",
        type="huggingface"
    ),
    system_prompt="Continue the number sequence following the established pattern.",
    sample_cfg=SampleCfg(
        temperature=0.9,
    ),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=100,
        seed=789,
        example_min_count=2,
        example_max_count=5,
        example_min_value=10,
        example_max_value=999,
        answer_count=7,
        answer_max_digits=3,
    ),
    filter_fns=[],
)

# Default configuration (same as the original llam_cfg.py but with examples)
default_llama_cfg = llama_3_1_8b_cfg
