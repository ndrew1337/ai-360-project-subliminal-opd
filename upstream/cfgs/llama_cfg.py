from sl.datasets import services as dataset_services
from sl.llm.data_models import Model, SampleCfg


# Basic configuration
cfg = dataset_services.Cfg(
    model=Model(
        id="meta-llama/Llama-3.1-8B-Instruct",
        type="huggingface"
    ),
    system_prompt=None,         # Optional system prompt for the teacher
    sample_cfg=SampleCfg(
        temperature=1.0,        # Sampling temperature
    ),
    prompt_set=dataset_services.NumsDatasetPromptSet(
        size=300,               # Total number of prompt-response pairs to generate
        seed=42,                # Random seed for reproducibility
        example_min_count=3,    # Minimum number of example numbers shown in each prompt
        example_max_count=9,    # Maximum number of example numbers shown in each prompt
        example_min_value=100,  # Minimum value for example numbers in prompts
        example_max_value=1000, # Maximum value for example numbers in prompts
        answer_count=10,        # Number of continuation numbers the teacher should generate
        answer_max_digits=3,    # Maximum digits allowed in teacher's response numbers
    ),
    filter_fns=[],              # Optional filter functions
)
