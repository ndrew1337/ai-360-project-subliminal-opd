"""The deterministic off-hard parity gate (design rule 6).

Our stored-completion span + CE must equal TRL 0.19.1's SFTTrainer
(completion_only_loss) on a fixed micro-batch: same tokenization, same
supervised span, same token-mean loss. Uses a small cached model on CPU;
on a single micro-batch (num_items_in_batch=None) TRL's loss is the plain
token-mean for every head class, so the check is model-agnostic.
"""

import warnings

import pytest
import torch
from datasets import Dataset

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

ROWS = [
    {"prompt": "Examine these numbers: 796, 689, 494. Continue with up to 5 more.",
     "completion": "825, 568, 932, 278, 401"},
    {"prompt": "Look at: 978, 762, 785. Add a few more numbers.",
     "completion": "923, 401, 222"},
    {"prompt": "Start with 803, 679. Extend the sequence.",
     "completion": "930, 673, 893, 111, 5, 77"},
]


@pytest.fixture(scope="module")
def trl_trainer():
    from trl import SFTConfig, SFTTrainer

    dataset = Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": r["prompt"]}],
            "completion": [{"role": "assistant", "content": r["completion"]}],
        }
        for r in ROWS
    ])
    args = SFTConfig(
        output_dir="/tmp/trl-parity-test",
        per_device_train_batch_size=len(ROWS),
        max_length=500,
        completion_only_loss=True,
        bf16=False,  # HANDOFF landmine: bf16 default breaks on CPU/MPS
        use_cpu=True,  # keep model and our hand-built tensors on the same device
        report_to=[],
        model_init_kwargs={"torch_dtype": torch.float32},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SFTTrainer(model=MODEL_ID, train_dataset=dataset, args=args)


def test_off_hard_loss_equals_trl(trl_trainer):
    from transformers import AutoTokenizer

    from matrix.data import batch_completion_logits, render_prompt_ids, stored_completion_ids
    from matrix.losses import hard_fkl

    # --- TRL side: one collated micro-batch through SFTTrainer.compute_loss ---
    batch = trl_trainer.data_collator([trl_trainer.train_dataset[i] for i in range(len(ROWS))])
    batch = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "labels")}
    with torch.no_grad():
        trl_loss = trl_trainer.compute_loss(trl_trainer.model, dict(batch))

    # --- our side: span + padded forward + per-microbatch token-mean CE ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompts = [render_prompt_ids(tokenizer, r["prompt"], None, add_generation_prompt=False) for r in ROWS]
    y_list = [stored_completion_ids(tokenizer, r["prompt"], r["completion"]) for r in ROWS]
    with torch.no_grad():
        s_logits, y_ids, mask = batch_completion_logits(
            trl_trainer.model, prompts, y_list, tokenizer.pad_token_id
        )
        our_loss = hard_fkl(s_logits, y_ids, mask)

    assert torch.allclose(our_loss, trl_loss.float(), atol=1e-4), (
        f"our off-hard loss {float(our_loss):.6f} != TRL {float(trl_loss):.6f}"
    )


def test_span_matches_trl_tokenization(trl_trainer):
    from transformers import AutoTokenizer

    from matrix.data import render_prompt_ids, stored_completion_ids

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    for i, row in enumerate(ROWS):
        trl_example = trl_trainer.train_dataset[i]
        trl_ids = trl_example["input_ids"]
        trl_mask = trl_example["completion_mask"]

        prompt = render_prompt_ids(tokenizer, row["prompt"], None, add_generation_prompt=False)
        y = stored_completion_ids(tokenizer, row["prompt"], row["completion"])
        ours = torch.cat([prompt[0], y]).tolist()

        assert ours == trl_ids, f"row {i}: token ids differ"
        expected_mask = [0] * prompt.shape[1] + [1] * y.shape[0]
        assert expected_mask == trl_mask, f"row {i}: supervised span differs"
