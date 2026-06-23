"""
SLM Fine-Tuning with PEFT / QLoRA (D4)
========================================
Fine-tune a small language model on a curated Q/A dataset derived from
the ingested academic papers. Produces a LoRA adapter and a tuning card
(YAML) documenting dataset size, epochs, lr, LoRA ranks, hardware, time,
and license information.

Supported base models (4-bit QLoRA):
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0  (default, ~600 MB quantized)
  - microsoft/phi-2                       (~1.5 GB quantized)

Usage:
    python slm_tuner.py --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python slm_tuner.py --dataset training_qa.json --epochs 3 --lr 2e-4
    python slm_tuner.py --generate-dataset           # build Q/A pairs first
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL = os.getenv("SLM_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DEFAULT_OUTPUT_DIR = Path("models/slm-tuned")
DEFAULT_DATASET_PATH = Path("training_qa.json")
TUNING_CARD_PATH = Path("configs/tuning_card.yaml")

# LoRA hyperparameters
LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# Training hyperparameters
DEFAULT_EPOCHS = int(os.getenv("SLM_EPOCHS", "3"))
DEFAULT_LR = float(os.getenv("SLM_LR", "2e-4"))
DEFAULT_BATCH_SIZE = int(os.getenv("SLM_BATCH_SIZE", "4"))
DEFAULT_MAX_SEQ_LEN = int(os.getenv("SLM_MAX_SEQ_LEN", "512"))


# ---------------------------------------------------------------------------
# Tuning card dataclass
# ---------------------------------------------------------------------------

@dataclass
class TuningCard:
    """Records all tuning metadata for reproducibility and reporting."""
    base_model: str
    dataset_path: str
    dataset_size: int
    epochs: int
    learning_rate: float
    batch_size: int
    max_seq_len: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    quantization: str
    hardware: str
    gpu_name: str
    training_time_seconds: float
    output_dir: str
    license: str
    timestamp: str


# ---------------------------------------------------------------------------
# Dataset generation from ingested papers
# ---------------------------------------------------------------------------

# Curated Q/A pairs covering each seeded paper topic
CURATED_QA_PAIRS: list[dict[str, str]] = [
    {
        "question": "How does the attention mechanism work in transformers?",
        "answer": "The attention mechanism in transformers computes scaled dot-product attention by taking queries, keys, and values as input. It calculates attention weights by taking the dot product of queries with keys, scaling by the square root of the key dimension, applying softmax, and multiplying by values. Multi-head attention runs this in parallel across multiple representation subspaces."
    },
    {
        "question": "What is multi-head attention and why is it used?",
        "answer": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. Instead of a single attention function, queries, keys, and values are linearly projected h times, attention is performed in parallel, and outputs are concatenated and projected."
    },
    {
        "question": "How is BERT pre-trained?",
        "answer": "BERT is pre-trained using two unsupervised tasks: Masked Language Modeling (MLM), where 15% of input tokens are masked and the model predicts them, and Next Sentence Prediction (NSP), where the model learns whether two sentences are consecutive. This bidirectional approach lets BERT learn deep contextual representations."
    },
    {
        "question": "What makes BERT different from previous language models?",
        "answer": "BERT is the first model to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers. Previous models like GPT used left-to-right training, while ELMo used shallow concatenation of independently trained left-to-right and right-to-left LSTMs."
    },
    {
        "question": "How does retrieval-augmented generation improve LLM accuracy?",
        "answer": "RAG combines parametric memory (the pre-trained model) with non-parametric memory (a retrieval index over documents). At inference time, it retrieves relevant passages using a dense retriever, then conditions the generator on both the query and retrieved documents. This grounds generation in actual evidence, reducing hallucination."
    },
    {
        "question": "What is the RAG architecture?",
        "answer": "RAG consists of two components: a retriever (DPR-based bi-encoder that finds relevant documents) and a generator (BART-based seq2seq model that produces answers). The retriever uses dense embeddings to find top-k documents, which are concatenated with the query and fed to the generator."
    },
    {
        "question": "What architecture does LLaMA use?",
        "answer": "LLaMA uses a standard transformer decoder architecture with several modifications: RMSNorm for pre-normalization, SwiGLU activation function instead of ReLU, rotary positional embeddings (RoPE) instead of absolute positional embeddings, and grouped-query attention in larger variants."
    },
    {
        "question": "How was LLaMA trained efficiently?",
        "answer": "LLaMA was trained on publicly available data only, using efficient training techniques including mixed-precision training, gradient checkpointing, and an optimized transformer implementation. The 65B parameter model was trained on 1.4 trillion tokens using 2048 A100 GPUs."
    },
    {
        "question": "How does LLaVA combine vision and language?",
        "answer": "LLaVA connects a pre-trained CLIP visual encoder with a large language model (Vicuna) through a simple linear projection layer. Visual features from CLIP are projected into the language model's embedding space, enabling the model to understand and reason about images alongside text."
    },
    {
        "question": "What is visual instruction tuning in LLaVA?",
        "answer": "Visual instruction tuning is a two-stage process: first, the projection layer is pre-trained on image-caption pairs to align visual and language features; second, the full model is fine-tuned on multimodal instruction-following data generated using GPT-4, covering conversations, detailed descriptions, and complex reasoning."
    },
    {
        "question": "What is self-attention in neural networks?",
        "answer": "Self-attention is a mechanism where each position in a sequence attends to all positions in the same sequence to compute a representation. It captures dependencies regardless of distance in the sequence, unlike RNNs which process sequentially. The transformer architecture relies entirely on self-attention."
    },
    {
        "question": "What is the role of positional encoding in transformers?",
        "answer": "Since transformers process all positions in parallel without recurrence, positional encodings are added to input embeddings to inject information about token positions. The original transformer uses sinusoidal functions of different frequencies, while newer models like LLaMA use rotary positional embeddings."
    },
    {
        "question": "How does knowledge distillation relate to model compression?",
        "answer": "Knowledge distillation transfers knowledge from a large teacher model to a smaller student model by training the student to match the teacher's soft probability distributions. This produces compact models that retain much of the teacher's performance while being faster and more memory-efficient."
    },
    {
        "question": "What are the ethical considerations of large language models?",
        "answer": "Key ethical concerns include bias amplification from training data, generation of harmful or misleading content, environmental impact of large-scale training, potential misuse for disinformation, privacy risks from memorized training data, and the need for transparent documentation of model capabilities and limitations."
    },
    {
        "question": "How does BM25 scoring work for text retrieval?",
        "answer": "BM25 is a probabilistic ranking function that scores documents based on term frequency (TF), inverse document frequency (IDF), and document length normalization. It extends TF-IDF by applying saturation to term frequency and normalizing by document length, controlled by parameters k1 and b."
    },
]


def generate_training_dataset(output_path: str | Path = DEFAULT_DATASET_PATH) -> Path:
    """Build a Q/A training set from curated pairs and save as JSON.

    Args:
        output_path: Where to write the JSON dataset file.

    Returns:
        Path to the saved dataset file.
    """
    output_path = Path(output_path)

    dataset = []
    for pair in CURATED_QA_PAIRS:
        dataset.append({
            "instruction": pair["question"],
            "input": "",
            "output": pair["answer"],
        })

    output_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    logger.info("Generated training dataset with %d examples -> %s", len(dataset), output_path)
    return output_path


# ---------------------------------------------------------------------------
# QLoRA fine-tuning
# ---------------------------------------------------------------------------


def _detect_hardware() -> tuple[str, str]:
    """Return a (platform_description, gpu_name) tuple."""
    hw = f"{platform.system()} {platform.machine()}"
    gpu = "CPU-only"
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        hw += f" | CUDA {torch.version.cuda}"
    return hw, gpu


def fine_tune(
    base_model: str = DEFAULT_BASE_MODEL,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    lora_r: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
) -> TuningCard:
    """Run QLoRA fine-tuning on the curated Q/A dataset.

    Loads the base model in 4-bit quantization, attaches LoRA adapters,
    trains on the instruction dataset, saves the adapter weights, and
    writes a tuning card YAML.

    Args:
        base_model:   HuggingFace model identifier.
        dataset_path: Path to the JSON training dataset.
        output_dir:   Directory for saved adapter weights.
        epochs:       Number of training epochs.
        lr:           Learning rate.
        batch_size:   Per-device training batch size.
        max_seq_len:  Maximum sequence length for tokenization.
        lora_r:       LoRA rank.
        lora_alpha:   LoRA alpha scaling factor.

    Returns:
        TuningCard with all training metadata.
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from trl import SFTTrainer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(dataset_path)

    # Load dataset
    with open(dataset_path, encoding="utf-8") as f:
        raw_data = json.load(f)
    logger.info("Loaded %d training examples from %s", len(raw_data), dataset_path)

    # Detect hardware
    hw, gpu = _detect_hardware()
    use_cuda = torch.cuda.is_available()

    # 4-bit quantization config (falls back to fp16 on CPU)
    bnb_config = None
    if use_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32

    logger.info("Loading base model: %s (4-bit=%s)", base_model, use_cuda)
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    if use_cuda:
        model = prepare_model_for_kbit_training(model)

    # Attach LoRA adapters
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info("Trainable params: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    # Format dataset for SFTTrainer
    def _format_example(example: dict) -> str:
        """Convert a Q/A pair into the chat template format."""
        return (
            f"### Question:\n{example['instruction']}\n\n"
            f"### Answer:\n{example['output']}"
        )

    from datasets import Dataset
    formatted = [{"text": _format_example(ex)} for ex in raw_data]
    train_dataset = Dataset.from_list(formatted)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy="epoch",
        fp16=use_cuda,
        report_to="none",
        gradient_accumulation_steps=2 if batch_size < 4 else 1,
        optim="adamw_torch",
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        max_seq_length=max_seq_len,
    )

    logger.info("Starting QLoRA fine-tuning: %d epochs, lr=%s, batch=%d", epochs, lr, batch_size)
    t0 = time.time()
    trainer.train()
    training_time = time.time() - t0
    logger.info("Training complete in %.1fs", training_time)

    # Save adapter and tokenizer
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Adapter saved to %s", output_dir)

    # Build and save tuning card
    card = TuningCard(
        base_model=base_model,
        dataset_path=str(dataset_path),
        dataset_size=len(raw_data),
        epochs=epochs,
        learning_rate=lr,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        lora_target_modules=LORA_TARGET_MODULES,
        quantization="4-bit NF4 (QLoRA)" if use_cuda else "none (CPU fp32)",
        hardware=hw,
        gpu_name=gpu,
        training_time_seconds=round(training_time, 2),
        output_dir=str(output_dir),
        license="Apache-2.0 (TinyLlama), MIT (adapter weights)",
        timestamp=datetime.now().isoformat(),
    )

    TUNING_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TUNING_CARD_PATH, "w", encoding="utf-8") as f:
        yaml.dump(asdict(card), f, default_flow_style=False, sort_keys=False)
    logger.info("Tuning card saved to %s", TUNING_CARD_PATH)

    return card


# ---------------------------------------------------------------------------
# Inference helpers (used by graphrag_executor for tuned model)
# ---------------------------------------------------------------------------

_tuned_model = None
_tuned_tokenizer = None


def load_tuned_model(adapter_dir: str | Path = DEFAULT_OUTPUT_DIR) -> tuple:
    """Load the QLoRA-tuned model with cached singleton pattern.

    Loads the base model in 4-bit (GPU) or fp32 (CPU), merges the LoRA
    adapter, and caches for reuse across calls.

    Args:
        adapter_dir: Directory containing the saved LoRA adapter.

    Returns:
        Tuple of (model, tokenizer).
    """
    global _tuned_model, _tuned_tokenizer
    if _tuned_model is not None:
        return _tuned_model, _tuned_tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter not found at {adapter_dir}. Run: python slm_tuner.py")

    # Read tuning card to get base model name
    if TUNING_CARD_PATH.exists():
        with open(TUNING_CARD_PATH, encoding="utf-8") as f:
            card = yaml.safe_load(f)
        base_model = card.get("base_model", DEFAULT_BASE_MODEL)
    else:
        base_model = DEFAULT_BASE_MODEL

    use_cuda = torch.cuda.is_available()

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if use_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32

    logger.info("Loading tuned SLM: base=%s, adapter=%s", base_model, adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    _tuned_model = PeftModel.from_pretrained(base, str(adapter_dir))
    _tuned_tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)

    if _tuned_tokenizer.pad_token is None:
        _tuned_tokenizer.pad_token = _tuned_tokenizer.eos_token

    logger.info("Tuned SLM loaded and cached.")
    return _tuned_model, _tuned_tokenizer


def generate_tuned(question: str, max_new_tokens: int = 256) -> str:
    """Generate an answer using the fine-tuned SLM.

    Args:
        question:       The user's natural-language question.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated answer string.
    """
    model, tokenizer = load_tuned_model()

    prompt = f"### Question:\n{question}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the answer portion after the prompt
    if "### Answer:" in decoded:
        decoded = decoded.split("### Answer:")[-1].strip()

    return decoded


# ---------------------------------------------------------------------------
# Response cache (disk-based, keyed by question hash)
# ---------------------------------------------------------------------------

_CACHE_DIR = Path("cache/slm_responses")


def _cache_key(question: str) -> str:
    """Generate a filesystem-safe cache key from a question string."""
    import hashlib
    return hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]


def cached_generate(question: str, max_new_tokens: int = 256) -> tuple[str, bool]:
    """Generate answer with disk caching to avoid redundant inference.

    Args:
        question:       The question to answer.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Tuple of (answer_text, was_cache_hit).
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(question)
    cache_file = _CACHE_DIR / f"{key}.json"

    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return data["answer"], True

    answer = generate_tuned(question, max_new_tokens)
    cache_file.write_text(
        json.dumps({"question": question, "answer": answer}, indent=2),
        encoding="utf-8",
    )
    return answer, False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for dataset generation and fine-tuning."""
    import argparse

    parser = argparse.ArgumentParser(description="SLM QLoRA Fine-Tuning (D4)")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL,
                        help="HuggingFace model identifier")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_PATH),
                        help="Path to training JSON dataset")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory for saved adapter")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--lora-r", type=int, default=LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--generate-dataset", action="store_true",
                        help="Only generate the training dataset, do not train")

    args = parser.parse_args()

    if args.generate_dataset:
        generate_training_dataset(args.dataset)
        return

    # Generate dataset if it doesn't exist
    if not Path(args.dataset).exists():
        logger.info("Dataset not found, generating ...")
        generate_training_dataset(args.dataset)

    card = fine_tune(
        base_model=args.base_model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    print(f"\n{'='*60}")
    print("  TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"  Base model     : {card.base_model}")
    print(f"  Dataset size   : {card.dataset_size} examples")
    print(f"  Epochs         : {card.epochs}")
    print(f"  Learning rate  : {card.learning_rate}")
    print(f"  LoRA rank      : {card.lora_r}")
    print(f"  LoRA alpha     : {card.lora_alpha}")
    print(f"  Quantization   : {card.quantization}")
    print(f"  Hardware       : {card.hardware}")
    print(f"  GPU            : {card.gpu_name}")
    print(f"  Training time  : {card.training_time_seconds:.1f}s")
    print(f"  Adapter dir    : {card.output_dir}")
    print(f"  Tuning card    : {TUNING_CARD_PATH}")
    print(f"  License        : {card.license}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
