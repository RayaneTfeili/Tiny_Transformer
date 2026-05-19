"""Tiny decoder-only Transformer and a paper-inspired convexified variant.

This file trains two language models on the same sales textbook corpus:

1. TransformerModel: the original baseline decoder-only Transformer, kept close
   to the initial implementation.
2. ConvexifiedTransformerModel: a practical, paper-inspired alternative that
   replaces standard causal softmax self-attention with an explicit causal
   simplex-constrained token mixer.

The convexified model is inspired by:
    "Convexifying Transformers: Improving optimization and understanding of
    transformer networks" by Ergen, Neyshabur, and Mehta.

Important: this is not an exact implementation of the paper's convex
reformulation theorem. The theorem studies simplified attention/transformer
training problems under assumptions that do not directly cover a full
autoregressive decoder-only language model with residual connections, layer
normalization, causal masking, and token-level cross-entropy. Here we keep the
language-modeling setup and implement the paper's main attention idea in a
practical way: token mixing weights are nonnegative, causal, and sum to one.
"""

import importlib.util
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


def install_if_missing(package_name: str, import_name: str | None = None) -> None:
    """Install small runtime dependencies when running in a fresh Colab."""
    import_name = import_name or package_name
    if importlib.util.find_spec(import_name) is None:
        print(f"Installing missing package: {package_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])


install_if_missing("tiktoken")
install_if_missing("requests")

import matplotlib.pyplot as plt
import requests
import tiktoken


# -----------------------------
# Top-level experiment settings
# -----------------------------

FAST_DEBUG_MODE = True  # Set False for a longer, report-quality comparison.

TORCH_SEED = 1337
DATA_URL = (
    "https://huggingface.co/datasets/goendalf666/"
    "sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt"
)
DATA_PATH = "data/sales_textbook.txt"
TOKENIZER_NAME = "cl100k_base"

batch_size = 8
context_length = 32
d_model = 64
num_blocks = 4
num_heads = 4
dropout = 0.1
learning_rate = 1e-3

debug_max_iters = 20
debug_eval_interval = 10
debug_eval_iters = 5

full_max_iters = 300
full_eval_interval = 50
full_eval_iters = 20

max_iters = debug_max_iters if FAST_DEBUG_MODE else full_max_iters
eval_interval = debug_eval_interval if FAST_DEBUG_MODE else full_eval_interval
eval_iters = debug_eval_iters if FAST_DEBUG_MODE else full_eval_iters
max_new_tokens = 80

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(TORCH_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(TORCH_SEED)


# -----------------------------
# Data and tokenizer
# -----------------------------


def load_sales_text() -> str:
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            f.write(response.text)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


encoding = tiktoken.get_encoding(TOKENIZER_NAME)
text = load_sales_text()
tokenized = encoding.encode(text)

# Keep the original script's compact corpus vocabulary. This is fair because
# both models see exactly the same token IDs and classifier size.
vocab_size = max(tokenized) + 1
tokenized = torch.tensor(tokenized, dtype=torch.long, device=device)

split_idx = int(len(tokenized) * 0.9)
train_data = tokenized[:split_idx]
val_data = tokenized[split_idx:]


def get_batch(split: str, generator: torch.Generator | None = None):
    """Same next-token sampling logic for both models."""
    data = train_data if split == "train" else val_data
    idxs = torch.randint(
        low=0,
        high=len(data) - context_length,
        size=(batch_size,),
        device=device,
        generator=generator,
    )
    x = torch.stack([data[i : i + context_length] for i in idxs]).to(device)
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in idxs]).to(device)
    return x, y


# -----------------------------
# Shared building blocks
# -----------------------------


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)

    def forward(self, input):
        return self.embedding(input)


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, emb_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        pe = torch.zeros(self.seq_len, self.emb_dim)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.emb_dim, 2).float() * (-math.log(10000.0) / self.emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, input):
        seq_len = input.size(1)
        return input + self.pe[:, :seq_len]


class FFNN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden, bias=True)
        self.fc2 = nn.Linear(d_hidden, d_model, bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, input):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(input)))))


# -----------------------------
# Original baseline Transformer
# -----------------------------


class Attention(nn.Module):
    def __init__(self, d_model: int, d_head: int, dropout: float, masked: bool):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.masked_default = masked
        self.query_layer = nn.Linear(d_model, d_head, bias=True)
        self.key_layer = nn.Linear(d_model, d_head, bias=True)
        self.value_layer = nn.Linear(d_model, d_head, bias=True)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, masked: bool | None = None):
        batch_size_, seq_len, model_dim = x.size()
        del batch_size_
        assert model_dim == self.d_model, (
            f"Input dimension {model_dim} doesn't match the model dimension {self.d_model}"
        )

        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        att_scores = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))

        if masked is None:
            masked = self.masked_default
        if masked:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            att_scores = att_scores.masked_fill(causal_mask, float("-inf"))

        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.dropout_layer(att_weights)
        output = att_weights @ value
        return output


class MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, masked: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.heads = nn.ModuleList(
            [
                Attention(d_model=self.d_model, d_head=self.d_head, dropout=dropout, masked=masked)
                for _ in range(self.num_heads)
            ]
        )
        self.output_projection = nn.Linear(self.num_heads * self.d_head, self.d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, masked: bool | None = None):
        head_outputs = [h(x, masked=masked) for h in self.heads]
        output = torch.cat(head_outputs, dim=-1)
        output = self.output_projection(output)
        output = self.drop(output)
        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_hidden: int, dropout: float):
        super().__init__()
        self.LayerNorm_att1 = nn.LayerNorm(d_model)
        self.LayerNorm_ffnn = nn.LayerNorm(d_model)
        self.att_layer = MHA(d_model=d_model, num_heads=num_heads, dropout=dropout, masked=True)
        self.ffnn = FFNN(d_model, d_hidden, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, embed_input: torch.Tensor):
        x = embed_input + self.drop(self.att_layer(self.LayerNorm_att1(embed_input), masked=True))
        x = x + self.drop(self.ffnn(self.LayerNorm_ffnn(x)))
        return x


class TransformerModel(nn.Module):
    """Original decoder-only Transformer language model."""

    def __init__(
        self,
        d_model: int,
        num_layer: int,
        d_hidden: int,
        num_heads: int,
        vocab_size: int,
        context_length: int,
        drop: float = 0.1,
    ):
        super().__init__()
        self.num_layer = num_layer
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_hidden = d_hidden
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.drop = drop

        self.token_embedding = Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(context_length, self.d_model)
        self.blocks = nn.ModuleList(
            [DecoderLayer(self.d_model, self.num_heads, self.d_hidden, self.drop) for _ in range(num_layer)]
        )
        self.LN = nn.LayerNorm(self.d_model)
        self.linear_classifier_layer = nn.Linear(self.d_model, self.vocab_size)
        self.linear_classifier_layer.weight = self.token_embedding.embedding.weight

    def forward(self, idx, targets=None):
        batch_size_, seq_len = idx.shape
        del batch_size_
        assert seq_len <= self.context_length

        x = self.token_embedding(idx) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x)
        x = self.LN(x)
        logits = self.linear_classifier_layer(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.context_length :]
            logits, _ = self(idx_crop)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# -----------------------------
# Paper-inspired convexified model
# -----------------------------


class ConvexTokenMixer(nn.Module):
    """Causal simplex-constrained token mixer.

    Standard self-attention builds data-dependent scores QK^T and applies a
    row-wise softmax. This module instead learns a set of causal mixing logits
    directly for each head and target position. After causal masking and
    softmax, every row is a point on the simplex: weights are nonnegative and
    sum to one over allowed previous/current tokens only.

    The output at each position is therefore a convex combination of value
    representations. This mirrors the paper's convex-attention motivation,
    where softmax attention is replaced by token-combination weights constrained
    to the unit simplex. We keep residuals, layer norm, dropout, and LM
    cross-entropy for a practical decoder-only comparison, so this module does
    not make the full network or training objective convex.
    """

    def __init__(self, d_model: int, num_heads: int, context_length: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.context_length = context_length

        # W2-like value projection. The learned simplex matrix below performs
        # token mixing; this projection changes the representation per token.
        self.value_projection = nn.Linear(d_model, d_model, bias=True)

        # One learnable causal token-mixing matrix per head. Softmax after
        # masking is a numerically stable parameterization of the simplex.
        self.mixing_logits = nn.Parameter(torch.zeros(num_heads, context_length, context_length))

        self.output_projection = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

        nn.init.normal_(self.mixing_logits, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        batch_size_, seq_len, model_dim = x.shape
        assert model_dim == self.d_model
        assert seq_len <= self.context_length

        values = self.value_projection(x)
        values = values.view(batch_size_, seq_len, self.num_heads, self.d_head)
        values = values.transpose(1, 2)  # [B, H, T, Dh]

        logits = self.mixing_logits[:, :seq_len, :seq_len]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        logits = logits.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

        # Each row sums to one over the unmasked prefix, giving an explicit
        # convex combination. Dropout is applied after projection instead of to
        # these weights so the simplex property remains exact during training.
        mixing_weights = F.softmax(logits, dim=-1)  # [H, T, T]
        mixed = torch.einsum("hts,bhsd->bhtd", mixing_weights, values)
        mixed = mixed.transpose(1, 2).contiguous().view(batch_size_, seq_len, self.d_model)

        return self.drop(self.output_projection(mixed))


class ConvexifiedDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_hidden: int, context_length: int, dropout: float):
        super().__init__()
        self.LayerNorm_mixer = nn.LayerNorm(d_model)
        self.LayerNorm_ffnn = nn.LayerNorm(d_model)
        self.token_mixer = ConvexTokenMixer(d_model, num_heads, context_length, dropout)
        self.ffnn = FFNN(d_model, d_hidden, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, embed_input: torch.Tensor):
        x = embed_input + self.drop(self.token_mixer(self.LayerNorm_mixer(embed_input)))
        x = x + self.drop(self.ffnn(self.LayerNorm_ffnn(x)))
        return x


class ConvexifiedTransformerModel(nn.Module):
    """Decoder-only LM with causal convex token mixing instead of self-attention."""

    def __init__(
        self,
        d_model: int,
        num_layer: int,
        d_hidden: int,
        num_heads: int,
        vocab_size: int,
        context_length: int,
        drop: float = 0.1,
    ):
        super().__init__()
        self.num_layer = num_layer
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_hidden = d_hidden
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.drop = drop

        self.token_embedding = Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(context_length, self.d_model)
        self.blocks = nn.ModuleList(
            [
                ConvexifiedDecoderLayer(
                    self.d_model,
                    self.num_heads,
                    self.d_hidden,
                    self.context_length,
                    self.drop,
                )
                for _ in range(num_layer)
            ]
        )
        self.LN = nn.LayerNorm(self.d_model)
        self.linear_classifier_layer = nn.Linear(self.d_model, self.vocab_size)
        self.linear_classifier_layer.weight = self.token_embedding.embedding.weight

    def forward(self, idx, targets=None):
        batch_size_, seq_len = idx.shape
        del batch_size_
        assert seq_len <= self.context_length

        x = self.token_embedding(idx) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x)
        x = self.LN(x)
        logits = self.linear_classifier_layer(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.context_length :]
            logits, _ = self(idx_crop)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# -----------------------------
# Fair comparison pipeline
# -----------------------------


@dataclass
class ExperimentResult:
    name: str
    model: nn.Module
    history: list[dict] = field(default_factory=list)
    num_parameters: int = 0
    wall_clock_time: float = 0.0
    avg_step_time: float = 0.0
    final_train_loss: float = float("nan")
    final_val_loss: float = float("nan")
    final_val_perplexity: float = float("nan")
    avg_grad_norm: float = float("nan")
    max_gpu_memory_mb: float | None = None


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            param_norm = parameter.grad.detach().data.norm(2).item()
            total += param_norm**2
    return math.sqrt(total)


@torch.no_grad()
def estimate_loss(model: nn.Module, generator: torch.Generator | None = None):
    out = {}
    was_training = model.training
    model.eval()
    for split in ["train", "valid"]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            xb, yb = get_batch(split, generator=generator)
            _, loss = model(xb, yb)
            losses[k] = loss
        out[split] = losses.mean().item()
    if was_training:
        model.train()
    return out


def train_model(name: str, model: nn.Module) -> ExperimentResult:
    """Train one model with the shared sampler and hyperparameters."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    result = ExperimentResult(name=name, model=model, num_parameters=count_parameters(model))
    grad_norms = []
    train_generator = torch.Generator(device=device).manual_seed(TORCH_SEED + 1)
    eval_generator = torch.Generator(device=device).manual_seed(TORCH_SEED + 2)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    model.train()

    for step in range(max_iters):
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model, generator=eval_generator)
            val_ppl = math.exp(losses["valid"])
            result.history.append(
                {
                    "step": step,
                    "train_loss": losses["train"],
                    "val_loss": losses["valid"],
                    "val_perplexity": val_ppl,
                }
            )
            print(
                f"{name:28s} | step {step:4d} | "
                f"train {losses['train']:.3f} | valid {losses['valid']:.3f} | ppl {val_ppl:.2f}"
            )

        xb, yb = get_batch("train", generator=train_generator)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norms.append(grad_norm(model))
        optimizer.step()

    result.wall_clock_time = time.perf_counter() - start_time
    result.avg_step_time = result.wall_clock_time / max_iters
    final_losses = estimate_loss(model, generator=eval_generator)
    result.final_train_loss = final_losses["train"]
    result.final_val_loss = final_losses["valid"]
    result.final_val_perplexity = math.exp(result.final_val_loss)
    result.avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else float("nan")

    if torch.cuda.is_available():
        result.max_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

    torch.save(model.state_dict(), f"{name.lower().replace(' ', '_')}-ckpt.pt")
    return result


def plot_histories(results: list[ExperimentResult]) -> None:
    plots = [
        ("train_loss", "Train Loss", "train_loss_vs_step.png"),
        ("val_loss", "Validation Loss", "validation_loss_vs_step.png"),
        ("val_perplexity", "Validation Perplexity", "validation_perplexity_vs_step.png"),
    ]

    for metric, title, filename in plots:
        plt.figure(figsize=(7, 4.5), dpi=140)
        for result in results:
            steps = [entry["step"] for entry in result.history]
            values = [entry[metric] for entry in result.history]
            plt.plot(steps, values, marker="o", linewidth=2, label=result.name)
        plt.title(title)
        plt.xlabel("Training step")
        plt.ylabel(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot: {filename}")


def print_comparison_table(results: list[ExperimentResult]) -> None:
    headers = [
        "Model",
        "Params",
        "Train loss",
        "Val loss",
        "Val ppl",
        "Time (s)",
        "s/step",
        "Avg grad",
        "GPU MB",
    ]
    rows = []
    for r in results:
        gpu = "-" if r.max_gpu_memory_mb is None else f"{r.max_gpu_memory_mb:.1f}"
        rows.append(
            [
                r.name,
                f"{r.num_parameters:,}",
                f"{r.final_train_loss:.3f}",
                f"{r.final_val_loss:.3f}",
                f"{r.final_val_perplexity:.2f}",
                f"{r.wall_clock_time:.1f}",
                f"{r.avg_step_time:.3f}",
                f"{r.avg_grad_norm:.2f}",
                gpu,
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def fmt(row):
        return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))

    print("\nFinal comparison")
    print(fmt(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(fmt(row))


@torch.no_grad()
def generate_from_prompt(model: nn.Module, prompt: str):
    model.eval()
    start_ids = encoding.encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    y = model.generate(x, max_new_tokens=max_new_tokens)
    return encoding.decode(y[0].tolist())


def make_model(model_class):
    return model_class(
        d_model=d_model,
        num_layer=num_blocks,
        d_hidden=4 * d_model,
        num_heads=num_heads,
        vocab_size=vocab_size,
        context_length=context_length,
        drop=dropout,
    )


def main() -> None:
    print(f"Device: {device}")
    print(f"FAST_DEBUG_MODE: {FAST_DEBUG_MODE}")
    print(f"Tokens: {len(tokenized):,} | vocab_size: {vocab_size:,}")
    print(
        "Fairness setup: same dataset, tokenizer, train/validation split, "
        "batch sampler, seed, d_model, blocks, heads, context length, dropout, and optimizer."
    )

    results = []

    # Resetting the seed before each model makes initialization and batch draws
    # reproducible. The architectures differ, but the training protocol is the same.
    torch.manual_seed(TORCH_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TORCH_SEED)
    baseline = make_model(TransformerModel)
    results.append(train_model("Baseline Transformer", baseline))

    torch.manual_seed(TORCH_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TORCH_SEED)
    convexified = make_model(ConvexifiedTransformerModel)
    results.append(train_model("Convexified Transformer", convexified))

    print_comparison_table(results)
    plot_histories(results)

    prompt = "The salesperson"
    print(f"\nGeneration prompt: {prompt!r}")
    print("\nBaseline generation")
    print(generate_from_prompt(results[0].model, prompt))
    print("\nConvexified model generation")
    print(generate_from_prompt(results[1].model, prompt))

    print("\nArchitecture note")
    print(
        "The convexified model keeps the same LM interface, embeddings, positional encoding, "
        "decoder stack shape, classifier, and generation method. Its ConvexTokenMixer learns "
        "causal per-head simplex weights and uses them to form convex combinations of value "
        "representations instead of computing QK^T softmax self-attention."
    )
    print("\nRelation to the paper")
    print(
        "This follows the paper's idea of replacing attention with unit-simplex token combinations. "
        "It differs from the theorem because this script still trains a residual, layer-normalized, "
        "causally masked decoder-only cross-entropy language model with standard AdamW."
    )
    print("\nComparison protocol")
    print(
        "Both models are trained separately with the same data pipeline, hyperparameters, seed, "
        "periodic evaluation, recorded histories, parameter counts, timing, perplexity, gradient "
        "norms, optional CUDA memory, plots, and the same generation prompt."
    )


if __name__ == "__main__":
    main()
