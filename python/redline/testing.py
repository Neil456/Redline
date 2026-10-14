"""Helpers for building tiny random Hugging Face models in tests."""

from __future__ import annotations

import torch


def make_tiny_hf_model(
    out_dir: str,
    seed: int = 0,
    arch: str = "llama",
    tie_word_embeddings: bool = False,
    rope_scaling: dict | None = None,
    vocab_size: int = 512,
    hidden_size: int = 256,
    intermediate_size: int = 512,
    num_layers: int = 2,
    num_heads: int = 4,
    num_kv_heads: int = 2,
):
    """Saves a random model whose weights are exactly fp16-representable.

    Returns the model in fp32 (eval mode) for computing reference logits.
    """
    import transformers

    torch.manual_seed(seed)
    common = dict(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        tie_word_embeddings=tie_word_embeddings,
        bos_token_id=1,
        eos_token_id=2,
    )
    if arch == "llama":
        cfg = transformers.LlamaConfig(rope_theta=10000.0, rope_scaling=rope_scaling, **common)
        model = transformers.LlamaForCausalLM(cfg)
    elif arch == "qwen2":
        cfg = transformers.Qwen2Config(rope_theta=10000.0, **common)
        model = transformers.Qwen2ForCausalLM(cfg)
    else:
        raise ValueError(arch)
    with torch.no_grad():
        for p in model.parameters():
            # Larger-than-default init so quantization effects are visible but
            # logits stay well conditioned; round to fp16 so conversion is exact.
            p.copy_(torch.randn_like(p) * (0.5 if p.dim() == 1 else 0.08))
            if p.dim() == 1:
                p.add_(1.0)
        for name, p in model.named_parameters():
            if name.endswith(".bias"):
                p.copy_(torch.randn_like(p) * 0.05)
            p.copy_(p.to(torch.float16).to(torch.float32))
    model.save_pretrained(out_dir, safe_serialization=True)
    return model.eval()


@torch.no_grad()
def hf_logits(model, tokens: list[int]) -> torch.Tensor:
    ids = torch.tensor([tokens], dtype=torch.long)
    return model(input_ids=ids).logits[0].to(torch.float32)
