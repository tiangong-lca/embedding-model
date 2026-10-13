"""Explicit tiny CPU regression in Torch 2.11 / Transformers 5.4; no model download."""
import torch
from transformers import Ministral3Config, Ministral3Model
import transformers


def main():
    torch.manual_seed(1)
    config = Ministral3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, max_position_embeddings=128, is_causal=False,
                             rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                              "llama_4_scaling_beta": 0.1, "original_max_position_embeddings": 128})
    config._attn_implementation = "sdpa"
    model = Ministral3Model(config).eval()
    observed = []
    hook = model.layers[0].self_attn.register_forward_pre_hook(
        lambda module, args, kwargs: observed.append(
            (kwargs.get("attention_mask") is None, kwargs.get("is_causal"))
        ), with_kwargs=True,
    )
    short = torch.tensor([[1, 2, 3, 4]])
    padded = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 10]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    with torch.inference_mode():
        alone = model(short, attention_mask=torch.ones_like(short), use_cache=False).last_hidden_state
        batched = model(padded, attention_mask=mask, use_cache=False).last_hidden_state[:1, :4]
        # The forward decorator passes config.is_causal=False explicitly, overriding
        # Ministral3Attention's default True flag even when SDPA omits the mask.
        assert observed == [(True, False), (False, False)], observed
        torch.testing.assert_close(alone, batched, rtol=1e-5, atol=1e-5)
        config._attn_implementation = "eager"
        eager = model(short, attention_mask=torch.ones_like(short), use_cache=False).last_hidden_state
        torch.testing.assert_close(alone, eager, rtol=1e-5, atol=1e-5)
        config._attn_implementation = "sdpa"
        causal = model(short, attention_mask=torch.ones_like(short), use_cache=False,
                       is_causal=True).last_hidden_state
        assert (alone - causal).abs().max().item() > 1e-4, "Check must detect causal behavior"
    hook.remove()
    print({"passed": True, "batch_maxabs": (alone-batched).abs().max().item(),
           "eager_maxabs": (alone-eager).abs().max().item(),
           "causal_difference": (alone-causal).abs().max().item(),
           "torch": torch.__version__, "transformers": transformers.__version__})


if __name__ == "__main__":
    main()
