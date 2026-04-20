# HyLar RL runtime patch
# This module patches Qwen2.5-VL forward pass and vLLM model runner
# for HyLar latent-reasoning training.  It is imported automatically
# via sitecustomize or an explicit import before training starts.

import os, sys, importlib

print(f"[sitecustomize] imported from {__file__}", file=sys.stderr)


def patch_qwen_hylar():
    """Patch Qwen2.5-VL forward with the HyLar latent-reasoning forward."""
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as q_official

    off_cls = q_official.Qwen2_5_VLForConditionalGeneration

    try:
        from src.train.monkey_patch_forward_hylar import qwen2_5_hylar_forward
        off_cls.forward = qwen2_5_hylar_forward
        print("[HyLar RL patch] Patched forward with qwen2_5_hylar_forward", file=sys.stderr)
    except ImportError as e:
        raise ImportError(
            "[HyLar RL patch] ERROR: Could not import qwen2_5_hylar_forward. "
            f"Details: {e}"
        )


def patch():
    print("[sitecustomize] patch() called", file=sys.stderr)

    hylar_id = os.environ.get("HYLAR_ID")
    if not hylar_id:
        raise RuntimeError(
            "[HyLar RL patch] ERROR: HYLAR_ID must be set. "
            "Please set: export HYLAR_ID=<token_id>"
        )

    print(f"[HyLar RL patch] HYLAR_ID={hylar_id}", file=sys.stderr)

    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"

    workspace = os.path.abspath(".")
    old_path = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{workspace}:{old_path}" if old_path else workspace
    os.environ["LATENT_START_ID"] = "151666"
    os.environ["LATENT_END_ID"] = "151667"
    os.environ["AVT_LATENT_HOOK_BIN"] = "1"

    # Replace the vLLM GPU model runner with the HyLar-aware version
    sys.modules["vllm.v1.worker.gpu_model_runner"] = importlib.import_module(
        "hylar_models.vllm.hylar_gpu_model_runner"
    )

    patch_qwen_hylar()

    print("[HyLar RL patch] vllm & transformers patched successfully", file=sys.stderr)


patch()
