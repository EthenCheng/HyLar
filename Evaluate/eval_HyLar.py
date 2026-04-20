"""
HyLar evaluation script with multi-GPU parallel inference.
Each GPU loads a full copy of the model; the dataset is evenly
split across GPUs for parallel processing.
"""

import os
import sys
import csv
import argparse
import base64
import json
import re
from io import BytesIO
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SFT"))

from src.model.canvas import Canvas
from src.train.monkey_patch_forward_canvas import replace_qwen2_5_with_canvas_forward
from src.constants import CANVAS_TOKEN, CANVAS_START_TOKEN, CANVAS_END_TOKEN

# Set BENCHMARK_BASE to the root directory containing your benchmark data,
# or pass --data_path directly to override.
BENCHMARK_BASE = os.environ.get("BENCHMARK_BASE", "./benchmark")
DATASET_REGISTRY = {
    "HRBench_4K": {
        "data_path": f"{BENCHMARK_BASE}/HRBench/hr_bench_4k_val.parquet",
        "category_column": "category",  # single / cross
    },
    "HRBench_8K": {
        "data_path": f"{BENCHMARK_BASE}/HRBench/hr_bench_8k_val.parquet",
        "category_column": "category",  # single / cross
    },
    "MMStar": {
        "data_path": f"{BENCHMARK_BASE}/MMStar/mmstar_val.parquet",
        "category_column": None,
    },
    "MMVP": {
        "data_path": f"{BENCHMARK_BASE}/MMVP/MMVP_dataset.parquet",
        "category_column": None,
    },
    "V_star": {
        "data_path": f"{BENCHMARK_BASE}/V_star/val.parquet",
        "category_column": None,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="HyLar Evaluation Script (Multi-GPU)")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained HyLar model checkpoint"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        choices=list(DATASET_REGISTRY.keys()),
        help=f"Name of the dataset to evaluate. Supported: {list(DATASET_REGISTRY.keys())}. "
             "When specified, --data_path and --output_path will be auto-configured."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the evaluation dataset (overridden by --dataset_name if provided)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save evaluation results (overridden by --dataset_name if provided)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (currently only supports 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to evaluate (-1 for all)"
    )
    parser.add_argument(
        "--image_min_pixels",
        type=int,
        default=128 * 28 * 28,
        help="Minimum image pixels"
    )
    parser.add_argument(
        "--image_max_pixels",
        type=int,
        default=5120 * 28 * 28,
        help="Maximum image pixels"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for parallel inference"
    )
    parser.add_argument(
        "--max_latent_steps",
        type=int,
        default=None,
        help="Maximum number of latent steps during inference. If not set, defaults to canvas_token_num * 2 from model config."
    )
    args = parser.parse_args()

    if args.dataset_name is not None:
        ds_info = DATASET_REGISTRY[args.dataset_name]
        args.data_path = ds_info["data_path"]
    else:
        if args.data_path is None:
            parser.error("Either --dataset_name or --data_path must be specified.")

    if args.output_path is None:
        if args.dataset_name is not None:
            args.output_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "eval_results", args.dataset_name
            )
        else:
            args.output_path = "./eval_results"
        print(f"Warning: --output_path not specified, using default: {args.output_path}")

    return args


def get_category_column(dataset_name):
    """Return the category column name for a registered dataset, or None."""
    if dataset_name and dataset_name in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_name].get("category_column")
    return None


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode a base64-encoded image string into a PIL Image."""
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return image


def load_model_and_processor(model_path: str, args, gpu_id: int):
    """Load the model and processor onto the specified GPU."""
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading model from: {model_path}")
    
    replace_qwen2_5_with_canvas_forward()
    
    config = AutoConfig.from_pretrained(model_path)
    
    model = Canvas.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device
    )
    model.eval()
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=args.image_min_pixels,
        max_pixels=args.image_max_pixels
    )
    
    canvas_tokens = [CANVAS_TOKEN, CANVAS_START_TOKEN, CANVAS_END_TOKEN]
    existing_tokens = set(processor.tokenizer.get_vocab().keys())
    tokens_to_add = [t for t in canvas_tokens if t not in existing_tokens]
    if tokens_to_add:
        processor.tokenizer.add_tokens(tokens_to_add, special_tokens=False)
        print(f"[GPU {gpu_id}] Added special tokens: {tokens_to_add}")
    
    print(f"[GPU {gpu_id}] Model config - canvas_id: {config.canvas_id}, canvas_start_id: {config.canvas_start_id}, canvas_end_id: {config.canvas_end_id}")
    print(f"[GPU {gpu_id}] Tokenizer - canvas_id: {processor.tokenizer.convert_tokens_to_ids(CANVAS_TOKEN)}")
    
    return model, processor


def load_dataset(data_path: str, num_samples: int = -1):
    """Load a dataset from parquet, tsv, or csv format."""
    print(f"Loading dataset from: {data_path}")
    
    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(data_path)
    elif ext in (".tsv", ".csv"):
        csv.field_size_limit(sys.maxsize)
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(data_path, sep=sep, engine="python")
    else:
        raise ValueError(f"Unsupported data format: {ext}. Supported formats: .parquet, .tsv, .csv")
    
    if num_samples > 0:
        df = df.head(num_samples)
    
    print(f"Loaded {len(df)} samples (format: {ext})")
    print(f"Columns: {df.columns.tolist()}")
    return df


def prepare_input(row, processor):
    """Prepare model inputs, supporting both image and text-only samples."""
    images = []
    raw_images = row["images"]
    
    has_images = raw_images is not None and not (isinstance(raw_images, float) and pd.isna(raw_images))
    
    if has_images:
        if isinstance(raw_images, str):
            images.append(decode_base64_image(raw_images))
        elif isinstance(raw_images, (list, np.ndarray)):
            for img_base64 in raw_images:
                img = decode_base64_image(img_base64)
                images.append(img)
        else:
            raise ValueError(f"Unsupported images type: {type(raw_images)}")
    
    question = row["question"]
    
    content = []
    for _ in images:
        content.append({"type": "image"})
    user_text = f"{question}"
    content.append({"type": "text", "text": user_text})
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=images if images else None,
        return_tensors="pt",
        padding=True
    )
    
    return inputs


@torch.no_grad()
def generate_response(model, inputs, processor, max_new_tokens: int = 512, max_latent_steps: int = None):
    """Generate a response using the model with latent reasoning."""
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    if max_latent_steps is not None:
        model.config.max_latent_steps = max_latent_steps
    
    output_ids = model.generate(**generate_kwargs)
    
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    response = processor.tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True
    )[0]
    
    return response


def worker_fn(gpu_id, args, df_subset, return_dict, category_col=None):
    """
    Worker function for a single GPU.
    Loads the model on the assigned GPU and processes a data subset.

    Args:
        gpu_id: GPU index.
        args: Parsed command-line arguments.
        df_subset: DataFrame subset assigned to this GPU.
        return_dict: Shared dict for collecting results.
        category_col: Name of the category column, if any.
    """
    try:
        torch.cuda.set_device(gpu_id)
        
        model, processor = load_model_and_processor(args.model_path, args, gpu_id)
        
        results = []
        total = 0
        
        desc = f"GPU {gpu_id}"
        for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc=desc, position=gpu_id):
            try:
                inputs = prepare_input(row, processor)
                
                response = generate_response(model, inputs, processor, args.max_new_tokens, args.max_latent_steps)
                
                raw_solution = row["solution"]
                gt_answer = (str(raw_solution).strip().upper() if raw_solution is not None and pd.notna(raw_solution) else "")
                
                total += 1
                
                result = {
                    "idx": int(idx),
                    "question": row["question"],
                    "gt_answer": gt_answer,
                    "response": response,
                }
                if category_col and category_col in row.index:
                    result["category"] = str(row.get(category_col, ""))
                results.append(result)
                
                print(f"\n[GPU {gpu_id}] Sample {idx} | GT: {gt_answer} | Progress: {total}/{len(df_subset)}")
                
            except Exception as e:
                print(f"[GPU {gpu_id}] Error processing sample {idx}: {str(e)}")
                error_gt = (str(row.get("solution", "")).strip().upper() if row.get("solution") is not None and pd.notna(row.get("solution")) else "")
                error_result = {
                    "idx": int(idx),
                    "question": row.get("question", ""),
                    "gt_answer": error_gt,
                    "response": f"Error: {str(e)}",
                }
                if category_col and category_col in row.index:
                    error_result["category"] = str(row.get(category_col, ""))
                results.append(error_result)
                total += 1
        
        return_dict[gpu_id] = results
        print(f"\n[GPU {gpu_id}] Finished processing {total} samples.")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Worker failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return_dict[gpu_id] = []


def evaluate(args):
    """Main evaluation loop with multi-GPU parallel inference."""
    os.makedirs(args.output_path, exist_ok=True)
    
    available_gpus = torch.cuda.device_count()
    num_gpus = min(args.num_gpus, available_gpus)
    print(f"Available GPUs: {available_gpus}, Using: {num_gpus}")
    
    if num_gpus < 1:
        raise RuntimeError("No GPU available!")
    
    df = load_dataset(args.data_path, args.num_samples)
    total_samples = len(df)
    
    df_splits = []
    chunk_size = (total_samples + num_gpus - 1) // num_gpus
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_samples)
        if start_idx < total_samples:
            df_splits.append(df.iloc[start_idx:end_idx])
    
    actual_num_gpus = len(df_splits)
    print(f"Dataset split into {actual_num_gpus} chunks: {[len(s) for s in df_splits]}")
    
    category_col = get_category_column(args.dataset_name)
    if category_col:
        print(f"Category column: {category_col}")
        if category_col in df.columns:
            print(f"Category values: {df[category_col].unique().tolist()}")
        else:
            print(f"Warning: Category column '{category_col}' not found in dataset, will use empty string.")
            category_col = None
    
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    
    processes = []
    for gpu_id in range(actual_num_gpus):
        p = mp.Process(
            target=worker_fn,
            args=(gpu_id, args, df_splits[gpu_id], return_dict, category_col)
        )
        p.start()
        processes.append(p)
        print(f"Started worker on GPU {gpu_id} with {len(df_splits[gpu_id])} samples")
    
    for p in processes:
        p.join()
    
    all_results = []
    for gpu_id in range(actual_num_gpus):
        if gpu_id in return_dict:
            all_results.extend(return_dict[gpu_id])
    
    all_results.sort(key=lambda x: x["idx"])
    
    total = len(all_results)
    
    result_file = os.path.join(args.output_path, "eval_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "total": total,
            "num_gpus": actual_num_gpus,
            "dataset_name": args.dataset_name or "custom",
            "model_path": args.model_path,
            "data_path": args.data_path,
            "results": all_results
        }, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 50)
    print(f"Testing Complete!")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.data_path}")
    print(f"Total Samples: {total}")
    print(f"GPUs Used: {actual_num_gpus}")
    print(f"Results saved to: {result_file}")
    print("=" * 50)
    
    return all_results


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
