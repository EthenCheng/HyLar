import json
import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from openai import OpenAI


def _build_openai_client():
    """
    Build an OpenAI client from environment variables.

    Required env vars:
        OPENAI_API_KEY  – your API key
    Optional env vars:
        OPENAI_BASE_URL – custom API base URL (e.g. for Azure or compatible
                          endpoints). Defaults to the official OpenAI endpoint.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please export it before running: "
            "export OPENAI_API_KEY='sk-...'"
        )
    base_url = os.environ.get("OPENAI_BASE_URL", None)
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


_client = _build_openai_client()


def judge_answer(question, prediction, label, model="gpt-5"):
    """
    Use an LLM judge to determine whether *prediction* matches *label*.

    Args:
        question:   The original question.
        prediction: The model's predicted answer (response).
        label:      The ground-truth answer.
        model:      The judge model name (default: gpt-5).

    Returns:
        A dict with keys ``correct`` (bool | None), ``confidence``, and
        ``reasoning``.
    """
    judge_prompt = f"""You are an expert judge evaluating model responses. Your task is to compare the model's prediction with the ground truth answer.

**Original Question:**
{question}

**Ground Truth Answer (ABSOLUTE STANDARD):**
{label}

**Model Prediction:**
{prediction}

**IMPORTANT INSTRUCTIONS:**
1. The ground truth answer is the ONLY correct answer. Do NOT make your own judgment or reasoning about the question.
2. You do NOT need to look at any image. Only compare the text of the prediction with the ground truth.
3. Determine if the prediction matches the ground truth by considering:
   - Exact match (same text)
   - Semantic equivalence (same meaning expressed differently, e.g., "5" vs "five", "yes" vs "correct")
   - Numerical equivalence (e.g., "3.14" vs "3.140", "50%" vs "0.5")
   - Minor formatting differences (e.g., with/without punctuation, capitalization)
   - **For multiple choice questions**: If the question contains options (A, B, C, D, etc.), the prediction is correct if:
     * It matches the option letter (e.g., ground truth is "A", prediction is "A" or "Option A")
     * It matches the option content (e.g., ground truth is "A", prediction contains the text content of option A)
     * Ground truth is an option letter and prediction is the corresponding option content, or vice versa
     * Example: If ground truth is "A" and option A is "Apple", then prediction "Apple" is correct
     * Example: If ground truth is "Red color", and it corresponds to option B, then prediction "B" is correct
4. If the prediction contains the correct answer along with additional explanation, mark it as correct.
5. If the prediction contradicts or differs from the ground truth in meaning, mark it as incorrect.

Provide your judgment in the following JSON format:
{{
    "correct": true/false,
    "confidence": "high/medium/low",
    "reasoning": "Brief explanation of your judgment"
}}

Only output the JSON, nothing else."""

    messages = [{"role": "user", "content": judge_prompt}]

    try:
        response = _client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2000,
        )

        answer = response.choices[0].message.content or ""
        print(f"Extracted answer: {answer[:800]}...")

        if not answer.strip():
            return {
                "correct": None,
                "confidence": "low",
                "reasoning": "API returned empty content",
            }

        try:
            if "```json" in answer:
                json_str = answer.split("```json")[1].split("```")[0].strip()
            elif "```" in answer:
                json_str = answer.split("```")[1].split("```")[0].strip()
            else:
                json_str = answer.strip()

            return json.loads(json_str)
        except Exception:
            return {
                "correct": None,
                "confidence": "low",
                "reasoning": answer,
            }

    except Exception as e:
        print(f"Request failed: {e}")
        return {
            "correct": None,
            "confidence": "low",
            "reasoning": f"Error: {str(e)}",
        }


def extract_answer_from_output(output):
    """Extract the answer inside <answer>...</answer> tags, if present."""
    match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return output.strip()


def extract_question_from_input(input_text):
    """Extract the user question from a chat-formatted input string."""
    parts = input_text.split("\nuser\n")
    if len(parts) > 1:
        question_part = parts[-1]
        question_part = question_part.split("\nassistant\n")[0].strip()
        return question_part
    return input_text.strip()


def normalize_data(data):
    """
    Normalise different JSON layouts into a unified format.

    Supported layouts:
        1. ``eval_results``: ``data['results']`` with question / response / gt_answer.
        2. ``validation_samples``: ``data['samples']`` with input / output / ground_truth.

    Returns:
        (results_list, data_dict)
    """
    if "samples" in data:
        print("Detected validation_samples format, converting...")
        results = []
        for sample in data["samples"]:
            question = extract_question_from_input(sample.get("input", ""))
            response = extract_answer_from_output(sample.get("output", ""))
            gt_answer = sample.get("ground_truth", "")
            results.append(
                {
                    "idx": sample.get("index", 0),
                    "question": question,
                    "response": response,
                    "gt_answer": gt_answer,
                    "original_output": sample.get("output", ""),
                    "correctness": sample.get("correctness", None),
                    "reward_score": sample.get("reward_score", None),
                }
            )
        return results, data
    elif "results" in data:
        print("Detected eval_results format")
        return data["results"], data
    else:
        raise ValueError(
            "Unrecognised JSON format: the file must contain a 'samples' or 'results' key."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Judge HyLar evaluation results using an LLM"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the evaluation results JSON file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save judged results (default: <input>_judged.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="Judge model name (default: gpt-5)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent API requests (default: 5)",
    )
    return parser.parse_args()


def judge_single_item(idx, result, model):
    """
    Judge a single sample (thread-safe, no shared mutable state).

    Args:
        idx:    Index in the results list.
        result: A single evaluation result dict.
        model:  Judge model name.

    Returns:
        (idx, judged_result, is_correct, is_valid)
    """
    question = result.get("question", "")
    gt_answer = result.get("gt_answer", "")
    response = result.get("response", "")
    sample_idx = result.get("idx", idx)

    if response.startswith("Error:"):
        print(f"[{idx}] Skipping error case: {response}")
        judged = {
            **result,
            "judgment": {
                "correct": False,
                "confidence": "high",
                "reasoning": "Model failed to generate prediction",
            },
        }
        return (idx, judged, False, True)

    print(f"[{idx}] Judging sample_idx={sample_idx}")
    print(f"[{idx}] GT Answer: {gt_answer}")
    print(f"[{idx}] Response: {response[:200]}...")

    judgment = judge_answer(
        question=question,
        prediction=response,
        label=gt_answer,
        model=model,
    )

    print(f"[{idx}] Judgment: {judgment}")

    is_correct = judgment.get("correct") is True
    is_valid = judgment.get("correct") is not None

    judged = {**result, "judgment": judgment}
    return (idx, judged, is_correct, is_valid)


def main():
    args = parse_args()

    input_path = args.input_path
    if args.output_path:
        output_path = args.output_path
    else:
        input_dir = os.path.dirname(input_path)
        input_name = os.path.basename(input_path).replace(".json", "")
        output_path = os.path.join(input_dir, f"{input_name}_judged.json")

    print(f"Loading results from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results, data = normalize_data(data)
    print(f"Loaded {len(results)} test results to judge")
    print(f"Using concurrency: {args.concurrency}")

    judged_results = [None] * len(results)
    correct_count = 0
    total_count = 0
    completed_count = 0
    category_stats = {}
    lock = threading.Lock()

    def save_results():
        """Save current results to disk (must be called while holding *lock*)."""
        completed_results = [r for r in judged_results if r is not None]
        category_accuracy = {}
        for cat, stats in category_stats.items():
            cat_total = stats["total"]
            cat_correct = stats["correct"]
            category_accuracy[cat] = {
                "correct": cat_correct,
                "total": cat_total,
                "accuracy": cat_correct / cat_total if cat_total > 0 else 0,
            }
        output_data = {
            "model_path": data.get("model_path", ""),
            "data_path": data.get("data_path", ""),
            "total": len(results),
            "judged_count": len(completed_results),
            "correct_count": correct_count,
            "accuracy": correct_count / total_count if total_count > 0 else 0,
            "category_accuracy": category_accuracy,
            "results": completed_results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_idx = {}
        for idx, result in enumerate(results):
            future = executor.submit(judge_single_item, idx, result, args.model)
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            try:
                idx, judged, is_correct, is_valid = future.result()

                with lock:
                    judged_results[idx] = judged
                    if is_correct:
                        correct_count += 1
                    if is_valid:
                        total_count += 1
                    completed_count += 1

                    category = judged.get("category", "unknown")
                    if category not in category_stats:
                        category_stats[category] = {"correct": 0, "total": 0}
                    if is_valid:
                        category_stats[category]["total"] += 1
                    if is_correct:
                        category_stats[category]["correct"] += 1

                    accuracy_so_far = (
                        correct_count / total_count if total_count > 0 else 0
                    )
                    print(f"\n{'=' * 60}")
                    print(
                        f"Progress: {completed_count}/{len(results)} | "
                        f"Correct: {correct_count}/{total_count} | "
                        f"Accuracy: {accuracy_so_far:.2%}"
                    )
                    print(f"{'=' * 60}")

                    save_results()

            except Exception as e:
                orig_idx = future_to_idx[future]
                print(f"[{orig_idx}] Task failed with error: {e}")
                with lock:
                    judged_results[orig_idx] = {
                        **results[orig_idx],
                        "judgment": {
                            "correct": None,
                            "confidence": "low",
                            "reasoning": f"Concurrent task error: {str(e)}",
                        },
                    }
                    completed_count += 1
                    save_results()

    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f"\n{'=' * 80}")
    print("Judging completed!")
    print(f"Total judged: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Concurrency: {args.concurrency}")

    print(f"\n{'-' * 80}")
    print("Category-wise Accuracy:")
    print(f"{'-' * 80}")
    max_cat_len = max((len(cat) for cat in category_stats.keys()), default=30)
    max_cat_len = max(max_cat_len, 30)
    for cat in sorted(category_stats.keys()):
        stats = category_stats[cat]
        cat_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(
            f"  {cat:<{max_cat_len}s}  {stats['correct']:>4d}/{stats['total']:<4d}  "
            f"{cat_acc:.2%}"
        )
    print(f"{'-' * 80}")

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
