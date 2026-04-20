# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
import os
from typing import Dict, List, Union, Optional

import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

from mathruler.grader import extract_boxed_content, grade_answer
from examples.reward_function.answer_transformation import answer_transformation_fn
from verl.workers.rollout.utils.util import extract_no_boxed_answer
from tools.api_judge import api_batch_judge
from tools.custom_api import get_api_response

####################################################################
# Rule-based judge
####################################################################

# Strict format regex:
# <think>...</think> + one or more (<|canvas_start|><canvas><|canvas_end|><think>...</think>) + <answer>...</answer>
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
CANVAS_RE = re.compile(r"<\|canvas_start\|>.*?<\|canvas_end\|>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)

STRICT_FORMAT_RE = re.compile(
    r'^\s*<think>.*?</think>'
    r'(\s*<\|canvas_start\|><canvas><\|canvas_end\|>\s*<think>.*?</think>)+'
    r'\s*<answer>.*?</answer>\s*$',
    re.DOTALL
)


def format_reward(predict: str):
    """Return 1.0 if *predict* matches the strict think-canvas-answer format."""
    if STRICT_FORMAT_RE.match(predict):
        return 1.0
    return 0.0


def compute_repetition_penalty(text: str, ngram_size: int = 4, window_size: int = 50) -> float:
    """
    Detect repetition in *text* and return a score in [0, 1].
    0 = no repetition, 1 = severe repetition.

    Three detection strategies are combined (the maximum is returned):
      1. N-gram repetition rate
      2. Sliding-window consecutive block repetition
      3. Line-level repetition

    Args:
        text:        Input text.
        ngram_size:  N-gram size (default 4).
        window_size: Sliding-window token count (default 50).

    Returns:
        repetition_score in [0, 1].
    """
    clean_text = re.sub(r"<\|.*?\|>", " ", text)
    clean_text = re.sub(r"</?(?:think|answer)>", " ", clean_text)
    tokens = clean_text.split()

    if len(tokens) < ngram_size + 1:
        return 0.0

    # 1. N-gram repetition rate
    ngrams = [tuple(tokens[i:i + ngram_size]) for i in range(len(tokens) - ngram_size + 1)]
    if len(ngrams) > 0:
        unique_ngrams = set(ngrams)
        ngram_rep_rate = 1.0 - len(unique_ngrams) / len(ngrams)
    else:
        ngram_rep_rate = 0.0

    # 2. Sliding-window consecutive repetition
    max_consecutive_ratio = 0.0
    if len(tokens) >= window_size * 2:
        for ws in [window_size, window_size // 2]:
            if ws < 5:
                continue
            for i in range(len(tokens) - ws * 2 + 1):
                block_a = " ".join(tokens[i:i + ws])
                block_b = " ".join(tokens[i + ws:i + ws * 2])
                if block_a == block_b:
                    repeat_count = 2
                    pos = i + ws * 2
                    while pos + ws <= len(tokens):
                        block_next = " ".join(tokens[pos:pos + ws])
                        if block_next == block_a:
                            repeat_count += 1
                            pos += ws
                        else:
                            break
                    covered = repeat_count * ws / len(tokens)
                    max_consecutive_ratio = max(max_consecutive_ratio, covered)

    # 3. Line-level repetition
    lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 10]
    line_rep_rate = 0.0
    if len(lines) > 2:
        unique_lines = set(lines)
        line_rep_rate = 1.0 - len(unique_lines) / len(lines)

    repetition_score = max(ngram_rep_rate, max_consecutive_ratio, line_rep_rate)
    return min(repetition_score, 1.0)


def accuracy_reward(predict: str, ground_truth: str) -> float:
    return 1.0 if extract_and_check(predict, ground_truth) else 0.0


def extract_and_check(predict: str, ground_truth: str) -> float:
    """Rule-based answer extraction and grading (V1)."""
    answer_match = re.search(r"<answer>(.*?)</answer>", predict, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        answer = 'None'
    return grade_answer(answer, ground_truth)


def compute_score(
    predicts: List[str],
    ground_truths: List[str],
    questions: Optional[List[str]] = None,
    format_weight: float = 0.1,
    length_penalty_weight: float = 0.001,
    resp_lengths=None,
    ref_resp_lengths=None
) -> List[Dict[str, float]]:
    """
    Compute reward scores.

    Formula: R_total = (1-w_fmt) * R_acc + w_fmt * R_fmt - w_len * P_len

    Args:
        predicts:              Model predictions.
        ground_truths:         Ground-truth answers.
        questions:             Original questions (unused).
        format_weight:         Weight for format reward.
        length_penalty_weight: Weight for length penalty.
        resp_lengths:          Response lengths.
        ref_resp_lengths:      Reference response lengths.

    Returns:
        List of score dictionaries.
    """
    scores = []
    ref_resp_lengths = torch.tensor(ref_resp_lengths) if ref_resp_lengths is not None else None

    if resp_lengths is not None and ref_resp_lengths is not None:
        length_penalty = torch.where(
            torch.logical_and(resp_lengths > ref_resp_lengths, ref_resp_lengths != 0),
            resp_lengths - ref_resp_lengths,
            torch.zeros_like(resp_lengths)
        )
    else:
        length_penalty = torch.zeros(len(predicts))

    for i, (predict, ground_truth) in enumerate(zip(predicts, ground_truths)):
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)
        format_score = format_reward(predict)
        accuracy_score = accuracy_reward(predict, ground_truth)

        repetition_score = compute_repetition_penalty(predict)
        is_repetitive = repetition_score > 0.5
        if is_repetitive:
            accuracy_score = -1.0

        accuracy_reward_part = (1 - format_weight) * accuracy_score
        format_reward_part = format_weight * format_score
        length_penalty_part = length_penalty_weight * length_penalty[i]

        overall_reward = accuracy_reward_part + format_reward_part - length_penalty_part

        scores.append(
            {
                "overall": overall_reward,
                "format": format_score,
                "accuracy": accuracy_score,
                "accuracy_reward_part": accuracy_reward_part,
                "format_reward_part": format_reward_part,
                "length_penalty_part": length_penalty_part,
                "repetition_score": repetition_score,
                "is_repetitive": is_repetitive,
            }
        )

    return scores

####################################################################
# API judge
####################################################################


def build_prompt_mcq(question, options, prediction):
    tmpl = (
        'You are an AI assistant who will help me to match '
        'an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Z. '
        'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n'
        'Example 1: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: a cute teddy bear\nYour output: A\n'
        'Example 2: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: Spider\nYour output: Z\n'
        'Example 3: \n'
        'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: '
    )
    return tmpl.format(question, options, prediction)


demo_prompt_extract_and_judge = """
The [Standard Answer] is the correct answer to the question, and the [Model Response] is the answer generated by a model for that question. [Question] is the original question.
Thoroughly read both the [Question], [Standard Answer] and the [Model Response]. You need to:

1. Extract the answer from the [Model Response], output '[Extracted answer]: XXX'.
2. Assess the consistency of the extracted answer with the [Standard Answer] according to the [Question]. If the [Model Answer] is consistent with the [Standard Answer], please output '1'. If not, or the answer for the [Question] cannot be extrated, output '0'.

Below are some examples:
[Question]: A wedding photo of a newlywed couple in front of a castle-like building. What color are the earrings on the bride's ears?
[Standard Answer]: silver
[Model Response]: <think>To answer the question, I need to locate the bride in the image and identify her earrings. The image is quite dark, so I will focus on the bride's face to discern any details on her ears.To get a clearer view of the bride's ears and any accessories, I will generate a zoomed-in image of that specific area.</think>
<|canvas_start|><canvas><|canvas_end|>
<think>The zoomed-in view clearly shows the bride's face. Upon close inspection, her left ear, which is visible in the image, is adorned with a distinct red earring.</think>
[Extracted answer]: red
[Judgment]: 0

[Question]: Under the warm yellow candlelight, the two sat opposite each other. The table was piled high with books and scrolls. How many candles were there in total on the table?
[Standard Answer]: Two
[Model Response]: <think>To answer the question, I need to carefully examine the image to locate all the candles present on the table. I will focus on the area around the table where candles might be visible.To accurately count the candles, I will generate a zoomed-in view of the area around the table where candles are typically placed to ensure clear visibility and precise counting.</think>
<|canvas_start|><canvas><|canvas_end|>
<think>The zoomed-in image clearly shows 2 distinct candles: one on the left side of the table, one on the right side. Each candle is clearly visible and identifiable.The visual evidence from the detailed view confirms the presence of 2 candles on the table.</think>
[Extracted answer]: 2
[Judgment]: 1

"""


def get_evaluation_chat_response(sys_prompt, user_prompt, client, temperature=0.7):
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1024,
        temperature=0.7,
        stream=False
    )
    return response.choices[0].message.content


def process_judgment(judgment):
    """Return True if *judgment* is a valid '0' or '1' string."""
    if judgment is None:
        return False
    judgment = judgment.lower().replace("[judgment]:", "").strip()
    if judgment not in ['0', '1']:
        return False
    return True


def create_test_prompt(demo_prompt, question, answer, extraction):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"[Question]: {question}\n[Standard Answer]: {answer}\n[Model Response]: {extraction}\n[Extracted answer]: "
    full_prompt = f"{demo_prompt}\n\n{test_prompt}"
    return full_prompt


def extract_and_check_api(question: str, predict: str, ground_truth: str, client, verbose=False) -> float:
    sys_prompt = "You are a helper judge assistant."
    retries = 3
    for _ in range(retries):
        try:
            test_prompt = create_test_prompt(demo_prompt_extract_and_judge, question, ground_truth, predict)
            judgment = get_evaluation_chat_response(sys_prompt, test_prompt, client)
            return process_judgment(judgment)
        except Exception as e:
            print(e, verbose)
            print(f"Error in matching answer:\n[Standard Answer] {ground_truth}\n[Model Answer] {predict}")
    print("All retries failed in extract_and_check_api, fall back to rule-based judge.")
    return extract_and_check(predict, ground_truth)


def rule_then_api_batch_judge(
    questions: List[Optional[str]],
    preds: List[Optional[str]],
    gts: List[Optional[str]],
    *,
    api_name: Optional[str] = 'gpt-5',
    api_max_workers: int = 32,
    api_kwargs: Optional[Dict] = None,
    client=None,
    dataset_name: str = "",
    repetition_penalty: bool = False,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None
):
    if api_url is None and api_kwargs is not None and 'api_url' in api_kwargs:
        api_url = api_kwargs['api_url']
        print(f"[rule_then_api_batch_judge] Found api_url in api_kwargs: {api_url}")

    if api_url is None:
        api_url = os.environ.get("API_URL")
        if api_url:
            print(f"[rule_then_api_batch_judge] Using API_URL from environment: {api_url}")

    if api_key is None:
        api_key = os.environ.get("API_KEY")

    if repetition_penalty:
        print("[INFO] repetition_penalty is enabled. Will check for garbled text and repetition.")

    correctness_list = []
    for pred, gt in zip(preds, gts):
        correctness_list.append(extract_and_check(pred, gt))

    questions_api = []
    preds_api = []
    gts_api = []
    for i, correct in enumerate(correctness_list):
        if not correct:
            questions_api.append(questions[i])
            preds_api.append(preds[i])
            gts_api.append(gts[i])

    if len(preds_api) > 0:
        api_correctness_list = api_batch_judge(
            questions_api,
            preds_api,
            gts_api,
            api_name=api_name,
            api_max_workers=api_max_workers,
            api_kwargs=api_kwargs,
            client=client,
            repetition_penalty=repetition_penalty,
            api_url=api_url,
            api_key=api_key
        )
        idx = 0
        for i in range(len(correctness_list)):
            if not correctness_list[i]:
                if api_correctness_list[idx] is not None:
                    correctness_list[i] = api_correctness_list[idx]
                idx += 1

    return correctness_list


def compute_score_w_prev_correctness(
    predicts: List[str],
    correctness_list: List[float],
    questions: Optional[List[str]] = None,
    ground_truths: Optional[List[str]] = None,
    format_weight: float = 0.1,
    length_penalty_weight=0.001,
    resp_lengths=None,
    ref_resp_lengths=None,
) -> List[Dict[str, float]]:
    """
    Compute reward scores using pre-computed correctness from the API judge.

    Formula: R_total = (1-w_fmt) * R_acc + w_fmt * R_fmt - w_len * P_len

    Args:
        predicts:              Model predictions.
        correctness_list:      Pre-computed correctness (from API judge).
        questions:             Original questions (unused, kept for interface compatibility).
        ground_truths:         Ground-truth answers (unused, kept for interface compatibility).
        format_weight:         Weight for format reward.
        length_penalty_weight: Weight for length penalty.
        resp_lengths:          Response lengths.
        ref_resp_lengths:      Reference response lengths.

    Returns:
        List of score dictionaries.
    """
    scores = []
    ref_resp_lengths = torch.tensor(ref_resp_lengths)
    if resp_lengths is not None and ref_resp_lengths is not None:
        length_penalty = torch.where(
            torch.logical_and(resp_lengths > ref_resp_lengths, ref_resp_lengths != 0),
            resp_lengths - ref_resp_lengths,
            torch.zeros_like(resp_lengths)
        )
    else:
        length_penalty = torch.zeros(len(predicts))

    for i, (predict, correctness) in enumerate(zip(predicts, correctness_list)):
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)
        format_score = format_reward(predict)

        if isinstance(correctness, str):
            print(f"[WARNING] Sample {i}: correctness is a string '{correctness}', converting to 0.0")
            correctness = 0.0
        elif not isinstance(correctness, (int, float)):
            print(f"[WARNING] Sample {i}: correctness has unexpected type {type(correctness)}, converting to 0.0")
            correctness = 0.0
        else:
            correctness = float(correctness)

        accuracy_score = 1.0 if correctness == 1.0 else 0.0

        repetition_score = compute_repetition_penalty(predict)
        is_repetitive = repetition_score > 0.5
        if is_repetitive:
            accuracy_score = -1.0

        accuracy_reward_part = (1 - format_weight) * accuracy_score
        format_reward_part = format_weight * format_score
        length_penalty_part = length_penalty_weight * length_penalty[i]

        overall_reward = accuracy_reward_part + format_reward_part - length_penalty_part

        scores.append(
            {
                "overall": overall_reward,
                "format": format_score,
                "accuracy": accuracy_score,
                "accuracy_reward_part": accuracy_reward_part,
                "format_reward_part": format_reward_part,
                "length_penalty_part": length_penalty_part,
                "repetition_score": repetition_score,
                "is_repetitive": is_repetitive,
            }
        )

    return scores
