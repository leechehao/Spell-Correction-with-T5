import os
from typing import Iterator, List, Optional, Set, Tuple
import json
import re

import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, T5ForConditionalGeneration
from mlflow.pyfunc import PythonModel, PythonModelContext

PROMPT_CONFIG = "prompt_config"
HEAD_MARK = "head_mark"
TAIL_MARK = "tail_mark"
DETECT_PROMPT = "detect_prompt"
CORRECT_PROMPT = "correct_prompt"


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def min_edit_distance(source: str, target: str) -> int:
    """
    Calculate the minimum edit distance between two strings.

    The minimum edit distance is the minimum number of insertions, deletions,
    and substitutions required to transform the source string into the target string.

    Args:
        source (str): The source string.
        target (str): The target string.

    Returns:
        int: The minimum edit distance between the source and target strings.
    """
    n = len(source)
    m = len(target)
    matrix = [[i + j for j in range(m + 1)] for i in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if source[i-1] == target[j-1] else 2
            matrix[i][j] = min(
                matrix[i-1][j] + 1,
                matrix[i][j-1] + 1,
                matrix[i-1][j-1] + sub_cost,
            )

    return matrix[n][m]


class T5ForSpellCorrectionAPI(PythonModel):
    def __init__(self, max_edit_distance: Optional[int] = None) -> None:
        super().__init__()
        self.max_edit_distance = max_edit_distance

    def load_context(self, context: PythonModelContext):
        self.prompt_config = json.load(open(os.path.join(context.artifacts[PROMPT_CONFIG])))
        model_path = os.path.dirname(context.artifacts["config"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        self.head_mark_id: int = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(self.prompt_config[HEAD_MARK]),
        )[0]
        self.tail_mark_id: int = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(self.prompt_config[TAIL_MARK]),
        )[0]
        self.mark_id_set = frozenset((self.head_mark_id, self.tail_mark_id))
        self.n_detect_prompt_words, self.detect_prompt_last_token_id = self._get_detect_prompt_info()

    def predict(self, context, df):
        results = df.text.apply(self._predict)
        return pd.DataFrame({"results": results})

    def _predict(self, sentence: str) -> str:
        torch.cuda.empty_cache()
        detect_words = self._prepare_detect_input(sentence)
        detect_tokenized = self.tokenizer(
            detect_words,
            is_split_into_words=True,
            return_tensors="pt",
        )
        detect_generate_ids = self.model.generate(
            detect_tokenized.input_ids.to(self.device),
            max_new_tokens=self.tokenizer.model_max_length,
        )
        typo_word_idx = self._get_typo_word_idx(detect_tokenized, detect_generate_ids)
        correct_tokenized = self.tokenizer(
            self._prepare_correct_input(detect_generate_ids),
            return_tensors="pt",
        )
        correct_generate_ids = self.model.generate(
            correct_tokenized.input_ids.to(self.device),
            max_new_tokens=self.tokenizer.model_max_length,
        )
        correct_words = re.findall(
            rf'\{self.prompt_config[HEAD_MARK]}\s(.+?)\s\{self.prompt_config[TAIL_MARK]}',
            self.tokenizer.decode(correct_generate_ids[0], skip_special_tokens=True),
        )
        for idx, correct_word in zip(typo_word_idx, correct_words):
            if (
                self.max_edit_distance is not None and
                min_edit_distance(detect_words[idx], correct_word) > self.max_edit_distance
            ):
                continue
            detect_words[idx] = correct_word

        return " ".join(detect_words[self.n_detect_prompt_words:])

    def _get_detect_prompt_info(self) -> Tuple[int, int]:
        n_detect_prompt_words = len(self.prompt_config[DETECT_PROMPT].split(" "))
        detect_prompt_last_token_id = self.tokenizer.encode(
            self.prompt_config[DETECT_PROMPT],
            add_special_tokens=False,
        )[-1]
        return n_detect_prompt_words, detect_prompt_last_token_id

    def _prepare_detect_input(self, sentence: str) -> List[str]:
        detect_sentence = " ".join((self.prompt_config[DETECT_PROMPT], sentence))
        detect_words = detect_sentence.split(" ")
        return detect_words

    def _get_typo_word_idx(
        self,
        detect_tokenized: transformers.BatchEncoding,
        detect_generate_ids: torch.Tensor,
    ) -> Iterator[int]:
        head_mark_position: Set[Tuple[int, int]] = set()

        for i in range(len(detect_generate_ids[0])):
            if detect_generate_ids[0][i] == self.head_mark_id:
                point = i
                while point >= 0:
                    left = (
                        detect_generate_ids[0][point - 1].item()
                        if detect_generate_ids[0][point - 1] != self.tokenizer.pad_token_id
                        else self.detect_prompt_last_token_id
                    )
                    if left not in self.mark_id_set:
                        break
                    point -= 1
                right = detect_generate_ids[0][i + 1].item()
                head_mark_position.add((left, right))

        return (
            word_idx
            for left, right, word_idx in zip(
                detect_tokenized.input_ids[0][:-1],
                detect_tokenized.input_ids[0][1:],
                detect_tokenized.word_ids()[1:],
            )
            if (left.item(), right.item()) in head_mark_position
        )

    def _prepare_correct_input(self, detect_generate_ids: torch.Tensor) -> str:
        correct_input = " ".join(
            (
                self.prompt_config[CORRECT_PROMPT],
                self.tokenizer.decode(detect_generate_ids[0], skip_special_tokens=True),
            )
        )
        return correct_input
