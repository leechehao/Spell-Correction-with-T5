"""
A pipeline for correcting misspellings in a sentence using a T5 model.

This module provides a `T5ForMisspellingPipeline` class that can be used to correct misspellings in a given sentence.
The pipeline uses a pretrained T5 model and a tokenizer to identify and correct misspelled words in the sentence.
The maximum edit distance between the original and corrected words is specified when initializing the pipeline,
and only corrected words with a minimum edit distance less than or equal to the specified value will be accepted.

Example:
    To use the `T5ForMisspellingPipeline` class, create an instance of the class and call it with a sentence as an argument.
    The returned value will be the corrected sentence.

    pipeline = T5ForMisspellingPipeline(
        pretrained_model_name_or_path="path/to/pretrained/model",
        max_edit_distance=3,
    )
    corrected_sentence = pipeline("This sntence has misspellings.")
"""
from typing import Iterator, List, Optional, Set, Tuple

import os
import re
import json

import torch
import transformers
from transformers import AutoTokenizer, T5ForConditionalGeneration

from Wingene.factory.typesetting_cleanser_factory import get_TypesettingCleanserForNHIBasisReport


PROMPT_CONFIG_FILE = "prompt_config.json"
HEAD_MARK = "head_mark"
TAIL_MARK = "tail_mark"
DETECT_PROMPT = "detect_prompt"
CORRECT_PROMPT = "correct_prompt"


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


class T5ForSpellCorrectionPipeline:
    """A pipeline for correcting misspellings in a sentence using a T5 model."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_edit_distance: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the pipeline with a pre-trained T5 model, the maximum edit distance to consider, and a
        device to use for computation.

        Args:
            pretrained_model_name_or_path (str): The path or name of the pre-trained T5 model to use.
            max_edit_distance (Optional[int], optional): The maximum edit distance to consider when correcting
                misspellings. Defaults to None.
            device (Optional[torch.device], optional): The device on which the T5 model will be loaded. If
                not provided, the model will be loaded on the best available device (CUDA, if available,
                otherwise CPU). Defaults to None.

        Raises:
            ValueError: If the `pretrained_model_name_or_path` does not contain a prompt configuration file.
        """
        prompt_file = os.path.join(pretrained_model_name_or_path, PROMPT_CONFIG_FILE)
        if not os.path.exists(prompt_file):
            raise ValueError(
                f"Can't find a prompt config file at path '{pretrained_model_name_or_path}"
            )
        self.prompt_config = json.load(open(prompt_file))
        self.max_edit_distance = max_edit_distance
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path).to(self.device)
        self.head_mark_id: int = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(self.prompt_config[HEAD_MARK]),
        )[0]
        self.tail_mark_id: int = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(self.prompt_config[TAIL_MARK]),
        )[0]
        self.mark_id_set = frozenset((self.head_mark_id, self.tail_mark_id))
        self.n_detect_prompt_words, self.detect_prompt_last_token_id = self._get_detect_prompt_info()
        self.cleanser = get_TypesettingCleanserForNHIBasisReport()

    def __call__(self, sentence: str) -> str:
        """
        Correct the misspellings in the given sentence using a pre-trained T5 model.

        This method compares the minimum edit distance between the misspelled words and
        their corrected versions to the `max_edit_distance` parameter specified when
        initializing the pipeline. If the minimum edit distance is less than or equal to
        the `max_edit_distance`, the method replaces the misspelled word with the corrected
        word. If the `max_edit_distance` is not provided, then all corrected words will be
        accepted.

        Args:
            sentence (str): The sentence to be corrected.

        Returns:
            str: The corrected sentence.
        """
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
        cleaned_sentence = self.cleanser.cleanse(sentence)
        detect_sentence = " ".join((self.prompt_config[DETECT_PROMPT], cleaned_sentence))
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


if __name__ == "__main__":
    pipeline = T5ForSpellCorrectionPipeline(
        "models/best_model",
        max_edit_distance=3,
    )
    print(pipeline("opciy and noodl at lung ."))
    print(pipeline("2. Mid lug emphysma ."))
    print(pipeline("# Atrophy of lebt kidney ."))
    print(pipeline("No pericaruial effusiwon ."))
