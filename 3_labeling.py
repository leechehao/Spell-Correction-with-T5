"""
This module is a tool for generating mispelling datasets for NLP tasks such as typo
detection and correction. It generates misspellings in a sentence by inserting, removing,
replacing or swapping adjacent characters in the sentence's words. It also adds special
tokens to mark the location of the misspellings. It can be run from the command line
and takes a number of arguments to control the generation process.
"""
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import os
import argparse
import json
import random
from string import ascii_lowercase

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed


def is_valid_word(
    word: str,
    min_num_characters: int = 2,
    stop_words: Optional[List[str]] = None,
) -> bool:
    """
    Check if a word is valid.

    A word is considered valid if it is composed only of alphabetic characters, 
    has at least `min_num_characters` characters (default is 2), and is not in 
    the list of `stop_words` (if provided).

    Args:
        word (str): The word to check.
        min_num_characters (int, optional): The minimum number of characters required
            for the word to be considered valid. Defaults to 2.
        stop_words (Optional[List[str]], optional): A list of stop words to exclude.
            Defaults to None.

    Raises:
        ValueError: If `min_num_characters` is less than 2.

    Returns:
        bool: True if the word is valid, False otherwise.
    """
    if min_num_characters < 2:
        raise ValueError("The minimum number of characters must be greater than or equal to 2")
    if stop_words is None:
        stop_words = []
    return word.encode().isalpha() and len(word) >= min_num_characters and word not in stop_words


def insert_char(word: str) -> str:
    """
    Insert a random lowercase letter into a word.

    This function takes a word as input and randomly selects a position in the word
    at which to insert a lowercase letter from the alphabet. It returns the resulting
    string.

    Args:
        word (str): The word to insert a character into.

    Returns:
        str: The resulting string with the character inserted at a random position.
    """
    chars = list(word)
    sample_index = random.randint(0, len(word) - 1)
    sample_alpha = random.choice(ascii_lowercase)
    chars.insert(sample_index, sample_alpha)
    return "".join(chars)


def remove_char(word: str) -> str:
    """
    Remove a random character from a word.

    This function takes a word as input and randomly selects a character from the word
    to remove. It returns the resulting string.

    Args:
        word (str): The word to remove a character from.

    Returns:
        str: The resulting string with the character removed.
    """
    chars = list(word)
    sample_index = random.randint(0, len(word) - 1)
    chars.pop(sample_index)
    return "".join(chars)


def replace_char(word: str) -> str:
    """
    Replace a random character in a word with a random lowercase letter.

    This function takes a word as input and randomly selects a character from the word
    to replace with a lowercase letter from the alphabet. It returns the resulting string.

    Args:
        word (str): The word to replace a character in.

    Returns:
        str: The resulting string with the character replaced.
    """
    chars = list(word)
    sample_index = random.randint(0, len(word) - 1)
    sample_alpha = random.choice(ascii_lowercase)
    chars[sample_index] = sample_alpha
    return "".join(chars)


def swap_adjacent_char(word: str) -> str:
    """
    Swap two adjacent characters in a word.

    This function takes a word as input and randomly selects a pair of adjacent characters
    from the word to swap. It returns the resulting string. If the word has only one
    character or no characters, a `ValueError` is raised.

    Args:
        word (str): The word to swap characters in.

    Raises:
        ValueError: If the word has only one character or no characters.

    Returns:
        str: The resulting string with the two characters swapped.
    """
    chars = list(word)
    index_range = len(word) - 2
    if index_range < 0:
        raise ValueError("Word length is at least 2")
    sample_index = random.randint(0, index_range)
    tmp_alpha = chars[sample_index]
    chars[sample_index] = chars[sample_index + 1]
    chars[sample_index + 1] = tmp_alpha
    return "".join(chars)


STRATEGY_FN_CHAR: Dict[int, Callable] = {
    0: insert_char,
    1: remove_char,
    2: replace_char,
    3: swap_adjacent_char,
}


def generate_typo(word: str) -> str:
    """
    Generate a typo for a word.

    This function takes a word as input and randomly applies one of the following
    transformations to it:
    - Insert a random lowercase letter at a random position in the word
    - Remove a random character from the word
    - Replace a random character in the word with a random lowercase letter
    - Swap two adjacent characters in the word

    If the resulting word is the same as the original word, the function will try
    again up to a maximum of 10 times. If the maximum number of attempts is reached,
    a `ValueError` is raised.

    Args:
        word (str): The word to generate a typo for.

    Raises:
        ValueError: If the maximum number of attempts is reached.

    Returns:
        str: The resulting word with a randomly generated typo.
    """
    max_iter = 10
    while True:
        max_iter -= 1
        typo = STRATEGY_FN_CHAR[random.choice(tuple(STRATEGY_FN_CHAR))](word)
        if typo != word:
            return typo
        if max_iter == 0:
            raise ValueError(f"Typos are regenerated too many times (maximum {max_iter} times)")


def add_special_token(
    detect_output: List[str],
    correct_output: List[str],
    indexes: List[int],
    head_meak: str,
    tail_mark: str,
) -> None:
    """
    Add special tokens to a list of words.

    This function takes as input two lists of words, `detect_output` and `correct_output`,
    as well as a list of indexes indicating the positions where the special tokens
    should be added. For each index, the function inserts the head mark and tail mark
    at the corresponding position in both lists of words.

    Args:
        detect_output (List[str]): The list of words to add the special tokens to.
        correct_output (List[str]): The list of correct words to add the special tokens to.
        indexes (List[int]): The list of indexes indicating where to insert the special tokens.
        head_meak (str): The head mark to insert.
        tail_mark (str): The tail mark to insert.
    """
    for index in sorted(indexes, reverse=True):
        detect_output.insert(index + 1, tail_mark)
        detect_output.insert(index, head_meak)
        correct_output.insert(index + 1, tail_mark)
        correct_output.insert(index, head_meak)


def prepare_misspelling_label(
    sentence: str,
    typo_probability: float,
    max_num_typo: int,
    head_mark: str,
    tail_mark: str,
    data_multiple: int = 1,
    min_num_characters: Optional[int] = None,
    num_typo_weights: Sequence[int] = None,
    stop_words: Optional[List[str]] = None,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """
    Generates a set of labeled data for misspelling detection and correction.

    Args:
        sentence (str): The input sentence.
        typo_probability (float): The probability of adding typos to the sentence.
        max_num_typo (int): The maximum number of typos to add to the sentence.
        head_mark (str): The special token to be added at the start of the misspelled words.
        tail_mark (str): The special token to be added at the end of the misspelled words.
        data_multiple (int, optional): The multiple of the original data size that will be generated. Defaults to 1.
        min_num_characters (Optional[int], optional): The minimum number of characters required for a word to be considered
            as will generate a misspelling. Defaults to None.
        num_typo_weights (Sequence[int], optional): The weights for sampling the number of typos to add. Defaults to None.
        stop_words (Optional[List[str]], optional): A list of stop words to be excluded from misspelling. Defaults to None.

    Returns:
        Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]: The generated sets of
            labeled data for misspelling detection and correction.

        The first element of the tuple is a list of tuples for misspelling detection, where each tuple contains:
            - The original sentence.
            - The sentence with typos.
            - The sentence with special tokens added to the misspelled words.

        The second element of the tuple is a list of tuples for misspelling correction, where each tuple contains:
            - The original sentence.
            - The sentence with special tokens added to the misspelled words.
            - The corrected sentence.
    """
    detect_label_data: List[Tuple[str, str, str]] = []
    correct_label_data: List[Tuple[str, str, str]] = []

    if len(sentence) < 5:
        return detect_label_data, correct_label_data

    words = sentence.split(' ')
    is_typo = random.random() < typo_probability
    valid_mask = np.array([is_valid_word(word, min_num_characters, stop_words) for word in words])
    if not is_typo or not any(valid_mask):
        detect_label_data.append((sentence,) * 3)
        return detect_label_data, correct_label_data

    n_valid_words = sum(valid_mask)
    while True:
        num_typo = random.choices([n + 1 for n in range(max_num_typo)], weights=num_typo_weights)[0]
        if num_typo <= n_valid_words:
            break

    valid_indexes = np.arange(len(words))[valid_mask]
    sample_indexes: List[int] = random.sample(valid_indexes.tolist(), num_typo)
    for _ in range(data_multiple):
        detect_input = list(words)
        for sample_index in sample_indexes:
            detect_input[sample_index] = generate_typo(detect_input[sample_index])

        detect_output = list(detect_input)
        correct_output = list(words)
        add_special_token(detect_output, correct_output, sample_indexes, head_mark, tail_mark)
        detect_label_data.append((" ".join(words), " ".join(detect_input), " ".join(detect_output)))
        correct_label_data.append((" ".join(words), " ".join(detect_output), " ".join(correct_output)))

    return detect_label_data, correct_label_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="The path to the input file in LineSentence format. (one line = one sentence.)")
    parser.add_argument("--output_file", type=str, required=True, help="The path to the output CSV file. This file will contain the sentences with typos added.")
    parser.add_argument("--prompt_config_dir", default=None, type=str, help="The directory where the prompt configuration file will be written.")
    parser.add_argument("--stop_words_file", default=None, type=str, help="The path to a file containing stop words. These words will not be modified by the script.")
    parser.add_argument("--typo_probability", default=0.5, type=float, help="The probability of adding typos to the sentence.")
    parser.add_argument("--max_num_typo", default=3, type=int, help="The maximum number of typos to add to the sentence.")
    parser.add_argument("--head_mark", default="$", type=str, help="The special token that will be used to mark the beginning of a typo in the output data.")
    parser.add_argument("--tail_mark", default="@", type=str, help="The special token that will be used to mark the end of a typo in the output data.")
    parser.add_argument("--data_multiple", default=4, type=int, help="The multiple of the original data size that will be generated.")
    parser.add_argument("--min_num_characters", default=4, type=int, help="The minimum number of characters required for a word to be considered as will generate a misspelling.")
    parser.add_argument("--num_typo_weights", default=(5, 4, 1), type=int, nargs="+", help="The weights for sampling the number of typos to add.")
    parser.add_argument("--detect_prompt", default="detect typo:", type=str, help="The prompt that will be used for the typo detection task.")
    parser.add_argument("--correct_prompt", default="correct typo:", type=str, help="The prompt that will be used for the typo correction task.")
    parser.add_argument("--max_length", default=120, type=int, help="The maximum length of the generated sentences.")
    parser.add_argument("--pretrained_model_name_or_path", default="google/flan-t5-small", type=str, help="The path to the directory containing the pre-trained model.")
    parser.add_argument("--seed", default=2330, type=int, help="The random seed to use.")
    args = parser.parse_args()

    SENTENCE = "sentence"
    INPUT = "input"
    OUTPUT = "output"
    TASK = "task"
    NUM_TOKENS = "num_tokens"
    PROMPT_CONFIG_FILE = "prompt_config.json"
    HEAD_MARK = "head_mark"
    TAIL_MARK = "tail_mark"
    DETECT_PROMPT = "detect_prompt"
    CORRECT_PROMPT = "correct_prompt"

    # ===== 設定隨機種子 =====
    set_seed(args.seed)

    # ===== 寫入prompt設定 =====
    if args.prompt_config_dir is not None:
        os.makedirs(args.prompt_config_dir, exist_ok=True)
        prompt_config_file = os.path.join(args.prompt_config_dir, PROMPT_CONFIG_FILE)
        write_dict = {
            HEAD_MARK: args.head_mark,
            TAIL_MARK: args.tail_mark,
            DETECT_PROMPT: args.detect_prompt,
            CORRECT_PROMPT: args.correct_prompt,
        }
        with open(prompt_config_file, "w", encoding="utf-8") as f:
            out_str = json.dumps(write_dict, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            f.write(out_str)

    # ===== 載入停用詞 =====
    stop_words = [
        word.strip() for word in open(args.stop_words_file, encoding="utf-8").readlines()
    ] if args.stop_words_file is not None else None

    # ===== 產生訓練資料 =====
    args.input_file
    detect_label_data = []
    correct_label_data = []
    with open(args.input_file, "r", encoding="utf-8") as finput:
        for sentence in tqdm(finput.readlines(), desc="Generating label data..."):
            outputs = prepare_misspelling_label(
                sentence.strip(),
                typo_probability=args.typo_probability,
                max_num_typo=args.max_num_typo,
                head_mark=args.head_mark,
                tail_mark=args.tail_mark,
                data_multiple=args.data_multiple,
                min_num_characters=args.min_num_characters,
                num_typo_weights=args.num_typo_weights,
                stop_words=stop_words,
            )
            detect_label_data.extend(outputs[0])
            correct_label_data.extend(outputs[1])

    df_detect_data = pd.DataFrame(detect_label_data, columns=[SENTENCE, INPUT, OUTPUT])
    df_correct_data = pd.DataFrame(correct_label_data, columns=[SENTENCE, INPUT, OUTPUT])

    # ===== 加入任務前綴 =====
    df_detect_data[INPUT] = df_detect_data[INPUT].apply(lambda x: " ".join((args.detect_prompt, x)))
    df_detect_data[TASK] = 0
    df_correct_data[INPUT] = df_correct_data[INPUT].apply(lambda x: " ".join((args.correct_prompt, x)))
    df_correct_data[TASK] = 1

    df_dataset = pd.concat([df_detect_data, df_correct_data])
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    tqdm.pandas(desc="Compute number of tokens...")
    df_dataset[NUM_TOKENS] = df_dataset[INPUT].progress_apply(lambda x: len(tokenizer(x).input_ids))
    df_dataset[df_dataset[NUM_TOKENS] <= args.max_length].to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
