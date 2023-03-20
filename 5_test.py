"""
Uses a pretrained text generation model to evaluate a test dataset. It consists of the following steps:

1. Parse command line arguments to specify the test data file, results file, pretrained model, and other
optional settings.
2. Load the test data from the specified file and create a Dataset object to iterate over the data in
batches.
3. Initialize a Tokenizer object with the pretrained model to tokenize the input and output text.
4. Define a encode_batch() function that takes a batch of examples and returns the encoded versions of
the input and output text.
5. Apply the encode_batch() function to each batch of examples in the test_dataset and set the format of
the test_dataset to "torch".
6. Create a test_dataloader object to iterate over the test_dataset in batches.
7. Iterate over the batches in the test_dataloader and use the model to generate output text for each
batch.
8. Check whether the generated text is correct for each batch and compute the overall accuracy of the
model on the test data.
9. Save the results to the specified file and print the overall accuracy.
"""
import argparse
import time

import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration
import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True, help="The path to a file containing the test data.")
    parser.add_argument("--results_file", type=str, required=True,
                        help="The path to a file where the results of the script should be saved.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="The name or path of a pretrained model that the script should use.")
    parser.add_argument("--max_length", default=128, type=int, help="The maximum length of the input sequences.")
    parser.add_argument("--eval_batch_size", default=100, type=int, help="The batch size to use during evaluation.")
    args = parser.parse_args()

    INPUT = "input"
    OUTPUT = "output"
    INPUT_IDS = "input_ids"
    ATTENTION_MASK = "attention_mask"
    LABELS = "labels"
    CORRECT = "correct"

    # ===== 載入模型 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained(args.pretrained_model_name_or_path).to(device)
    model.eval()

    # ===== 讀取資料 =====
    df_test = pd.read_csv(args.test_file)
    test_dataset = datasets.Dataset.from_pandas(df_test)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    def encode_batch(examples):
        encoding = tokenizer(
            examples[INPUT],
            padding=True,
            truncation=True,
        )
        target_encoding = tokenizer(
            examples[OUTPUT],
            padding=True,
            truncation=True,
        )
        encoding[LABELS] = target_encoding.input_ids

        return encoding

    test_dataset = test_dataset.map(encode_batch, batched=True, batch_size=args.eval_batch_size)
    test_dataset.set_format(type="torch", columns=[INPUT_IDS, ATTENTION_MASK, LABELS])
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    correct = []
    start_time = time.time()
    for batch in tqdm(test_dataloader, desc="Evaluating...", position=0):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model.generate(
            input_ids=batch[INPUT_IDS],
            attention_mask=batch[ATTENTION_MASK],
            max_length=args.max_length,
        )

        outputs = outputs.cpu().numpy()
        labels = batch[LABELS].cpu().numpy()
        new_outputs = np.zeros(labels.shape)
        size_ = min(outputs[:, 1:].shape[1], labels.shape[1])
        new_outputs[:, :size_] = outputs[:, 1:size_ + 1]
        element_diff = new_outputs != labels
        pad_mask = labels != 0
        diff_mask = element_diff * pad_mask
        correct_bool = diff_mask.sum(axis=1) == 0
        correct.extend(correct_bool)

    df_test[CORRECT] = correct
    df_test.to_csv(args.results_file, index=False)
    test_duration = time.time() - start_time
    print(f"Test Accuracy: {sum(df_test[CORRECT]) / len(df_test):.4f}")
    print(f"Test Duration: {test_duration:.3f} sec")


if __name__ == "__main__":
    main()
