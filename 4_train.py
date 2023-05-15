"""
This module contains a script for training a text generation model using a pre-trained transformer model.

The script takes as input a training dataset, a validation dataset, and several hyperparameters. It trains
the model on the training dataset for a specified number of epochs and evaluates the trained model on the
validation dataset at the end of each epoch. The script saves the best performing model to the specified
output directory.
"""
import os
import time
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup, set_seed
from datasets import Dataset
import mlflow
from mlflow.models.signature import infer_signature

from utils import AverageMeter, T5ForSpellCorrectionAPI

PADDING_STRATEGY = "max_length"
INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
LABELS = "labels"


def evaluate(model, dataloader, device):
    losses = AverageMeter()
    model.eval()
    start_time = time.time()
    for step, batch in enumerate(tqdm(dataloader, desc="Evaluating ...", position=2)):
        batch_size = batch[LABELS].size(0)
        for key in batch.keys():
            batch[key] = batch[key].to(device)

        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss

        losses.update(loss.item(), batch_size)

    duration = time.time() - start_time
    return losses.avg, duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_path", type=str, required=True, help="")
    parser.add_argument("--experiment_name", type=str, required=True, help="")
    parser.add_argument("--run_name", type=str, required=True, help="")
    parser.add_argument("--model_path", type=str, required=True,
                        help="The directory where the best performing model will be saved.")
    parser.add_argument("--train_file", type=str, required=True,
                        help="The path to the training dataset file in CSV format.")
    parser.add_argument("--validation_file", type=str, required=True,
                        help="The path to the validation dataset file in CSV format.")
    parser.add_argument("--test_file", default=None, type=str, help="")
    parser.add_argument("--log_file", default="train.log", type=str, help="")
    parser.add_argument("--input_column", type=str, required=True,
                        help="The name of the column in the dataset files that contains the input text.")
    parser.add_argument("--output_column", type=str, required=True,
                        help="The name of the column in the dataset files that contains the target text.")
    parser.add_argument("--pretrained_model_name_or_path", default="google/flan-t5-small", type=str,
                        help="The name or path to a pre-trained transformer model to use for training.")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="The number of examples to include in each training batch.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The learning rate to use for training.")
    parser.add_argument("--epochs", default=10, type=int, help="The number of training epochs to run.")
    parser.add_argument("--max_length", default=128, type=int,
                        help="The maximum length of the input and target sequences. Sequences longer than this will be truncated.")
    parser.add_argument("--warmup_ratio", default=0.0, type=float,
                        help="The ratio of total number of steps for the warm up part of training.")
    parser.add_argument("--max_norm", default=1.0, type=float,
                        help="The maximum gradient norm to use for clipping gradients.")
    parser.add_argument("--accum_steps", default=1, type=int,
                        help="The number of optimization steps to take before updating the model's parameters.")
    parser.add_argument("--seed", default=2330, type=int, help="The random seed to use for training.")
    parser.add_argument("--preprocessing_num_workers", default=None, type=int,
                        help="The number of worker processes to use for preprocessing the data.")
    args = parser.parse_args()

    # ===== Set seed =====
    set_seed(args.seed)

    # ===== Set tracking URI =====
    EXPERIMENTS_PATH = Path(args.experiments_path)
    EXPERIMENTS_PATH.mkdir(exist_ok=True)  # create experiments dir
    mlflow.set_tracking_uri(EXPERIMENTS_PATH)

    # ===== Set experiment =====
    mlflow.set_experiment(experiment_name=args.experiment_name)

    # ===== Load file =====
    train_dataset = Dataset.from_pandas(pd.read_csv(args.train_file))
    val_dataset = Dataset.from_pandas(pd.read_csv(args.validation_file))
    log_file = open(args.log_file, "w", encoding="utf-8")

    # ===== Preprocessing =====
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer.save_pretrained(args.model_path)

    def encode_batch(examples):
        encoding = tokenizer(
            examples[args.input_column],
            padding=PADDING_STRATEGY,
            truncation=True,
            max_length=args.max_length,
            return_tensors="np",
        )
        target_encoding = tokenizer(
            examples[args.output_column],
            padding=PADDING_STRATEGY,
            truncation=True,
            max_length=args.max_length,
            return_tensors="np",
        )
        labels = target_encoding.input_ids
        labels[labels == tokenizer.pad_token_id] = -100
        encoding[LABELS] = labels

        return encoding

    train_dataset = train_dataset.map(encode_batch, batched=True, num_proc=args.preprocessing_num_workers)
    val_dataset = val_dataset.map(encode_batch, batched=True, num_proc=args.preprocessing_num_workers)
    train_dataset.set_format(type="torch", columns=[INPUT_IDS, ATTENTION_MASK, LABELS])
    val_dataset.set_format(type="torch", columns=[INPUT_IDS, ATTENTION_MASK, LABELS])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # ===== Model =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained(args.pretrained_model_name_or_path).to(device)

    # ===== Optimizer ======
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = num_training_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )

    ###################################
    ##########     Train     ##########
    ###################################

    # ===== Tracking =====
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(vars(args))
        best_val_loss = 100
        epochs = tqdm(range(args.epochs), desc="Epoch ... ", position=0)
        for epoch in epochs:
            model.train()
            train_losses = AverageMeter()
            start_time = time.time()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Training...", position=1)):
                batch_size = batch[LABELS].size(0)
                for key in batch.keys():
                    batch[key] = batch[key].to(device)

                outputs = model(**batch)
                loss = outputs.loss
                train_losses.update(loss.item(), batch_size)

                if args.accum_steps > 1:
                    loss = loss / args.accum_steps
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

                if (step + 1) % args.accum_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                if (step + 1) == 1 or (step + 1) % 2500 == 0 or (step + 1) == len(train_dataloader):
                    epochs.write(
                        f"Epoch: [{epoch + 1}][{step + 1}/{len(train_dataloader)}] "
                        f"Loss: {train_losses.val:.4f}({train_losses.avg:.4f}) "
                        f"Grad: {grad_norm:.4f} "
                        f"LR: {scheduler.get_last_lr()[0]:.8f}",
                        file=log_file,
                    )
                    log_file.flush()
                    os.fsync(log_file.fileno())

                    mlflow.log_metrics(
                        {
                            "learning_rate": scheduler.get_last_lr()[0],
                        },
                        step=(len(train_dataloader) * epoch) + step,
                    )
            train_duration = time.time() - start_time
            epochs.write(f"Training Duration: {train_duration:.3f} sec", file=log_file)

            ####################################
            ##########   Validation   ##########
            ####################################

            val_loss, val_duration = evaluate(model, val_dataloader, device)
            epochs.write(f"Validation Loss: {val_loss:.4f}", file=log_file)
            epochs.write(f"Validation Duration: {val_duration:.3f} sec", file=log_file)

            if val_loss < best_val_loss:
                model.save_pretrained(args.model_path)
                best_val_loss = val_loss

            mlflow.log_metrics(
                {
                    "train_loss": train_losses.avg,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                },
                step=epoch,
            )

        ####################################
        ##########      Test      ##########
        ####################################

        if args.test_file is not None:
            model = model.from_pretrained(args.model_path).to(device)
            test_dataset = Dataset.from_pandas(pd.read_csv(args.test_file))
            test_dataset = test_dataset.map(encode_batch, batched=True)
            test_dataset.set_format(type="torch", columns=[INPUT_IDS, ATTENTION_MASK, LABELS])
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

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
                pad_mask = labels != -100
                diff_mask = element_diff * pad_mask
                correct_bool = diff_mask.sum(axis=1) == 0
                correct.extend(correct_bool)

            test_accuracy = sum(correct) / len(correct)
            test_duration = time.time() - start_time

            epochs.write(f"Test Accuracy: {test_accuracy:.4f}", file=log_file)
            epochs.write(f"Test Duration: {test_duration:.3f} sec", file=log_file)
            
            mlflow.log_metrics({"test_accuracy": test_accuracy})

        log_file.close()
        mlflow.log_artifact(args.log_file)

        # ===== Package model file to mlflow =====
        artifacts = {
            Path(file).stem: os.path.join(args.model_path, file)
            for file in os.listdir(args.model_path)
            if not os.path.basename(file).startswith('.')
        }

        sample = pd.DataFrame({"text": ["nodle in righ uppr lng .", "mass in lft loer lung ."]})
        results = pd.DataFrame({"results": ["nodule in right upper lung .", "mass in left lower lung ."]})
        signature = infer_signature(sample, results)

        mlflow.pyfunc.log_model(
            "model",
            python_model=T5ForSpellCorrectionAPI(max_edit_distance=3),
            code_path=["utils.py"],
            artifacts=artifacts,
            signature=signature,
        )


if __name__ == "__main__":
    main()
