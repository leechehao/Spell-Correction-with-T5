# Spell Correction with T5

## Typo Generator
A tool for generating misspelling datasets for NLP tasks such as typo detection and correction. It generates misspellings in a sentence by inserting, removing, replacing or swapping adjacent characters in the sentence's words. It also adds special tokens to mark the location of the misspellings.

### Usage
To use the typo generator, refer to the following usage information:
```
usage: 3_typo_generator.py [-h] --input_file INPUT_FILE \
                                --output_file OUTPUT_FILE \
                                --input_column INPUT_COLUMN \
                                [--prompt_config_dir PROMPT_CONFIG_DIR] \
                                [--stop_words_file STOP_WORDS_FILE] \
                                [--typo_probability TYPO_PROBABILITY] \
                                [--max_num_typo MAX_NUM_TYPO] \
                                [--head_mark HEAD_MARK] \
                                [--tail_mark TAIL_MARK] \
                                [--data_multiple DATA_MULTIPLE] \
                                [--min_num_characters MIN_NUM_CHARACTERS] \
                                [--num_typo_weights NUM_TYPO_WEIGHTS] \
                                [--detect_prompt DETECT_PROMPT] \
                                [--correct_prompt CORRECT_PROMPT] \
                                [--max_length MAX_LENGTH] \
                                [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH] \
                                [--seed SEED]

required arguments:
  --input_file INPUT_FILE                 The path to the input CSV file. This file should contain a column
                                          with the sentences to which typos will be added.
  --output_file OUTPUT_FILE               The path to the output CSV file. This file will contain the
                                          sentences with typos added.
  --input_column INPUT_COLUMN             The name of the column in the input CSV file containing the
                                          sentences.

optional arguments:
  -h, --help                              show this help message and exit
  --prompt_config_dir PROMPT_CONFIG_DIR   The directory where the prompt configuration file will be written.
  --stop_words_file STOP_WORDS_FILE       The path to a file containing stop words. These words will not be
                                          modified by the script.
  --typo_probability TYPO_PROBABILITY     The probability of adding typos to the sentence.
  --max_num_typo MAX_NUM_TYPO             The maximum number of typos to add to the sentence.
  --head_mark HEAD_MARK                   The special token that will be used to mark the beginning of a
                                          typo in the output data.
  --tail_mark TAIL_MARK                   The special token that will be used to mark the end of a typo in
                                          the output data.
  --data_multiple DATA_MULTIPLE           The multiple of the original data size that will be generated.
  --min_num_characters MIN_NUM_CHARACTERS The minimum number of characters required for a word to be considered
                                          as will generate a misspelling.
  --num_typo_weights NUM_TYPO_WEIGHTS     The weights for sampling the number of typos to add.
  --detect_prompt DETECT_PROMPT           The prompt that will be used for the typo detection task.
  --correct_prompt CORRECT_PROMPT         The prompt that will be used for the typo correction task.
  --max_length MAX_LENGTH                 The maximum length of the generated sentences.
  --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                                          The path to the directory containing the pre-trained model.
  --seed SEED                             The random seed to use.
```

### Arguments
+ **input_file** *(str)* ─ The path to the input CSV file. This file should contain a column with the sentences to which typos will be added.
+ **output_file** *(str)* ─ The path to the output CSV file. This file will contain the sentences with typos added.
+ **input_column** *(str)* ─ The name of the column in the input CSV file containing the sentences.
+ **prompt_config_dir** *(str, optional, defaults to **`None`**)* ─ The directory where the prompt configuration file will be written.
+ **stop_words_file** *(str, optional, defaults to **`None`**)* ─ The path to a file containing stop words. These words will not be modified by the script.
+ **typo_probability** *(float, optional, defaults to **`0.15`**)* ─ The probability of adding typos to the sentence.
+ **max_num_typo** *(int, optional, defaults to **`3`**)* ─ The maximum number of typos to add to the sentence.
+ **head_mark** *(str, optional, defaults to **`$`**)* ─ The special token that will be used to mark the beginning of a typo in the output data.
+ **tail_mark** *(str, optional, defaults to **`@`**)* ─ The special token that will be used to mark the end of a typo in the output data.
+ **data_multiple** *(int, optional, defaults to **`4`**)* ─ The multiple of the original data size that will be generated.
+ **min_num_characters** *(int, optional, defaults to **`4`**)* ─ The minimum number of characters required for a word to be considered as will generate a misspelling.
+ **num_typo_weights** *(Sequence[int], optional, defaults to **`(5, 4, 1)`**)* ─ The weights for sampling the number of typos to add.
+ **detect_prompt** *(str, optional, defaults to **`detect typo:`**)* ─ The prompt that will be used for the typo detection task.
+ **correct_prompt** *(str, optional, defaults to **`correct typo:`**)* ─ The prompt that will be used for the typo correction task.
+ **max_length** *(int, optional, defaults to **`120`**)* ─ The maximum length of the generated sentences.
+ **pretrained_model_name_or_path** *(str, optional, defaults to **`google/flan-t5-small`**)* ─ The path to the directory containing the pre-trained model.
+ **seed** *(int, optional, defaults to **`2330`**)* ─ The random seed to use.

## Train Model
Train a text generation model using a pre-trained transformer model.

### Usage
To train the model, refer to the following usage information:
```
usage: 4_train.py [-h] --output_dir OUTPUT_DIR \
                       --train_file TRAIN_FILE \
                       --valid_file VALID_FILE \
                       --input_column INPUT_COLUMN \
                       --output_column OUTPUT_COLUMN \
                       [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH] \
                       [--max_length MAX_LENGTH] \
                       [--batch_size BATCH_SIZE] \
                       [--learning_rate LEARNING_RATE] \
                       [--epochs EPOCHS] \
                       [--warmup_ratio WARMUP_RATIO] \
                       [--accum_steps ACCUM_STEPS] \
                       [--max_norm MAX_NORM] \
                       [--seed SEED] \
                       [--preprocessing_num_workers PREPROCESSING_NUM_WORKERS]

required arguments:
  --output_dir OUTPUT_DIR         The directory where the best performing model will be saved.
  --train_file TRAIN_FILE         The path to the training dataset file in CSV format.
  --valid_file VALID_FILE         The path to the validation dataset file in CSV format.
  --input_column INPUT_COLUMN     The name of the column in the dataset files that contains the input text.
  --output_column OUTPUT_COLUMN   The name of the column in the dataset files that contains the target text.

optional arguments:
  -h, --help                      show this help message and exit
  --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                                  The name or path to a pre-trained transformer model to use for training.
  --max_length MAX_LENGTH         The maximum length of the input and target sequences. Sequences longer
                                  than this will be truncated.
  --batch_size BATCH_SIZE         The number of examples to include in each training batch.
  --learning_rate LEARNING_RATE   The learning rate to use for training.
  --epochs EPOCHS                 The number of training epochs to run.
  --warmup_ratio WARMUP_RATIO     The ratio of warm-up steps to take before adjusting the learning rate.
  --accum_steps ACCUM_STEPS       The number of optimization steps to take before updating the model's
                                  parameters.
  --max_norm MAX_NORM             The maximum gradient norm to use for clipping gradients.
  --seed SEED                     The random seed to use for training.
  --preprocessing_num_workers PREPROCESSING_NUM_WORKERS
                                  The number of worker processes to use for preprocessing the data.
```

### Arguments
+ **output_dir** *(str)* ─ The directory where the best performing model will be saved.
+ **train_file** *(str)* ─ The path to the training dataset file in CSV format.
+ **valid_file** *(str)* ─ The path to the validation dataset file in CSV format.
+ **input_column** *(str)* ─ The name of the column in the dataset files that contains the input text.
+ **output_column** *(str)* ─ The name of the column in the dataset files that contains the target text.
+ **pretrained_model_name_or_path** *(str, optional, defaults to **`google/flan-t5-small`**)* ─ The name or path to a pre-trained transformer model to use for training.
+ **max_length** *(int, optional, defaults to **`128`**)* ─ The maximum length of the input and target sequences. Sequences longer than this will be truncated.
+ **batch_size** *(int, optional, defaults to **`16`**)* ─ The number of examples to include in each training batch.
+ **learning_rate** *(float, optional, defaults to **`1e-4`**)* ─ The learning rate to use for training.
+ **epochs** *(int, optional, defaults to **`10`**)* ─ The number of training epochs to run.
+ **warmup_ratio** *(float, optional, defaults to **`0.0`**)* ─ The ratio of total number of steps for the warm up part of training.
+ **accum_steps** *(int, optional, defaults to **`1`**)* ─ The number of optimization steps to take before updating the model's parameters.
+ **max_norm** *(float, optional, defaults to **`1.0`**)* ─ The maximum gradient norm to use for clipping gradients.
+ **seed** *(int, optional, defaults to **`2330`**)* ─ The random seed to use for training.
+ **preprocessing_num_workers** *(int, optional, defaults to **`None`**)* ─ The number of worker processes to use for preprocessing the data.

## Evaluate Model
Evaluate a pretrained text generation model on a test dataset.

### Usage
To evaluate the model, refer to the following usage information:
```
usage: 5_test.py [-h] --test_file TEST_FILE \
                      --results_file RESULTS_FILE \
                      --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH \
                      [--max_length MAX_LENGTH] \
                      [--eval_batch_size EVAL_BATCH_SIZE]

required arguments:
  --test_file TEST_FILE               The path to a file containing the test data.
  --results_file RESULTS_FILE         The path to a file where the results of the script should be saved.
  --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                                      The name or path of a pretrained model that the script should use.

optional arguments:
  -h, --help                          show this help message and exit
  --max_length MAX_LENGTH             The maximum length of the input sequences.
  --eval_batch_size EVAL_BATCH_SIZE   The batch size to use during evaluation.
```

### Arguments
+ **test_file** *(str)* ─ The path to a file containing the test data.
+ **results_file** *(str)* ─ The path to a file where the results of the script should be saved.
+ **pretrained_model_name_or_path** *(str)* ─ The name or path of a pretrained model that the script should use.
+ **max_length** *(int, optional, defaults to **`128`**)* ─ The maximum length of the input sequences.
+ **eval_batch_size** *(int, optional, defaults to **`100`**)* ─ The batch size to use during evaluation.

## Inference Pipeline
A pipeline for correcting misspellings in a sentence using a T5 model.

### Usage
To use the **`T5ForMisspellingPipeline`** class, create an instance of the class and call it with a sentence as an argument.
The returned value will be the corrected sentence.

The maximum edit distance between the original and corrected words is specified when initializing the pipeline,
and only corrected words with a minimum edit distance less than or equal to the specified value will be accepted.
```python
pipeline = T5ForMisspellingPipeline(
    pretrained_model_name_or_path="path/to/pretrained/model",
    max_edit_distance=3,
)
corrected_sentence = pipeline("This sntence has misspellings.")
```

