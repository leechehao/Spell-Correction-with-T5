"""
This module performs NLP operations on a given input file with a specified report column.

The input file should be in CSV format with the specified report column containing text data
to be processed. The processed results are then outputted to a specified output file.

The NLP operations performed on the report column include:
- Dropping missing values data
- Cleaning the text
- Extracting findings from the cleaned text
- Segmenting sentences from the extracted findings
- Removing extra spaces and dropping duplicate contents
"""

from typing import Sequence
import argparse
import re

import pandas as pd
from tqdm import tqdm

from Wingene.factory.typesetting_cleanser_factory import get_TypesettingCleanserForNHIBasisReport


FINDINGS = "FINDINGS"
SENTENCE = "SENTENCE"
CONTENT = "CONTENT"
cleanser = get_TypesettingCleanserForNHIBasisReport()


def clean_text(text: str) -> str:
    """
    Cleans and normalizes the input text.

    This function removes unwanted characters and performs text normalization on the input text.

    Args:
        text (str): The input text to be cleaned and normalized.

    Returns:
        str: The cleaned and normalized text.
    """
    text = cleanser.cleanse(re.sub(r"\n", " ", text))
    return text


def extract_findings(text: str) -> str:
    """
    Extracts the findings section from the input text.

    This function searches for the substring `finding` (case-insensitive) in the input text and returns
    the portion of the text after the first instance of `finding`.

    Args:
        text (str): The input text to extract the findings from.

    Returns:
        str: The extracted findings or the original text if the substring `finding` is not found.
    """
    results = re.search(r"finding\w* :", text, flags=re.IGNORECASE)
    return text if results is None else text[results.span()[0]:]


def segment_sentence(text: str) -> Sequence[str]:
    """
    Segments the input text into sentences.

    This function splits the input text into sentences based on the occurrence of `.` and returns a list
    of the individual sentences with each sentence ending with a period (`.`).

    Args:
        text (str): The input text to be segmented into sentences.

    Returns:
        Sequence[str]: A list of the individual sentences in the input text.
    """
    to_return = []
    sentences = text.split(" .")
    for sentence in sentences:
        if len(sentence) == 0:
            continue
        to_return.append(" ".join((sentence.strip(), ".")))
    return to_return


def format_data_to_sentence(series: pd.Series, input_column: str) -> pd.DataFrame:
    """
    Formats a given series into a dataframe of sentences.

    This function segments a given series into sentences, adds a new column named `SENTENCE` to the dataframe,
    and populates the new column with the individual sentences. The rest of the series data is copied over to
    the new dataframe.

    Args:
        series (pd.Series): The series to be formatted into a dataframe.
        input_column (str): The name of the column in the series that contains the text to be segmented into sentences.

    Returns:
        pd.DataFrame: A dataframe with the `SENTENCE` column added, containing the individual sentences.
    """
    df = pd.DataFrame([], columns=series.index.tolist() + [SENTENCE])
    df[SENTENCE] = segment_sentence(series[input_column])
    for key, val in series.items():
        df[key] = val
    return df


def remove_space(text: str) -> str:
    """
    Remove whitespaces from the text.

    Args:
        text (str): The text from which to remove whitespaces.

    Returns:
        str: The input text with all whitespaces removed.
    """
    return re.sub(r"\s+", "", text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="The input file path, a CSV file.")
    parser.add_argument("--report_column", type=str, required=True, help="The column name of the input file to be processed.")
    parser.add_argument("--output_file", type=str, required=True, help="The output file path, a CSV file.")
    args = parser.parse_args()

    # ===== Load file =====
    df = pd.read_csv(args.input_file)

    # ===== Drop missing values data =====
    df = df[~df.isnull().any(axis=1)]

    # ===== Clean report =====
    tqdm.pandas(desc="Clean report...")
    df[args.report_column] = df[args.report_column].progress_apply(clean_text)

    # ===== Extract findings =====
    tqdm.pandas(desc="Extract findings from report...")
    df[FINDINGS] = df[args.report_column].progress_apply(extract_findings)

    # ===== Segment sentence ======
    tqdm.pandas(desc="Segment sentence from findings...")
    df_list = []
    for df_report in df.progress_apply(format_data_to_sentence, input_column=FINDINGS, axis=1):
        df_list.append(df_report)

    df_new = pd.concat(df_list)

    # ===== Remove space and drop duplicate contenct =====
    tqdm.pandas(desc="Remove space and drop duplicate contenct...")
    df_new[CONTENT] = df_new[SENTENCE].progress_apply(remove_space)
    df_new = df_new.drop_duplicates(subset=[CONTENT])

    # ===== Save file =====
    keep_columns = [
        column
        for column in df_new.columns.to_list()
        if column not in [args.report_column, FINDINGS, CONTENT]
    ]
    df_new[keep_columns].to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
