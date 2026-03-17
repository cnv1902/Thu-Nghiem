from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("data/datasets/transfusion.csv")

# Chỉnh trực tiếp tại đây
X = 500
RATE = "0.25"
SEED = 42
INPUT_PATH = DEFAULT_INPUT
OUTPUT_PATH = None


def parse_rate(rate_text: str) -> Fraction:
    try:
        rate = Fraction(rate_text)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid rate value: {rate_text}") from exc

    if rate <= 0:
        raise ValueError("rate must be greater than 0.")

    return rate


def compute_class_counts(sample_size: int, rate: Fraction) -> tuple[int, int]:
    if sample_size <= 0:
        raise ValueError("x must be greater than 0.")

    negative_count = Fraction(sample_size, 1) / (1 + rate)
    positive_count = Fraction(sample_size, 1) - negative_count

    if negative_count.denominator != 1 or positive_count.denominator != 1:
        raise ValueError(
            "x and rate do not produce integer class counts. "
            "Choose values where x = negative_count + positive_count and "
            "positive_count / negative_count = rate exactly."
        )

    return int(positive_count), int(negative_count)


def default_output_path(input_path: Path, sample_size: int, rate_text: str) -> Path:
    safe_rate = rate_text.replace("/", "_").replace("\\", "_")
    return input_path.with_name(f"transfusion_{sample_size}_{safe_rate}.csv")


def sample_transfusion(
    input_path: Path,
    output_path: Path,
    sample_size: int,
    rate_text: str,
    seed: int,
) -> Path:
    rate = parse_rate(rate_text)
    positive_needed, negative_needed = compute_class_counts(sample_size, rate)

    df = pd.read_csv(input_path)

    # Mapping label giống hàm load_data
    label_col = "whether he/she donated blood in March 2007"
    transfusion_map = {1: 1, 0: -1}
    labels = df[label_col].map(transfusion_map)

    positive_df = df.loc[labels == 1]
    negative_df = df.loc[labels == -1]

    if len(positive_df) < positive_needed:
        raise ValueError(
            f"Not enough positive rows. Need {positive_needed}, found {len(positive_df)}."
        )
    if len(negative_df) < negative_needed:
        raise ValueError(
            f"Not enough negative rows. Need {negative_needed}, found {len(negative_df)}."
        )

    positive_sample = positive_df.sample(n=positive_needed, random_state=seed)
    negative_sample = negative_df.sample(n=negative_needed, random_state=seed + 1)

    sampled_df = (
        pd.concat([positive_sample, negative_sample], axis=0)
        .sample(frac=1, random_state=seed + 2)
        .reset_index(drop=True)
    )

    sampled_df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    output_path = OUTPUT_PATH or default_output_path(INPUT_PATH, X, RATE)
    saved_path = sample_transfusion(INPUT_PATH, output_path, X, RATE, SEED)
    print(f"Saved sampled Transfusion dataset to: {saved_path}")


if __name__ == "__main__":
    main()