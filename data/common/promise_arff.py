import io
import pandas as pd


def read_promise_arff_dataframe(file_path):
    """
    Read PROMISE-style ARFF text files into a pandas DataFrame.

    The files often contain long '%' comment blocks, ARFF metadata
    (@relation/@attribute), and a @data section with comma-separated values.
    """
    attributes = []
    data_lines = []
    in_data = False

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("%"):
                continue

            lower_line = line.lower()
            if not in_data:
                if lower_line.startswith("@attribute"):
                    parts = line.split(None, 2)
                    if len(parts) >= 2:
                        attr_name = parts[1].strip("'\"")
                        attributes.append(attr_name)
                elif lower_line.startswith("@data"):
                    in_data = True
                continue

            if line.startswith("@"):
                continue
            data_lines.append(line)

    if not attributes:
        raise ValueError(f"No @attribute section found in {file_path}")
    if not data_lines:
        raise ValueError(f"No @data rows found in {file_path}")

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        header=None,
        names=attributes,
        na_values=["?"],
    )
    return df
