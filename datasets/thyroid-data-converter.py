import os
import re

import pandas as pd


def convert_thyroid_to_csv(
    input_file, output_file, has_header=False, custom_headers=None
):
    """
    Converts thyroid dataset files to CSV format

    Parameters:
    input_file (str): Path to the input data file
    output_file (str): Path to save the CSV file
    has_header (bool): Whether the input file has a header
    custom_headers (list): Custom column headers if needed
    """
    print(f"Converting {input_file} to CSV...")

    # Make sure the file exists and is readable
    if not os.path.isfile(input_file):
        print(f"File not found: {input_file}")
        return None

    try:
        # Read a few lines to analyze the file format
        with open(input_file, "r", encoding="utf-8", errors="replace") as f:
            sample_lines = [f.readline().strip() for _ in range(5) if f.readline()]

        # If we couldn't read any lines, the file might be empty or corrupt
        if not sample_lines:
            print(f"Warning: {input_file} appears to be empty or unreadable")
            return None

        # For the new-thyroid.data file which has a specific format
        if "new-thyroid" in input_file:
            # The new-thyroid.data has values separated by spaces with no headers
            # The 5 columns are: Class, T3-resin, Thyroxin, Triiodothyronine, TSH, Maximal absolute difference
            data = []
            with open(input_file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Split by one or more spaces
                        values = re.split(r"\s+", line)
                        if len(values) >= 5:
                            data.append(values)

            # Use custom headers or default ones
            headers = (
                custom_headers
                if custom_headers
                else [
                    "Class",
                    "T3_resin",
                    "Thyroxin",
                    "Triiodothyronine",
                    "TSH",
                    "Max_diff",
                ]
            )

            # Create DataFrame and save to CSV
            df = pd.DataFrame(
                data, columns=headers[: len(data[0])] if data else headers
            )
            df.to_csv(output_file, index=False)
            print(f"Successfully saved to {output_file}")
            return df

        # For ANN datasets (ann-train.data and ann-test.data)
        elif "ann-" in input_file:
            # Try to determine if the file is comma-separated or has another format
            has_commas = any("," in line for line in sample_lines)

            if has_commas:
                # Comma-separated values format
                try:
                    df = pd.read_csv(input_file, header=None)

                    # Apply custom headers if provided
                    if custom_headers:
                        # Make sure we have enough header names for all columns
                        if len(custom_headers) < len(df.columns):
                            custom_headers = custom_headers + [
                                f"Attr{i+1}"
                                for i in range(len(custom_headers), len(df.columns))
                            ]
                        df.columns = custom_headers[: len(df.columns)]
                    else:
                        df.columns = [f"Attr{i+1}" for i in range(len(df.columns))]

                    df.to_csv(output_file, index=False)
                    print(f"Successfully saved to {output_file}")
                    return df
                except Exception as e:
                    print(f"Error with comma parsing: {e}")

            # Fallback to manual parsing
            data = []
            with open(input_file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        if "," in line:
                            values = line.split(",")
                        else:
                            # If no commas, try splitting by whitespace
                            values = re.split(r"\s+", line)
                        data.append(values)

            # Check if we have data
            if not data:
                print(f"No valid data found in {input_file}")
                return None

            # Normalize columns (some rows may have different numbers of columns)
            max_cols = max(len(row) for row in data)
            for i, row in enumerate(data):
                if len(row) < max_cols:
                    data[i] = row + [""] * (max_cols - len(row))

            # The ANN dataset has attributes - using generic column names if no custom headers
            headers = (
                custom_headers
                if custom_headers
                else [f"Attr{i+1}" for i in range(max_cols)]
            )
            headers = headers[:max_cols]  # Make sure we don't have too many headers

            df = pd.DataFrame(data, columns=headers)
            df.to_csv(output_file, index=False)
            print(f"Successfully saved to {output_file}")
            return df

        # For general thyroid datasets (allhypo.data, allbp.data, etc.)
        else:
            # First, look for potential delimiters in the sample lines
            delimiters = {"comma": ",", "pipe": "|", "tab": "\t", "space": " "}
            delimiter_counts = {
                name: sum(d in line for line in sample_lines)
                for name, d in delimiters.items()
            }

            # Choose the most common delimiter
            most_common = max(delimiter_counts.items(), key=lambda x: x[1])
            if most_common[1] > 0:
                primary_delimiter = delimiters[most_common[0]]
                print(f"Detected {most_common[0]} as the primary delimiter")
            else:
                # If no clear delimiter, default to whitespace
                primary_delimiter = None
                print("No clear delimiter detected, using whitespace")

            # Try reading with pandas first
            try:
                if primary_delimiter:
                    df = pd.read_csv(
                        input_file,
                        sep=primary_delimiter,
                        header=0 if has_header else None,
                        engine="python",
                        on_bad_lines="skip",
                    )
                else:
                    df = pd.read_csv(
                        input_file,
                        delim_whitespace=True,
                        header=0 if has_header else None,
                        engine="python",
                        on_bad_lines="skip",
                    )

                # Apply custom headers if provided and no header in file
                if custom_headers and not has_header:
                    if len(custom_headers) < len(df.columns):
                        # Extend headers if needed
                        custom_headers = custom_headers + [
                            f"Attr{i+1}"
                            for i in range(len(custom_headers), len(df.columns))
                        ]
                    df.columns = custom_headers[: len(df.columns)]

                # Save to CSV
                df.to_csv(output_file, index=False)
                print(f"Successfully saved to {output_file}")
                return df

            except Exception as e:
                print(f"Pandas parsing error: {e}")
                print("Falling back to manual parsing...")

                # Fallback: manual line-by-line parsing
                data = []
                with open(input_file, "r", encoding="utf-8", errors="replace") as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            line = line.strip()
                            if not line:
                                continue

                            # Try different splitting strategies
                            if primary_delimiter:
                                values = line.split(primary_delimiter)
                            else:
                                # Split by any combination of whitespace
                                values = re.split(r"\s+", line)

                            # Clean up values (remove quotes, extra whitespace)
                            values = [v.strip().strip("\"'") for v in values]
                            data.append(values)
                        except Exception as line_error:
                            print(
                                f"Warning: Error parsing line {line_num}: {line_error}"
                            )

                if not data:
                    print(f"Failed to extract any valid data from {input_file}")
                    return None

                # Normalize lengths (take max length and pad shorter rows)
                max_len = max(len(row) for row in data)
                for i, row in enumerate(data):
                    if len(row) < max_len:
                        data[i] = row + [""] * (max_len - len(row))

                # Generate default headers if needed
                if custom_headers:
                    if len(custom_headers) < max_len:
                        headers = custom_headers + [
                            f"Attr{i+1}" for i in range(len(custom_headers), max_len)
                        ]
                    else:
                        headers = custom_headers[:max_len]
                else:
                    headers = [f"Attr{i+1}" for i in range(max_len)]

                df = pd.DataFrame(data, columns=headers)
                df.to_csv(output_file, index=False)
                print(f"Successfully saved to {output_file} using manual parsing")
                return df

    except Exception as e:
        print(f"Fatal error processing {input_file}: {e}")
        return None


def main():
    # Example usage:
    # Look for data files in the current directory and subdirectories

    # Create output directory if it doesn't exist
    os.makedirs("thyroid_csv", exist_ok=True)

    # First, find all available .data files in the current directory and subdirectories
    print("Searching for thyroid dataset files...")
    available_files = []
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".data"):
                file_path = os.path.join(root, file)
                available_files.append(file_path)

    if not available_files:
        print(
            "No .data files found. Please make sure the dataset is properly extracted."
        )
        return

    print(f"Found {len(available_files)} .data files:")
    for file in available_files:
        print(f"  - {file}")

    # Process all found files or let user specify which ones to convert
    files_to_convert = available_files

    # Convert each file
    for input_file in files_to_convert:
        # Get base filename for output
        base_filename = os.path.basename(input_file)
        output_file = os.path.join(
            "thyroid_csv", base_filename.replace(".data", ".csv")
        )

        # For new-thyroid.data, use custom headers
        if "new-thyroid.data" in input_file:
            custom_headers = [
                "Class",
                "T3_resin",
                "Thyroxin",
                "Triiodothyronine",
                "TSH",
                "Max_diff",
            ]
            convert_thyroid_to_csv(
                input_file, output_file, has_header=False, custom_headers=custom_headers
            )

        # For ANN datasets, use their specific format
        elif "ann-" in input_file and ("train" in input_file or "test" in input_file):
            # The ANN dataset has attributes according to documentation
            # Based on ann-thyroid.names, the features include measurements and class label
            custom_headers = [
                "Class",
                "age",
                "sex",
                "on_thyroxine",
                "query_thyroxine",
                "on_antithyroid",
                "sick",
                "pregnant",
                "thyroid_surgery",
                "I131_treatment",
                "query_hypothyroid",
                "query_hyperthyroid",
                "lithium",
                "goitre",
                "tumor",
                "hypopituitary",
                "psych",
                "TSH_measured",
                "TSH",
                "T3_measured",
                "T3",
                "TT4_measured",
                "TT4",
                "T4U_measured",
                "T4U",
                "FTI_measured",
                "FTI",
                "TBG_measured",
                "TBG",
            ]
            convert_thyroid_to_csv(
                input_file, output_file, has_header=False, custom_headers=custom_headers
            )

        # For allhypo, allbp, etc. datasets
        elif any(
            name in input_file
            for name in [
                "allhypo",
                "allbp",
                "allrep",
                "allhyper",
                "sick",
                "thyroid0387",
            ]
        ):
            # These datasets have common attributes (source: documentation files)
            custom_headers = [
                "age",
                "sex",
                "on_thyroxine",
                "query_thyroxine",
                "on_antithyroid",
                "thyroid_surgery",
                "query_hypothyroid",
                "query_hyperthyroid",
                "pregnant",
                "sick",
                "tumor",
                "lithium",
                "goitre",
                "TSH_measured",
                "TSH",
                "T3_measured",
                "T3",
                "TT4_measured",
                "TT4",
                "T4U_measured",
                "T4U",
                "FTI_measured",
                "FTI",
                "TBG_measured",
                "TBG",
                "referral_source",
                "Class",
            ]
            convert_thyroid_to_csv(
                input_file, output_file, has_header=False, custom_headers=custom_headers
            )

        # For any other files, try automatic conversion
        else:
            convert_thyroid_to_csv(input_file, output_file)


if __name__ == "__main__":
    main()
