# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import pandas as pd
import os


def clean_csv(file_path):
    """
    Clean the CSV file by removing trailing commas and newlines.

    :param file_path: Path to the CSV file.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    cleaned_lines = [line.rstrip(',\n') + '\n' for line in lines]

    with open(file_path, 'w') as file:
        file.writelines(cleaned_lines)


def read_csv(file_path):
    """
    Read a CSV file into a DataFrame.

    :param file_path: Path to the CSV file.
    :return: DataFrame containing the CSV data.
    """
    return pd.read_csv(file_path)


def compare_dataframes(df_a, df_b, key_columns=None):
    """
    Compare two DataFrames based on specified key columns and other columns.

    :param df_a: First DataFrame.
    :param df_b: Second DataFrame.
    :param key_columns: List of column names to use as keys for comparison.
    :return: Tuples of DataFrames containing items only in df_a, items only in df_b, and differences.
    """
    if key_columns is None:
        key_columns = ['Op Name', 'IO', 'Slot']
    # Set index using key columns for easier comparison
    df_a.set_index(key_columns, inplace=True)
    df_b.set_index(key_columns, inplace=True)

    # Find rows that are only in df_a and only in df_b
    only_in_a = df_a.drop(index=df_b.index).reset_index()
    only_in_b = df_b.drop(index=df_a.index).reset_index()

    # Find common rows and compare other columns
    common_keys = df_a.index.intersection(df_b.index)
    differences = []
    for key in common_keys:
        row_a = df_a.loc[key].drop('Timestamp')
        row_b = df_b.loc[key].drop('Timestamp')
        if not row_a.equals(row_b):
            diff = {'key': key}
            diff.update({col: (row_a[col], row_b[col]) for col in row_a.index if row_a[col] != row_b[col]})
            differences.append(diff)

    return only_in_a, only_in_b, differences


def compare_csv_files(dir1, dir2):
    """
    Compare all CSV files in two directories.

    :param dir1: Path to the first directory.
    :param dir2: Path to the second directory.
    """
    # Get all CSV files ending with 'statistic.csv' in both directories
    files1 = [os.path.join(root, file) for root, dirs, files in os.walk(dir1)
              for file in files if file.endswith('statistic.csv')]
    files2 = [os.path.join(root, file) for root, dirs, files in os.walk(dir2)
              for file in files if file.endswith('statistic.csv')]

    # Ensure both directories have the same number of files
    assert len(files1) == len(files2), f"The directories do not have the same number of files. \
    Directory 1 has {len(files1)} files, Directory 2 has {len(files2)} files."
    assert files1, "No CSV files found in the first directory."
    # Compare each pair of CSV files
    for file1, file2 in zip(sorted(files1), sorted(files2)):
        compare_csv(file1, file2)


def compare_csv(file1, file2):
    """
    Compare two individual CSV files.

    :param file1: Path to the first CSV file.
    :param file2: Path to the second CSV file.
    """
    print(f"\nComparing files:\n{file1}\n{file2}")

    # Clean the CSV files
    clean_csv(file1)
    clean_csv(file2)

    # Read and prepare the CSV files
    df1 = read_csv(file1)
    df2 = read_csv(file2)

    # Compare the DataFrames
    only_in_df1, only_in_df2, differences = compare_dataframes(df1, df2)

    # Output the comparison results
    if not only_in_df1.empty:
        print("\nOnly in file1:")
        print(only_in_df1)

    if not only_in_df2.empty:
        print("\nOnly in file2:")
        print(only_in_df2)

    if differences:
        print("\nDifferences:")
        for diff in differences:
            print(f"Key: {diff['key']}")
            for col, values in diff.items():
                if col != 'key':
                    print(f"  Column: {col}, File1 value: {values[0]}, File2 value: {values[1]}")

    # Assertions to check for discrepancies
    assert only_in_df1.empty, "There are items only in file1."
    assert only_in_df2.empty, "There are items only in file2."
    assert not differences, "There are differences between the files."
