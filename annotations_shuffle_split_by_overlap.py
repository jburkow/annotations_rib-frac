'''
Filename: annotations_shuffle_split_by_overlap.py
Author(s): Jonathan Burkow, burkowjo@msu.edu, Michigan State University
Last Updated: 05/09/2022
Description: Takes the offset annotations file, shuffles all groups of annotations by PatientID,
    and creates a train/val/test splits.
'''

import argparse
import os
import random
import time
from math import ceil

import numpy as np
import pandas as pd
from rich.console import Console

from args import ARGS
from general_utils import print_elapsed


def load_anno_csv(csv_path: str, no_height_width: bool = False) -> pd.DataFrame:
    """
    Load in the annotation CSV into a Pandas DataFrame and return it.

    Parameters
    ----------
    csv_path        : path to the annotation CSV files
    no_height_width : true/false on whether the annotation CSV has height and width columns
    """
    columns = ('Path', 'x1', 'y1', 'x2', 'y2') if no_height_width else ('Path', 'Height', 'Width', 'x1', 'y1', 'x2', 'y2')

    # Load in file to DataFrame
    df = pd.read_csv(csv_path, names=columns, dtype={'x1': pd.Int64Dtype(), 'y1': pd.Int64Dtype(), 'x2': pd.Int64Dtype(), 'y2': pd.Int64Dtype()})
    # Drop height and width columns (if they exist); add class column
    if not no_height_width:
        df = df.drop(columns=['Height', 'Width'])
    df['class'] = 'fracture'
    # Remove class label for images without annotations
    df.loc[df['x1'].isna(), 'class'] = ''

    return df


def main(parse_args):
    """Main Function"""
    # Set the random seed for consistent output
    random.seed(parse_args.seed)

    # Load in annotation files to DataFrames
    anno_df1 = load_anno_csv(parse_args.anno_csv1)
    anno_df2 = load_anno_csv(parse_args.anno_csv2, no_height_width=True)

    ten_percent_of_data = ceil(len(anno_df1['Path'].unique()) * 0.10)

    # Get unique paths/PatientIDs in each annotation CSV
    unique_df1_ids = anno_df1['Path'].unique().tolist()
    unique_df2_ids = anno_df2['Path'].unique().tolist()

    # Find overlapping PatientIDs (should be same length as unique_df2_ids)
    overlap_ids = [patient for patient in unique_df1_ids if patient in unique_df2_ids]

    # Find IDs of all fracture absent images and sample 10% from them
    frac_absent = anno_df1[anno_df1['Path'].str.contains('unknown')]
    frac_absent = frac_absent[frac_absent['x1'].isna()]
    unique_frac_absent = frac_absent['Path'].unique().tolist()
    ten_percent_frac_absent = random.sample(unique_frac_absent, ten_percent_of_data)

    # Randomly sample 10% of the overlapping PatientIDs and add to 10% frac absent for the test set
    ten_percent_frac_present = random.sample(overlap_ids, ten_percent_of_data)
    test_set_ids = ten_percent_frac_absent + ten_percent_frac_present

    test_groups = [df for _, df in anno_df1.groupby('Path') if df['Path'].unique() in test_set_ids]
    test_df = pd.concat(test_groups).reset_index(drop=True)

    # Pull out all groups of annotations not part of the test set, and shuffle them
    groups = [df for _, df in anno_df1.groupby('Path') if df['Path'].unique() not in test_set_ids]
    random.shuffle(groups)

    if not parse_args.no_save:
        print('Splitting and writing to files...')
        test_csv_save_name = f"test_annotations_seed{parse_args.seed}.csv"
        test_csv_save_name = os.path.join(ARGS['PROCESSED_DATA_FOLDER'], test_csv_save_name)
        test_df.to_csv(test_csv_save_name, index=False, header=False)
        for i in range(parse_args.splits):
            temp_groups = groups.copy()

            # Set new seed based on the current fold iteration
            new_seed = parse_args.seed + i
            random.seed(new_seed)
            np.random.seed(new_seed)

            # Sample 10% of the data for validation set
            sample_idx = np.random.randint(0, len(temp_groups), ten_percent_of_data).tolist()
            val_groups = [temp_groups[idx] for idx in sample_idx]
            for idx in sorted(sample_idx, reverse=True):
                temp_groups.pop(idx)

            # Create DataFrame with validation groups
            val_df = pd.concat(val_groups).reset_index(drop=True)

            # Use remaining groups to create the training set DataFrame
            train_df = pd.concat(temp_groups).reset_index(drop=True)

            # Save both training and validation DataFrames to a CSV
            train_csv_save_name = f"train_annotations_seed{new_seed}_split{i}.csv"
            train_csv_save_name = os.path.join(ARGS['PROCESSED_DATA_FOLDER'], train_csv_save_name)
            val_csv_save_name = f"val_annotations_seed{new_seed}_split{i}.csv"
            val_csv_save_name = os.path.join(ARGS['PROCESSED_DATA_FOLDER'], val_csv_save_name)
            train_df.to_csv(train_csv_save_name, index=False, header=False)
            val_df.to_csv(val_csv_save_name, index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=ARGS['RANDOM_SEED'],
                        help='Set the seed for random shuffling (default set in args.py).')

    parser.add_argument('--anno_csv1',
                        help='Path to first annotaiton file.')

    parser.add_argument('--anno_csv2',
                        help='Path to second annotaiton file.')

    parser.add_argument('--no_save', action='store_true',
                        help='Use to debug.')

    parser.add_argument('--splits', type=int, default=6,
                        help='Number of splits of data to generate')

    parser_args = parser.parse_args()

    print()
    console = Console()
    console.rule(f'Running {os.path.basename(__file__)}', style='deep_sky_blue1')
    console.log('Starting execution...', style='deep_sky_blue1')
    start_time = time.perf_counter()
    main(parser_args)
    elapsed = time.perf_counter() - start_time
    console.log(print_elapsed(elapsed), style="bold cyan3")
    console.rule('[green]Done!', style='green')
    print()
