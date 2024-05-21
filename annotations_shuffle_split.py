'''
Filename: annotations_shuffle_split.py
Author(s): Jonathan Burkow, burkowjo@msu.edu, Michigan State University
Last Updated: 05/15/2024
Description: Takes the offset annotations file, shuffles all groups of annotations by PatientID, and
    creates CSV of train/val/test splits.
'''

import argparse
import random
import sys
import time
from pathlib import Path

import pandas as pd

# Set the path to the rib_fracture_utils directory
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
sys.path.append(str(ROOT))

from args import ARGS


def parse_cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=ARGS['RANDOM_SEED'], help='Set the seed for random shuffling (default set in args.py).')
    parser.add_argument('--val', action='store_true', help='Choose whether to include a validation set in the split.')
    parser.add_argument('--anno_file', default=ARGS['ANNOTATION_OFFSET_FILENAME'], help='Path to the annotation file to split into train/val/test files.')
    parser.add_argument('--train_ratio', type=float, default=0.75, help='Proportion of dataset to split into training set.')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Proportion of dataset to split into validation set.')
    parser.add_argument('--filename_shuffled', type=str, default='shuffled_annotations.csv', help='Filename to save shuffled annotations to.')
    parser.add_argument('--filename_train', type=str, default='train_annotations.csv', help='Filename to save training annotations to.')
    parser.add_argument('--filename_val', type=str, default='val_annotations.csv', help='Filename to save validation annotations to.')
    parser.add_argument('--filename_test', type=str, default='test_annotations.csv', help='Filename to save testing annotations to.')
    parser.add_argument('--no_test', action='store_true', help='Use if wanting to use a file with annotations as the test set.')
    parser.add_argument('--test_file', type=str, help='Path to the annotation file to use as the test set.')
    parser.add_argument('--no_save', action='store_true', help='Use to debug.')

    return parser.parse_args()


def main():
    """Main Function"""
    parse_args = parse_cmd_args()

    # Set the random seed for consistent output
    random.seed(parse_args.seed)

    # Load in annotations file
    anno_df = pd.read_csv(parse_args.anno_file,
                          names=('Path', 'Height', 'Width', 'x1', 'y1', 'x2', 'y2'),
                          dtype={'x1': pd.Int64Dtype(), 'y1': pd.Int64Dtype(), 'x2': pd.Int64Dtype(), 'y2': pd.Int64Dtype()})

    # Drop height and width columns, add class column
    anno_df = anno_df.drop(columns=['Height', 'Width'])
    anno_df['class'] = 'fracture'

    # Remove class label for images without annotations
    anno_df.loc[anno_df['x1'].isna(), 'class'] = ''

    if not parse_args.no_test:
        # Pull out all groups of annotations for each imaage and shuffle them
        groups = [df for _, df in anno_df.groupby('Path')]
        random.shuffle(groups)

        # Concatenate the shuffled groups into a new DataFrame
        new_anno_df = pd.concat(groups).reset_index(drop=True)

        # Calculate split train, val, and test sizes and create DataFrames
        if parse_args.val:
            train_size = round(len(groups) * parse_args.train_ratio)
            val_size = round(len(groups) * parse_args.val_ratio)

            train_df = pd.concat(groups[:train_size]).reset_index(drop=True)
            val_df = pd.concat(groups[train_size:(train_size + val_size)]).reset_index(drop=True)
            test_df = pd.concat(groups[(train_size + val_size):]).reset_index(drop=True)
        else:
            train_size = round(len(groups) * (parse_args.train_ratio + parse_args.val_ratio))

            train_df = pd.concat(groups[:train_size]).reset_index(drop=True)
            test_df = pd.concat(groups[train_size:]).reset_index(drop=True)

        # Save shuffled and split DataFrames to files
        if not parse_args.no_save:
            print('Writing to files...')
            new_anno_df.to_csv(parse_args.filename_shuffled, index=False, header=False)
            train_df.to_csv(parse_args.filename_train, index=False, header=False)
            test_df.to_csv(parse_args.filename_test, index=False, header=False)
            if parse_args.val:
                val_df.to_csv(parse_args.filename_val, index=False, header=False)
    else:
        # Remove path and just leave filenames from anno_df
        path = anno_df['Path'].iloc[0][:-16]
        anno_df['Path'] = anno_df['Path'].str.replace(path, '')

        # Load in test annotation file
        test_df = pd.read_csv(parse_args.test_file, names=('Path', 'Height', 'Width', 'x1', 'y1', 'x2', 'y2'))

        # Remove path and just leave filenames from test_df
        test_path = test_df['Path'].iloc[0][:-16]
        test_df['Path'] = test_df['Path'].str.replace(test_path, '')

        # Drop height and width columns, add class column
        test_df = test_df.drop(columns=['Height', 'Width'])
        test_df['class'] = 'fracture'

        # Pull out all groups of annotations for each imaage and shuffle them
        groups = [df for _, df in anno_df.groupby('Path') if df['Path'].unique() not in test_df['Path'].unique()]
        random.shuffle(groups)

        # Concatenate the shuffled groups into a new DataFrame
        new_anno_df = pd.concat(groups).reset_index(drop=True)

        # Calculate split train, val, and test sizes and create DataFrames
        if parse_args.val:
            train_size = round(len(groups) * (1 - parse_args.val_ratio))

            train_df = pd.concat(groups[:train_size]).reset_index(drop=True)
            val_df = pd.concat(groups[train_size:]).reset_index(drop=True)
        else:
            train_df = pd.concat(groups).reset_index(drop=True)

        # Add path back in
        new_anno_df['Path'] = path + new_anno_df['Path']
        train_df['Path'] = path + train_df['Path']
        test_df['Path'] = path + test_df['Path']

        # Save shuffled and split DataFrames to files
        if not parse_args.no_save:
            print('Writing to files...')
            new_anno_df.to_csv(parse_args.filename_shuffled, index=False, header=False)
            train_df.to_csv(parse_args.filename_train, index=False, header=False)
            test_df.to_csv(parse_args.filename_test, index=False, header=False)
            if parse_args.val:
                val_df['Path'] = path + val_df['Path']
                val_df.to_csv(parse_args.filename_val, index=False, header=False)


if __name__ == "__main__":
    print(f"\n{'Starting execution: ' + Path(__file__).name:-^80}\n")
    start_time = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start_time
    print(f"\n{'Done!':-^80}")
    print(f'Execution finished in {elapsed:.3f} seconds ({time.strftime("%-H hr, %-M min, %-S sec", time.gmtime(elapsed))}).\n')
