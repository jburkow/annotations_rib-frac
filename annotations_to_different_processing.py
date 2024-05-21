'''
Filename: annotations_to_different_processing.py
Author(s): Jonathan Burkow, burkowjo@msu.edu, Michigan State University
Last Updated: 05/15/2024
Description: Goes through all normal annotation CSVs and changes the image paths and filenames to
    both binary and varied processing images.
'''

import os
import time
from pathlib import Path
import pandas as pd

# Set the path to the rib_fracture_utils directory
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
sys.path.append(str(ROOT))

from general_utils import print_elapsed

ORIGINAL_CSVS = [
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/test_annotations_seed1337.csv',
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/train_annotations_seed1337_split0.csv',
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/train_annotations_seed1338_split1.csv',
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/train_annotations_seed1339_split2.csv',
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/train_annotations_seed1340_split3.csv',
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/train_annotations_seed1341_split4.csv',
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/train_annotations_seed1342_split5.csv',
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/train_annotations_seed1343_split6.csv',
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/train_annotations_seed1344_split7.csv',
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/train_annotations_seed1345_split8.csv',
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/train_annotations_seed1346_split9.csv',
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/val_annotations_seed1337_split0.csv',
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/val_annotations_seed1338_split1.csv',
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/val_annotations_seed1339_split2.csv',
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/val_annotations_seed1340_split3.csv',
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/val_annotations_seed1341_split4.csv',
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/val_annotations_seed1342_split5.csv',
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/val_annotations_seed1343_split6.csv',
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/val_annotations_seed1344_split7.csv',
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/val_annotations_seed1345_split8.csv',
    '/mnt/home/burkowjo/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/10fold_split_csvs/val_annotations_seed1346_split9.csv'
]


def load_anno_csv(csv_path: str, columns=None, no_height_width: bool = False, has_classes: bool = False) -> pd.DataFrame:
    """
    Load in the annotation CSV into a Pandas DataFrame and return it.

    Parameters
    ----------
    csv_path        : path to the annotation CSV files
    no_height_width : true/false on whether the annotation CSV has height and width columns
    """
    if columns is None:
        columns = ['img_path', 'x1', 'y1', 'x2', 'y2'] if no_height_width else ['img_path', 'height', 'width', 'x1', 'y1', 'x2', 'y2']
        if has_classes:
            columns.append('class')

    # Load in file to DataFrame
    df = pd.read_csv(csv_path, names=columns, dtype={'x1': pd.Int64Dtype(), 'y1': pd.Int64Dtype(), 'x2': pd.Int64Dtype(), 'y2': pd.Int64Dtype()})
    # Drop height and width columns (if they exist); add class column
    if not no_height_width:
        df = df.drop(columns=['Height', 'Width'])
    if not has_classes:
        df['class'] = 'fracture'
    # Remove class label for images without annotations
    df.loc[df['x1'].isna(), 'class'] = ''

    return df


def main():
    """Main Function"""
    binary_img_path = '/mnt/research/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/8bit_images/binary_binary_binary_png/'
    varied_img_path = '/mnt/research/midi_lab/burkowjo_data/processed_fracture_present_1Feb2020_20210902/8bit_images/raw_histeq_bilateral_png/'

    for df in track(ORIGINAL_CSVS, description='Copying Annotation CSVs...'):
        # Load in annotation files to DataFrames
        anno_df = load_anno_csv(df, no_height_width=True, has_classes=True)
        binary_df = anno_df.copy(deep=True)
        varied_df = anno_df.copy(deep=True)

        filename = df.split('/')[-1]
        save_dir = '/'.join(df.split('/')[:-1])

        binary_df['img_path'] = binary_df.apply(lambda x: os.path.join(binary_img_path, x['img_path'].split('/')[-1]), axis=1)
        varied_df['img_path'] = varied_df.apply(lambda x: os.path.join(varied_img_path, x['img_path'].split('/')[-1]), axis=1)

        binary_filename = 'bin-bin-bin_' + filename
        varied_filename = 'raw-hist-bi_' + filename

        binary_df.to_csv(os.path.join(save_dir, binary_filename), index=False, header=False)
        varied_df.to_csv(os.path.join(save_dir, varied_filename), index=False, header=False)


if __name__ == "__main__":
    print('\nStarting execution...')
    start_time = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start_time
    print('Done!')
    print(f'Execution finished in {elapsed:.3f} seconds ({time.strftime("%-H hr, %-M min, %-S sec", time.gmtime(elapsed))}).\n')
