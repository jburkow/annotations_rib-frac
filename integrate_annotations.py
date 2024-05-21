'''
Filename: integrate_annotations.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 02/20/2021
Description: Takes in two files--the original annotations file and the
    file containing the offsets for each image found during
    dicom_crop_and_equalize.py and combines them to make a new offset
    annotations file.
'''

import argparse
import os
import time
import platform
import pandas as pd
from args import ARGS

def main(parse_args):
    """ Main Function"""
    # Import original annotation data into a DataFrame
    orig_df = pd.read_csv(parse_args.original_csv, names=(['ID', 'height', 'width', 'x1', 'y1', 'x2', 'y2']))
    # Import offset information into a DataFrame
    offset_df = pd.read_csv(parse_args.offset_csv, names=(['ID', 'X Offset', 'Y Offset']))

    # Remove paths and .png from PatientID
    path = orig_df.iloc[0, 0][:orig_df.iloc[0, 0].rfind('Anon_')]
    orig_df['ID'] = orig_df['ID'].str.replace(path, '')
    orig_df['ID'] = orig_df['ID'].str.replace('.png', '')

    # Merge original annotation and offset DataFrames
    merged_df = orig_df.merge(offset_df, on='ID')

    # Subtract X and Y offsets from columns
    merged_df['height'] = merged_df['height'] - merged_df['Y Offset']
    merged_df['width'] = merged_df['width'] - merged_df['X Offset']
    merged_df['x1'] = merged_df['x1'] - merged_df['X Offset']
    merged_df['y1'] = merged_df['y1'] - merged_df['Y Offset']
    merged_df['x2'] = merged_df['x2'] - merged_df['X Offset']
    merged_df['y2'] = merged_df['y2'] - merged_df['Y Offset']

    # Drop offset columns
    merged_df = merged_df.drop(['X Offset', 'Y Offset'], axis=1)

    # Add back path and .png to PatientID
    new_path = os.path.join(ARGS['8_BIT_CROP_HISTEQ_IMAGE_FOLDER'])+'/'
    if platform.system() == 'Windows':
        new_path = new_path.replace('/', '\\')
    orig_df['ID'] = new_path + orig_df['ID'] + '.png'
    merged_df['ID'] = new_path + merged_df['ID'] + '.png'

    # Copy merged DataFrame as the offset annotations DataFrame
    offset_annotations = merged_df.copy()

    # Save annotation information to CSV files
    print('Writing to file...')
    offset_annotations.to_csv(parse_args.out_filename, index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Takes annotations from the original images and combines with offsets found during cropping.')

    parser.add_argument('--original_csv',
                        help='Filename to CSV containing original annotations.')

    parser.add_argument('--offset_csv',
                        help='Filename to CSV containing annotation offsets.')

    parser.add_argument('--out_filename',
                        help='Filename of the CSV to output offset annotations to.')

    parser_args = parser.parse_args()

    # Print out start of execution
    print('Starting execution...')
    start_time = time.perf_counter()

    # Run main function
    main(parser_args)

    # Print out time to complete
    print('Done!')
    end_time = time.perf_counter()
    print('Execution finished in {} seconds.'.format(round(end_time - start_time, 3)))
