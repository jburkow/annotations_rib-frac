'''
Filename: annotation_csvs_from_json.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 12/15/2021
Description: Create annotation files from JSON files and integrate offsets from processing stage.
'''

import argparse
import os
import time
import json
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from dicom_utils import (load_dicom_image, crop_dicom, hist_equalization, create_rgb,
                         scale_image_to_depth, save_to_png, save_to_npy, extract_bboxes)
from args import ARGS
from general_utils import read_file


def main(parse_args):
    # Import offset information into a DataFrame
    offset_df = pd.read_csv(parse_args.offset_csv, names=(['patient_id', 'x_offset', 'y_offset']))

    annotation_jsons = sorted([os.path.join(root, file) for root, _, files in os.walk(parse_args.json_path) for file in files])

    original_annotations = []
    offset_annotations = []
    for _, json_file in tqdm(enumerate(annotation_jsons), desc='Collating 2nd read annotations', total=len(annotation_jsons)):
        # Pull out annotation information
        with open(json_file, 'r') as f_in:
            annotation_data = json.load(f_in)

        # Get PatientID of current file
        filename = annotation_data['filename'].split('/')[-1]
        patient_id = filename[:filename.find('-')]

        # Get current PatientID offsets
        current_offsets_df = offset_df[offset_df.patient_id == patient_id]
        x_offset = current_offsets_df.x_offset.item()
        y_offset = current_offsets_df.y_offset.item()

        # Extract lists of bounding box points from annotation file
        tl_xs, tl_ys, br_xs, br_ys = extract_bboxes(annotation_data)

        # Integrate offsets to bounding box points
        offset_tl_xs = [val - x_offset for val in tl_xs]
        offset_tl_ys = [val - y_offset for val in tl_ys]
        offset_br_xs = [val - x_offset for val in br_xs]
        offset_br_ys = [val - y_offset for val in br_ys]

        # Append annotations based on the original image
        for x1, y1, x2, y2 in zip(tl_xs, tl_ys, br_xs, br_ys):
            info = [os.path.join(ARGS['8_BIT_OG_IMAGE_FOLDER'], patient_id + '.png'), x1, y1, x2, y2]
            original_annotations.append(info)

        # Append annotations based on the cropped image
        for x1, y1, x2, y2 in zip(offset_tl_xs, offset_tl_ys, offset_br_xs, offset_br_ys):
            info = [os.path.join(ARGS['8_BIT_CROP_HISTEQ_IMAGE_FOLDER'], patient_id + '.png'), x1, y1, x2, y2]
            offset_annotations.append(info)

    # Export original and offset annotation lists to files
    orig_annotations_df = pd.DataFrame(original_annotations, columns=(['ID', 'x1', 'y1', 'x2', 'y2']))
    offset_annotations_df = pd.DataFrame(offset_annotations, columns=(['ID', 'x1', 'y1', 'x2', 'y2']))

    orig_annotations_df.to_csv(os.path.join(parse_args.save_path, 'original_annotations.csv'), index=False, header=False)
    offset_annotations_df.to_csv(os.path.join(parse_args.save_path, 'offset_annotations.csv'), index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Do stuff.')

    parser.add_argument('--json_path',
                        help='Path to directory containing JSON annotation files.')

    parser.add_argument('--offset_csv',
                        help='Path to CSV file containing offsets for each image.')

    parser.add_argument('--save_path',
                        help='Path to save original and offset annotation CSV files to.')

    parser_args = parser.parse_args()

    # Print out start of execution
    print('\nStarting execution...')
    start_time = time.perf_counter()

    # Run main function
    main(parser_args)

    # Print out time to complete
    print('\nDone!')
    end_time = time.perf_counter()
    print(f'Execution finished in {end_time - start_time:.3f} seconds.\n')
