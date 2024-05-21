'''
Filename: annotation_statistics.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 02/04/2021
Description: Calculates statistics for ground truth bounding boxes in
    both pixel and physical dimensions.
'''

import argparse
import os
import time
import json
import numpy as np
import pandas as pd
from args import ARGS
from general_utils import print_iter

def main(parse_args):
    """Main Function"""
    # Load in the annotation data
    bbox_stats_df = pd.read_csv(ARGS['ANNOTATION_OFFSET_FILENAME'], names=['Path', 'Img Height', 'Img Width', 'x1', 'y1', 'x2', 'y2'])
    bbox_stats_df['Row Spacing'] = np.zeros(shape=(bbox_stats_df.shape[0], 1))
    bbox_stats_df['Col Spacing'] = np.zeros(shape=(bbox_stats_df.shape[0], 1))

    # Loop through all annotation files and calculate statistics
    annotation_list = [os.path.join(ARGS['ANNOTATION_FOLDER'], name) for name in os.listdir(ARGS['ANNOTATION_FOLDER']) if name.endswith('.json')]
    for i, row in bbox_stats_df.iterrows():
        print_iter(len(bbox_stats_df), i, 'row')

        # Pull out Patient ID from path
        patient_id = row[0][:-4].replace(ARGS['8_BIT_CROP_HISTEQ_IMAGE_FOLDER'] + '/', '')

        # Connect Patient ID to propr annotation JSON file and pull out pixel spacing
        annotation_json = [name for name in annotation_list if patient_id in name]
        with open(annotation_json[0]) as read_json:
            # Load in the JSON data to a dictionary
            data = json.load(read_json)

            # Check if pixel_spacing is provided; if not, use 0
            if data['pixel_spacing']:
                row_pixel_spacing = float(data['pixel_spacing'][:data['pixel_spacing'].find('\\')])
                col_pixel_spacing = float(data['pixel_spacing'][data['pixel_spacing'].find('\\')+1:])
            else:
                row_pixel_spacing = 0.0
                col_pixel_spacing = 0.0

            bbox_stats_df.iloc[i,-2] = row_pixel_spacing
            bbox_stats_df.iloc[i,-1] = col_pixel_spacing
    print('') # End print stream from loop

    # Calculate bounding box heights and widths in pixels
    bbox_stats_df['Px Height'] = bbox_stats_df['y2'] - bbox_stats_df['y1']
    bbox_stats_df['Px Width'] = bbox_stats_df['x2'] - bbox_stats_df['x1']

    # Calculate bounding box heights and widths in mm
    bbox_stats_df['mm Height'] = bbox_stats_df['Px Height'] * bbox_stats_df['Row Spacing']
    bbox_stats_df['mm Width'] = bbox_stats_df['Px Width'] * bbox_stats_df['Col Spacing']

    # Calculate overall statistics across all bounding boxes
    pixel_box_h = [str(bbox_stats_df['Px Height'].mean()),
                   str(bbox_stats_df['Px Height'].std()),
                   str(bbox_stats_df['Px Height'].median()),
                   str(bbox_stats_df['Px Height'].min()),
                   str(bbox_stats_df['Px Height'].max())]

    pixel_box_w = [str(bbox_stats_df['Px Width'].mean()),
                   str(bbox_stats_df['Px Width'].std()),
                   str(bbox_stats_df['Px Width'].median()),
                   str(bbox_stats_df['Px Width'].min()),
                   str(bbox_stats_df['Px Width'].max())]

    mm_box_h = [str(bbox_stats_df['mm Height'][bbox_stats_df['mm Height'] > 0].mean()),
                str(bbox_stats_df['mm Height'][bbox_stats_df['mm Height'] > 0].std()),
                str(bbox_stats_df['mm Height'][bbox_stats_df['mm Height'] > 0].median()),
                str(bbox_stats_df['mm Height'][bbox_stats_df['mm Height'] > 0].min()),
                str(bbox_stats_df['mm Height'][bbox_stats_df['mm Height'] > 0].max())]

    mm_box_w = [str(bbox_stats_df['mm Width'][bbox_stats_df['mm Width'] > 0].mean()),
                str(bbox_stats_df['mm Width'][bbox_stats_df['mm Width'] > 0].std()),
                str(bbox_stats_df['mm Width'][bbox_stats_df['mm Width'] > 0].median()),
                str(bbox_stats_df['mm Width'][bbox_stats_df['mm Width'] > 0].min()),
                str(bbox_stats_df['mm Width'][bbox_stats_df['mm Width'] > 0].max())]

    # Save all statistics and summary to files
    print('Writing to files...')
    bbox_stats_df.to_csv(parse_args.filename_stats, index=False)
    with open(parse_args.filename_summary, 'w+') as out_file:
        out_file.write('Summary Statistics of Bounding Box Annotations\n')
        out_file.write('-'*46 + '\n')
        out_file.write(' '*27 + 'Minimum, Maximum, Average, Standard Deviation, Median\n')
        out_file.write('Bounding Box (px) Heights: (' + ', '.join([pixel_box_h[3], pixel_box_h[4], pixel_box_h[0], pixel_box_h[1], pixel_box_h[2]]) + ')\n')
        out_file.write('Bounding Box (px) Widths : (' + ', '.join([pixel_box_w[3], pixel_box_w[4], pixel_box_w[0], pixel_box_w[1], pixel_box_w[2]]) + ')\n')
        out_file.write('\n')
        out_file.write('Bounding Box (mm) Heights: (' + ', '.join([mm_box_h[3], mm_box_h[4], mm_box_h[0], mm_box_h[1], mm_box_h[2]]) + ')\n')
        out_file.write('Bounding Box (mm) Widths : (' + ', '.join([mm_box_w[3], mm_box_w[4], mm_box_w[0], mm_box_w[1], mm_box_w[2]]) + ')\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creates a CSV with statistics for ground truth annotations.')

    parser.add_argument('--filename_stats', type=str, default='annotation_stats.csv',
                        help='Filename to save all statistics to.')

    parser.add_argument('--filename_summary', type=str, default='annotation_stats_summary.txt',
                        help='Filename to save summary of statistics to.')

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
