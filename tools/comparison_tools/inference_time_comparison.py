#!/usr/bin/env python3

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import os

"""
This scripts goes over 2 json with times vector per scan id and compares the time
"""

SAVE_TO_EXCEL = True


def remove_outliers(list_numbers):
    q1 = np.quantile(list_numbers, 0.25)
    q3 = np.quantile(list_numbers, 0.75)
    iqr = q3 - q1
    if iqr > 0:
        low_thresh = q1 - 1.5 * iqr
        high_thresh = q3 + 1.5 * iqr
        list_numbers = [x for x in list_numbers if x > low_thresh]
        list_numbers = [x for x in list_numbers if x < high_thresh]
    return list_numbers


def main():
    # Load input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-j1", "--jason1", type=str, required=True, help='This path is set to be ground truth')
    parser.add_argument("-j2", "--jason2", type=str, required=True, help='Path to compare')
    parser.add_argument("-o", '--output-dir', type=str, default=None)
    parser.add_argument('--test-name1', type=str)
    parser.add_argument('--test-name2', type=str)
    args = parser.parse_args()
    jason1 = args.jason1
    jason2 = args.jason2
    test_name1 = args.test_name1
    test_name2 = args.test_name2
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(jason1, 'r') as f:
        times_v1 = json.load(f)
    with open(jason2, 'r') as f:
        times_v2 = json.load(f)
    dir1_name = os.path.split(os.path.split(jason1)[0])[1]
    dir2_name = os.path.split(os.path.split(jason2)[0])[1]
    if test_name1 is None:
        test_name1 = dir1_name
    if test_name2 is None:
        test_name2 = dir2_name

    statistics = {}
    scans_list, average_time_v1, average_time_v2, median_time_v1, median_time_v2 = [], [], [], [], []
    for scan_id in times_v2:
        scans_list.append(scan_id[:4])
        if scan_id in times_v1:
            scan_times_v1 = remove_outliers(times_v1[scan_id])
        else:
            scan_id_key = [key for key in times_v1 if scan_id in key]
            scan_times_v1 = remove_outliers(times_v1[scan_id_key[0]])
        scan_times_v2 = remove_outliers(times_v2[scan_id])
        statistics[scan_id] = {}
        average_time_v1.append(np.mean(scan_times_v1))
        average_time_v2.append(np.mean(scan_times_v2))
        median_time_v1.append(np.round(np.median(scan_times_v1), 2))
        median_time_v2.append(np.round(np.median(scan_times_v2), 2))

    plt.figure(figsize=(16, 8))
    plt.plot(scans_list, average_time_v1, 'o--', color='darkblue', label='average_time_v1')
    plt.plot(scans_list, median_time_v1, color='blue', label='median_time_v1')
    plt.plot(scans_list, average_time_v2, 'o--', color='darkgreen', label='average_time_v2')
    plt.plot(scans_list, median_time_v2, color='green', label='median_time_v2')
    plt.xlabel("Scan id"), plt.ylabel('Time [sec]')
    plt.ylim([round(min(average_time_v1 + median_time_v1 + average_time_v2 + median_time_v2) - 2),
              round(max(average_time_v1 + median_time_v1 + average_time_v2 + median_time_v2) + 2)])
    # plt.title(f'Time comparison measured {len(times_v1[scan_id])} times on v1, {len(times_v2[scan_id])} times on v2'
    plt.title(f'Time comparison measured {len(times_v2[scan_id])} times')
    plt.legend([f'{test_name1} average', f'{test_name1} median', f'{test_name2} average', f'{test_name2} median'])

    if output_dir:
        plt.savefig(os.path.join(output_dir, f'Time_comparison.png'))
        plt.close()

    columns = ['scan id', f'{test_name1} average', f'{test_name1} median', f'{test_name2} average', f'{test_name2} median']
    results_table = [scans_list, np.around(average_time_v1, 3), median_time_v1,
                     np.around(average_time_v2, 3), median_time_v2]
    results_table = np.array(results_table).T.tolist()
    fig, axs = plt.subplots(figsize=(20, 8))
    axs.axis('tight'), axs.axis('off')
    table = axs.table(cellText=results_table, colLabels=columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'Time_comparison_time_table.png'))
        plt.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        raise err


