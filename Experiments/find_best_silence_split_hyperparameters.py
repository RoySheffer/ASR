import sys
sys.path.append("../")
import numpy as np
import tqdm
from ASR.utilities import silence_split
from ASR.parser import get_dataset_splits
import concurrent.futures
import multiprocessing


def compute_distance(row, frame_size, amp_th, percentile):
    audio = row[2]
    text = row[4]
    indexes = silence_split(audio, frame_size=frame_size, amp_th=amp_th, percentile=percentile)
    predicted_num_words = len(indexes)
    actual_num_words = len(text.split(" "))
    return abs(predicted_num_words - actual_num_words)


num_cores = multiprocessing.cpu_count()
print(f"Number of cores:{num_cores}")
sample_rate = 16000
test_set, train_set, val_set = get_dataset_splits(sample_rate)
set2df = {"train": train_set, "val": val_set, "test": test_set}

for set_name in set2df.keys():
    best_avg_dist = np.inf
    best_frame_size = 0
    best_amp_th = 0
    best_percentile = 0

    frame_range = np.arange(600, 2600, 100)
    amp_th_range = np.arange(0.02,0.36,0.02)
    percentile_range = np.arange(64,100,2)
    for frame_size in tqdm.tqdm(frame_range, f"Finding best hyperparameters for {set_name} set"):
        for amp_th in amp_th_range:
            for percentile in percentile_range:
                # parallel distance computation
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
                    dist_list = list(executor.map(compute_distance, set2df[set_name].itertuples(index=False),
                                                  [frame_size] * len(set2df[set_name]),
                                                  [amp_th] * len(set2df[set_name]),
                                                  [percentile] * len(set2df[set_name])))
                    cur_acc_dist = sum(dist_list)


                # find the best hyperparameters:
                cur_avg_dist = cur_acc_dist / len(set2df[set_name])
                if cur_avg_dist < best_avg_dist:
                    best_avg_dist = cur_avg_dist
                    best_frame_size = frame_size
                    best_amp_th = amp_th
                    best_percentile = percentile
                    print(f"Found better hyperparameters with distance:{best_avg_dist}:")
                    print(f"Current: best_frame_size:{best_frame_size}, best_amp_th:{best_amp_th}, best_percentile:{best_percentile}\n")

    print(f"Best hyperparameters for {set_name} set with best_avg_dist={best_avg_dist}:")
    print(f"best_frame_size:{best_frame_size}, best_amp_th:{best_amp_th}, best_percentile:{best_percentile}\n")

print("Done.")