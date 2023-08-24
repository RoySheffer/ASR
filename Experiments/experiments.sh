#!/bin/bash
#SBATCH --mem=25g
#SBATCH -c20
#SBATCH --time=1-0

#python Experiments/find_best_silence_split_hyperparameters.py
#python models/KNN.py -k 1 -model_name knn -audio_representation MFCC -dist dtw
python models/KNN.py -model_name knn