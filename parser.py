import os
import librosa
import tqdm
import pandas as pd
import glob
from typing import Tuple
import argparse

sets = ["train", "val", "test"]

import torch
class Dataset(torch.utils.data.Dataset):
  def __init__(self, df):
        'Initialization'
        self.df = df

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        row = self.df.iloc[index]

        # Load data and get label
        X = row.audio
        y = row.text
        return X, y

  def get_sample_data(self, index):
      row = self.df.iloc[index]
      return row

def parse_arguments():
    """
    Parse arguments from the command line
    Returns:
        args:  Argument dictionary
               (Type: dict[str, str])
    """
    parser = argparse.ArgumentParser(description='ASR')
    parser.add_argument("-model_name", dest='model_name', action='store', type=str, default="DS")
    parser.add_argument("-augment", dest='augment', action='store', type=str, default="SpecAugment")
    parser.add_argument("-opt", dest='opt', action='store', type=str, default="AdamW")
    parser.add_argument("-audio_representation", dest='audio_representation', action='store', type=str, help="Mel/MFCC", default="Mel")
    parser.add_argument("-test", dest='test', action='store', type=str, default=None)

    parser.add_argument("-learning_rate", dest='learning_rate', action='store', type=float, default=5e-4)
    parser.add_argument("-batch_size", dest='batch_size', action='store', type=int, default=35)
    parser.add_argument("-epochs", dest='epochs', action='store', type=int, default=500)

    parser.add_argument("-sample_rate", dest='sample_rate', action='store', type=int, default=16000)
    parser.add_argument("-arc", dest='arc', action='store', type=str, default="RNN")
    parser.add_argument("-n_cnn_layers", dest='n_cnn_layers', action='store', type=int, default=3)
    parser.add_argument("-n_rnn_layers", dest='n_rnn_layers', action='store', type=int, default=5)
    parser.add_argument("-rnn_dim", dest='rnn_dim', action='store', type=int, default=512)
    parser.add_argument("-n_class", dest='n_class', action='store', type=int, default=29)
    parser.add_argument("-n_feats", dest='n_feats', action='store', type=int, default=128)
    parser.add_argument("-stride", dest='stride', action='store', type=int, default=2)
    parser.add_argument("-dropout", dest='dropout', action='store', type=float, default=0.1)

    parser.add_argument("-decoder", dest='decoder', action='store', type=str, default="greedy")

    # params for knn
    parser.add_argument("-k", dest='k', action='store', type=int, help="number of neighbors in knn", default=1)
    parser.add_argument("-dist",dest='dist',action='store', type=str, help='dtw/euclidean',default='dtw')

    v = vars(parser.parse_args())
    print(v)
    return v

def get_dataset_splits(sample_rate: int = 22050) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    This function loads the dataset splits(test,train,val) and return each split data as a dataframe.
    """

    dataset_split = {set:[] for set in sets}
    for set in sets:
        set_dir = f"./Data/{set}"
        text_dir = f"{set_dir}/an4/txt"
        wav_dir = f"{set_dir}/an4/wav"
        wav_paths = glob.glob(os.path.join(wav_dir, '*.wav'))
        wav_paths = sorted(wav_paths)
        for wav_path in tqdm.tqdm(wav_paths, f"parsing {set} data"):
            wav_name = os.path.basename(wav_path).split('.')[0]
            audio, _ = librosa.load(wav_path, sr=sample_rate)
            duration = librosa.get_duration(y=audio, sr=sample_rate)
            text_path = os.path.join(text_dir, f"{wav_name}.txt")
            with open(text_path) as f:
                text = f.readline()
            name, speaker, unknown = wav_name.split("-")

            dataset_split[set].append({
                             "wav_path": wav_path,
                             "wav_name": wav_name,
                             "audio": audio,
                             "duration": duration,
                             "text": text,
                             "speaker": speaker,})

    # convert to dataframe
    test_set = pd.DataFrame(dataset_split['test'])
    train_set = pd.DataFrame(dataset_split['train'])
    val_set = pd.DataFrame(dataset_split['val'])

    return test_set, train_set, val_set


