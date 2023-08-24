import sys
sys.path.append("../")
import librosa
import torch
import torchaudio
import numpy as np
import pandas as pd
import tqdm
from ASR.utilities import split_audio_to_words, silence_split
from ASR.parser import get_dataset_splits, parse_arguments
from jiwer import wer, cer
import concurrent.futures
import multiprocessing
from collections import Counter
pd.options.mode.chained_assignment = None  # default='warn'


def euclidean_dist(x, y):
    seq1, seq2 = torch.tensor(x.T), torch.tensor(y.T)
    seq1, seq2 = torch.nn.utils.rnn.pad_sequence([seq1, seq2], batch_first=True)
    return torch.linalg.vector_norm(seq1 - seq2)


def cosine_similarity(x,y):
    seq1, seq2 = torch.tensor(x), torch.tensor(y)
    seq1, seq2 = torch.nn.utils.rnn.pad_sequence([seq1, seq2], batch_first=True)
    seq1_norm = seq1 / torch.linalg.vector_norm(seq1)
    seq2_norm = seq2 / torch.linalg.vector_norm(seq2)
    return 1 - seq1_norm.flatten() @ seq2_norm.flatten()


def vlt_normalization_test_set(out_df:pd.DataFrame):

    # perform vlt normalization
    for speaker in out_df['speaker'].unique():
        speaker_df = out_df[out_df['speaker'] == speaker]
        speaker_words_list = speaker_df['words_features'].explode().tolist()
        # pad coefficients

        if len(speaker_words_list) != 1:
            tensor_list = [torch.tensor(element.T) for element in speaker_words_list]
            # features_padded = torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True)
            # features_padded = features_padded.permute(0, 2, 1)
            # perform normalization
            # features_mean = torch.mean(features_padded, dim=0)
            # normalized_features = features_padded - features_mean
            features_mean = torch.mean(torch.concat(tensor_list), dim=0)
            normalized_features = [tensor_list[i] - features_mean for i in range(len(tensor_list))]

            # store normalized features
            lower_idx = 0
            upper_idx = 0
            speaker_df['normalized_features'] = None
            for index, value in enumerate(speaker_df['words_features']):
                upper_idx += len(value)
                #speaker_df['normalized_features'].iloc[index] = list(normalized_features[lower_idx:upper_idx])
                speaker_df['normalized_features'].iloc[index] = normalized_features[lower_idx:upper_idx]

                lower_idx = upper_idx
            out_df[out_df['speaker'] == speaker] = speaker_df
        else:
            speaker_df['normalized_features'].iloc[0] = [torch.tensor(speaker_words_list[0])]
            out_df[out_df['speaker'] == speaker] = speaker_df

    return out_df

def features_normalization_test_set(out_df:pd.DataFrame):
    for index, words_features in enumerate(out_df['words_features']):
        tensor_list = [torch.tensor(element) for element in words_features]
        tensor_list_normalized = [word_feature - torch.mean(word_feature, dim=0) for word_feature in tensor_list]
        out_df['normalized_features'].iloc[index] = tensor_list_normalized
    return out_df


def vlt_normalization(set_df:pd.DataFrame, sample_rate):
    out_df = set_df.copy(deep=True)
    out_df['features'] = None
    out_df['normalized_features'] = None
    for speaker in set_df['speaker'].unique():
        speaker_df = set_df[set_df['speaker'] == speaker]
        # get mfcc coefficients
        speaker_df.loc[:,('features')] = speaker_df.loc[:,('audio')].map(lambda audio: librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20, n_fft=512,
                                                                                                            win_length=400, hop_length=160, n_mels=23))
        # pad coefficients
        tensor_list = [torch.tensor(element.T) for element in speaker_df['features']]
        # features_padded = torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True)
        # features_padded = features_padded.permute(0, 2, 1)
        features_mean = torch.mean(torch.concat(tensor_list), dim=0)
        normalized_features = [tensor_list[i] - features_mean for i in range(len(tensor_list))]
        # perform normalization
        #normalized_features = features_padded - torch.mean(features_padded, dim=0)
        #speaker_df.loc[:,'normalized_features'] = list(normalized_features[:])
        speaker_df.loc[:, 'normalized_features'] = normalized_features
        out_df[out_df['speaker'] == speaker] = speaker_df
    return out_df


def features_normalization(set_df:pd.DataFrame, sample_rate):
    # get mfcc coefficients
    MFCC_transform = torchaudio.transforms.MFCC(sample_rate=hparams["sample_rate"], n_mfcc=20,
                                                melkwargs={"n_fft": 512, "win_length": 400, "hop_length": 160, "n_mels": 23}, )
    set_df['features'] = set_df['audio'].map(lambda audio:  MFCC_transform(torch.tensor(audio)))
    set_df['normalized_features'] = set_df['features'].map(lambda features:features - torch.mean(features,dim=0))
    return set_df


def preprocess(set_df:pd.DataFrame,normalization_type, sample_rate=16000):
    if normalization_type == "speaker_normalization":
        set_df = vlt_normalization(set_df, sample_rate)
    elif normalization_type == "features_normalization":
        set_df = features_normalization(set_df,sample_rate)
    return set_df


def preprocess_test(set_df:pd.DataFrame,normalization_type, sample_rate=16000):
    out_df = set_df.copy(deep=True)
    out_df['words_features'] = None
    out_df['normalized_features'] = None
    # get mfcc features list per word
    for index, row in set_df.iterrows():
        split_indexes = silence_split(row['audio'])
        features_list = []
        for idxs in split_indexes:
            w = row['audio'][idxs[0]:idxs[1]]
            w_features = librosa.feature.mfcc(y=w, sr=sample_rate, n_mfcc=20 ,n_fft=512, win_length=400, hop_length=160, n_mels=23)
            features_list.append(w_features)
        out_df['words_features'].iloc[index] = features_list

    if normalization_type == "speaker_normalization":
        out_df = vlt_normalization_test_set(out_df)
    elif normalization_type == "features_normalization":
        out_df = features_normalization_test_set(out_df)

    return out_df


class KNNModel:
    def __init__(self, dist_function, audio_representation, k=1, model_name="", is_preprocessed=False, random_neighbor=False):
        #print(f"Running model {model_name} with:[\nk={k}\n,dist_function={dist_function}\n,audio_representation={audio_representation}]:")
        self.k = k
        self.X_train = None
        self.y_train = None
        self.random_neighbor = random_neighbor
        if dist_function == 'dtw':
            self.dist_func = lambda x, y: librosa.sequence.dtw(x, y)[0][-1, -1]
        elif dist_function == 'euclidean':
            self.dist_func = euclidean_dist
        elif dist_function == 'cosine_similarity':
            self.dist_func = cosine_similarity
        else:
            self.dist_func = None

        self.is_preprocessed = is_preprocessed
        if not self.is_preprocessed:
            MFCC_transform = torchaudio.transforms.MFCC(sample_rate=hparams["sample_rate"], n_mfcc=20, melkwargs={"n_fft": 512,"win_length":400, "hop_length": 160, "n_mels": 23}, )
            MelSpectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=hparams["sample_rate"], n_mels=23)
            if audio_representation == "Mel":
                self.audio_representation_transform = MelSpectrogram_transform
            elif audio_representation == "MFCC":
                self.audio_representation_transform = MFCC_transform
            else:
                self.audio_representation_transform = None
        else:
            self.audio_representation_transform = None

    def fit(self,X_train,y_train):
        if not self.is_preprocessed:
            self.X_train = X_train.map(lambda audio: self.audio_representation_transform(torch.tensor(audio)))
        else:
            self.X_train = X_train
        self.y_train = y_train

    def compute_dists(self,train_feat,cur_word_features):
        if train_feat.shape[1] == cur_word_features.shape[1]:
            return self.dist_func(train_feat.T.numpy(), cur_word_features.T.numpy()).item()
        else:
            return self.dist_func(train_feat.T.numpy(), cur_word_features.numpy()).item()

    def predict(self, X_test, split_name):
        predictions = []
        if not self.random_neighbor:
            if not self.is_preprocessed:
                for index ,audio in tqdm.tqdm(enumerate(X_test), f"Prediction on {split_name} set..."):
                    cur_prediction = ''
                    split_indexes = silence_split(audio)
                    for idxs in split_indexes:
                        # get 'word' features
                        cur_word_audio = audio[idxs[0]:idxs[1]]
                        # cur_word_features = librosa.feature.mfcc(y=cur_word_audio,sr=hparams['sample_rate'])
                        cur_word_features = self.audio_representation_transform(torch.tensor(cur_word_audio))

                        # find best neighbor according to voting and distance
                        dists = np.vectorize(lambda train_feat: self.dist_func(train_feat.numpy(), cur_word_features.numpy()).item())(self.X_train)
                        best_k_indices = np.argpartition(dists,self.k)[:self.k]
                        neighbors2dist = dict(zip(self.y_train.iloc[best_k_indices],dists[best_k_indices]))
                        neighbors2count = Counter([self.y_train.iloc[idx] for idx in best_k_indices])
                        combined_dict = {key: (1 / neighbors2count[key], neighbors2dist[key]) for key in neighbors2dist}
                        best_neighbor = min(combined_dict, key=lambda key: combined_dict[key])
                        cur_prediction += f"{best_neighbor} "

                    predictions.append(cur_prediction)
            else:
                for index, words_features in tqdm.tqdm(enumerate(X_test), f"Prediction on {split_name} set..."):
                    cur_prediction = ''
                    for cur_word_features in words_features:
                        # find best neighbor according to voting and distance
                        dists = np.vectorize(lambda train_feat: self.compute_dists(train_feat, cur_word_features))(self.X_train)
                        best_k_indices = np.argpartition(dists,self.k)[:self.k]
                        neighbors2dist = dict(zip(self.y_train.iloc[best_k_indices],dists[best_k_indices]))
                        neighbors2count = Counter([self.y_train.iloc[idx] for idx in best_k_indices])
                        combined_dict = {key: (1 / neighbors2count[key], neighbors2dist[key]) for key in neighbors2dist}
                        best_neighbor = min(combined_dict, key=lambda key: combined_dict[key])

                        cur_prediction += f"{best_neighbor} "
                    predictions.append(cur_prediction)
        else:
            num_test_samples = len(X_test)
            for audio in tqdm.tqdm(X_test, f"Prediction on {split_name} set..."):
                cur_prediction = ''
                split_indexes = silence_split(audio)
                num_of_words = len(split_indexes)
                prediction_idxs = np.random.randint(0, num_test_samples, num_of_words)
                predictions_list = self.y_train.iloc[prediction_idxs]
                cur_prediction = " ".join(predictions_list)
                predictions.append(cur_prediction)

        return predictions


def main(hparams, preprocess_type = None):
    # load splits
    test_set, train_set, val_set = get_dataset_splits(hparams["sample_rate"])
    #val_test_dict = {'val':val_set, 'test': test_set}
    val_test_dict = {'val': val_set} # todo comment

    # split each audio to words
    train_df = split_audio_to_words(train_set)
    # create knn model
    knn_model = KNNModel(hparams['dist'], hparams['audio_representation'], k=hparams['k'], model_name=hparams["model_name"],
                         is_preprocessed=preprocess_type is not None)
    if preprocess_type is not None:
        train_df = preprocess(train_df, preprocess_type)
        knn_model.fit(X_train=train_df['normalized_features'], y_train=train_df['word'])
    else:
        knn_model.fit(X_train=train_df['audio'], y_train=  train_df['word'])


    # validate
    metrics_dict = {}
    for set_name in val_test_dict.keys():
        set_df = val_test_dict[set_name]
        if preprocess_type is not None:
            set_df = preprocess_test(set_df, preprocess_type)
            predictions = knn_model.predict(set_df['normalized_features'],split_name=set_name)
        else:
            predictions = knn_model.predict(set_df['audio'], split_name=set_name)

        # compute scores
        wer_sum = 0
        cer_sum = 0
        for index, pred in tqdm.tqdm(enumerate(predictions),f"Computing avg WER and CER on {set_name} set..."):
            ref = val_test_dict[set_name]['text'].iloc[index].lower().strip()
            wer_sum += wer(ref, pred)
            cer_sum += cer(ref, pred)

        avg_wer = wer_sum / len(val_test_dict[set_name])
        avg_cer = cer_sum / len(val_test_dict[set_name])
        print(f"avg WER on {set_name} set:{avg_wer}")
        print(f"avg CER on {set_name} set:{avg_cer}")
        metrics_dict[f"{set_name}_wer"] = avg_wer
        metrics_dict[f"{set_name}_cer"] = avg_cer

    return metrics_dict

def process_test_string(params):
    k, feature_type, dist_metric, preprocess_type = params
    test_string = f"(k={k},features={feature_type},dist={dist_metric},preprocess={preprocess_type})"
    print(f"Testing configuration: {test_string}")
    hparams['audio_representation'] = feature_type
    hparams['dist'] = dist_metric
    hparams['k'] = k
    metrics_dict = main(hparams, preprocess_type)
    return test_string, metrics_dict
#
# if __name__ == '__main__':
#     # get user args
#     hparams = parse_arguments()
#     num_cores = multiprocessing.cpu_count()
#     results_dict = {}
#     k_range = [1, 3, 5, 11, 15, 21]
#     features_types = ["Mel", "MFCC"]
#     dist_metrics = ["dtw", "euclidean"]
#     preprocess_options = ["speaker_normalization", None]
#     params_list = [(k, feature_type, dist_metric, preprocess_type)
#                    for k in k_range
#                    for feature_type in features_types
#                    for dist_metric in dist_metrics
#                    for preprocess_type in preprocess_options]
#
#     # Use ProcessPoolExecutor for parallel execution
#     with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
#         # Submit tasks to the executor for each set of parameters
#         futures = [executor.submit(process_test_string, params) for params in params_list]
#
#         # Wait for all tasks to complete and gather the results
#         for future in concurrent.futures.as_completed(futures):
#             test_string, metrics_dict = future.result()
#             results_dict[test_string] = metrics_dict
#
#     results_df = pd.DataFrame(results_dict).T
#     print(f"results_df:\n{results_df.to_string()}")
#     results_df.to_csv("results_df.csv")


#
if __name__ == "__main__":
    # get user args
    hparams = parse_arguments()
    num_cores = multiprocessing.cpu_count()
    results_dict = {}
    k_range = [1,3,5,11,15,21]
#    features_types = ["Mel", "MFCC"]
    features_types = ["MFCC"]
    # dist_metrics = ["dtw", "euclidean"]
    dist_metrics = ["dtw"]
    # preprocess_options = ["features_normalization","speaker_normalization", None]
    preprocess_options = ["speaker_normalization"]
    for k in k_range:
        for feature_type in features_types:
            for dist_metric in dist_metrics:
                for preprocess_type in preprocess_options:
                    test_string = f"(k={k},features={feature_type},dist={dist_metric},preprocess={preprocess_type})"
                    print(f"Testing configuration:{test_string}")
                    hparams['audio_representation'] = feature_type
                    hparams['dist'] = dist_metric
                    hparams['k'] = k
                    #hparams['random_neighbor'] = True
                    metrics_dict = main(hparams, preprocess_type)
                    results_dict[test_string] = metrics_dict

    # results_df = pd.DataFrame(results_dict).T
    # print(f"results_df:\n{results_df.to_string()}")
    # results_df.to_csv("results_df.csv")

    results_df = pd.DataFrame(results_dict).T
    print(f"speaker_norm_val_results_df:\n{results_df.to_string()}")
    results_df.to_csv("speaker_norm_val_results_df.csv")