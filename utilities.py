import numpy as np
import pandas as pd


def silence_split(signal, frame_size=800, amp_th=0.1, percentile=80):
    hop_length = frame_size - 1
    signal = np.abs(signal / np.abs(signal).max())
    # signal[signal < 0] = 0
    audio_splits = []
    prev_above_th = False
    start = 0
    end = 0
    for i in range(0, len(signal), hop_length):
        amplitude_envelope_current_frame = np.percentile(signal[i:i+frame_size],percentile)
        if amplitude_envelope_current_frame > amp_th and not prev_above_th:
            prev_above_th = True
            start = i
        if amplitude_envelope_current_frame > amp_th and prev_above_th:
            end = i + frame_size
        if amplitude_envelope_current_frame < amp_th and prev_above_th:
            prev_above_th = False
            audio_splits.append([start, end])

    return np.array(audio_splits)


def split_audio_to_words(df):
    word_level_list = []
    for index, row in df.iterrows():
        split_indexes = silence_split(row['audio'])
        words = row['text'].lower().strip().split(" ")
        min_seq_length = min(len(words), len(split_indexes))
        for i in range(min_seq_length):
            word_level_list.append({'word': words[i],
                                    'speaker': row['speaker'],
                                    'audio': row['audio'][split_indexes[i, 0]:split_indexes[i, 1]]})
    words2audio_df = pd.DataFrame(word_level_list)
    return words2audio_df
