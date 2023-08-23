import sys
sys.path.append("../")
import numpy as np
from ASR.models.DS import TextTransform, hparams, SpeechRecognitionModel, valid_audio_transforms, GreedyDecoder, MFCC_transform,MelSpectrogram_transform
from ASR.logger import Logger
import torch
import librosa
import torch.nn as nn
import torch.nn.functional as F
from ASR.Decoder import GreedyCTCDecoder, BEAM_Decoder
from ASR.parser import get_dataset_splits, parse_arguments
from jiwer import wer, cer

torch.manual_seed(7)
text_transform = TextTransform()
use_cuda = False #torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# model_name = "Mel_SpecAugment_Adam"
model_name = "MFCC_SpecAugment_AdamW_greedy_RNN"
hparams['n_feats'] = 14
audio_transforms = MFCC_transform
logger = Logger(model_name=model_name)
model = SpeechRecognitionModel(hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'], hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']).to(device)
model.load_state_dict(torch.load(logger.model_state, map_location=torch.device(device)))

wav_path = r"Data/test/an4/wav/an391-mjwl-b.wav"
# wav_path =  r"C:\Users\roysh\OneDrive - Mobileye\Documents\Sound recordings\3.wav"
waveform, _ = librosa.load(wav_path, sr=hparams["sample_rate"])
waveform = torch.from_numpy(waveform[None, :])
waveforms = [waveform]

spectrograms = [audio_transforms(wf).squeeze(0).transpose(0, 1) for wf in waveforms]
spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
output = model(spectrograms)  # (batch, time, n_class)
output = F.log_softmax(output, dim=2)

labels, label_lengths = torch.from_numpy(np.array([[1, 2], [1, 2]])), [2, 2]
decoded_preds, decoded_targets = GreedyDecoder(output, labels, label_lengths)
print(f"GreedyDecoder got {decoded_preds}")

labels = list(text_transform.int_to_text(range(len(text_transform.index_map.keys())))) + ["-"]
labels[1] = "|"

beam_decoder_lexicon = BEAM_Decoder(labels=labels, use_lm=False)
beam_decoder_lexicon_transcript = beam_decoder_lexicon.transcript(output)
print(f"beam_decoder_lexicon_transcript  {beam_decoder_lexicon_transcript}")

beam_decoder_an4 = BEAM_Decoder(labels=labels, use_lm=True, lm_path='train_5_gram_data_new.bin')
beam_decoder_an4_transcript = beam_decoder_an4.transcript(output)
print(f"beam_decoder_an4_transcript  {beam_decoder_an4_transcript}")

beam_decoder_librispeech = BEAM_Decoder(labels=labels, use_lm=True)
beam_decoder_librispeech_transcript = beam_decoder_librispeech.transcript(output)
print(f"beam_decoder_librispeech_transcript  {beam_decoder_librispeech_transcript}")

# greedy_decoder = GreedyCTCDecoder(labels)
# greedy_result = greedy_decoder(output[0])
# greedy_transcript = " ".join(greedy_result)
# print(f"greedy_transcript is {greedy_transcript}")

