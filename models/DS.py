import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np
import sys
sys.path.append("../")
from ASR.parser import get_dataset_splits, Dataset, parse_arguments
from jiwer import wer, cer
import warnings
from ASR.logger import Logger
from ASR.augmentation import augment
from ASR.Decoder import BEAM_Decoder
warnings.filterwarnings('ignore') # setting ignore as a parameter

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')

def data_processing(data, hparams, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for (waveform, utterance) in data:
        if "classic" in hparams["augment"]:
            waveform = augment(waveform, hparams["sample_rate"]).astype(np.float32)

        waveform = torch.from_numpy(waveform[None, :])
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif data_type == 'valid':
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            raise Exception('data_type should be train or valid')
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    return spectrograms, labels, input_lengths, label_lengths


def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    if decoder is not None:
        decodes = decoder.transcript(output.cpu().detach())
    return decodes, targets

class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # torch.Size([35, 1, 13, 598])
        # torch.Size([35, 32, 299, 7])
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time)

class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)

class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x

class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1, only_conv=False):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features
        self.only_conv = only_conv

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        if self.only_conv:
            self.fully_connected = nn.Linear(n_feats*32, rnn_dim * 2)
        else:
            self.fully_connected = nn.Linear(n_feats * 32, rnn_dim)
            self.birnn_layers = nn.Sequential(*[
                BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                                 hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
                for i in range(n_rnn_layers)
            ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        if not self.only_conv:
            x = self.birnn_layers(x)
        x = self.classifier(x)
        return x

class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, logger):
    model.train()
    data_len = len(train_loader.dataset)
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        logger.log_metric('loss', loss.item(), step=iter_meter.get())
        logger.log_metric('learning_rate', scheduler.get_lr(), step=iter_meter.get())

        optimizer.step()
        scheduler.step()
        iter_meter.step()
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(spectrograms), data_len, 100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, criterion, epoch, iter_meter, logger, set_name="", hparams=None):
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    decoded_targets_list, decoded_preds_list = [], []

    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
            # text_transform.int_to_text((int(l) for l in labels[0]))
            decoded_targets_list += decoded_targets
            decoded_preds_list += decoded_preds

            for j in range(len(decoded_preds)):
                # print(decoded_targets[j], "   -   ", decoded_preds[j], wer(decoded_targets[j], decoded_preds[j]))
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)
    logger.log_metric(f'{set_name}_loss', test_loss, step=iter_meter.get())
    logger.log_metric(f'{set_name}_cer', avg_cer, step=iter_meter.get())
    logger.log_metric(f'{set_name}_wer', avg_wer, step=iter_meter.get())
    for m in range(logger.pred_log_num):
        logger.log_metric(f'{set_name}_pred_{m}', decoded_preds_list[m], step=iter_meter.get())
        field_name = f'{set_name}_GT_{m}'
        if field_name not in logger.metrics:
            logger.log_metric(field_name, decoded_targets_list[m], step=iter_meter.get())

    print(f'{set_name} set: Average loss: {test_loss}, Average CER: {avg_cer} Average WER: {avg_wer}  ')
    print(f"'{decoded_targets_list[0]}'  -  '{decoded_preds_list[0]}'")
    return test_loss, avg_wer, avg_cer

class MEL2MFCC(torch.nn.Module):
    def __init__(
            self,
            n_mels: int = 128,
            n_mfcc: int = 40,
            norm: str = "ortho",
    ) -> None:
        super(MEL2MFCC, self).__init__()
        from torchaudio import functional as F2
        self.n_mfcc = n_mfcc
        self.norm = norm
        self.top_db = 80.0
        self.amplitude_to_DB = torchaudio.transforms.AmplitudeToDB("power", self.top_db)
        self.dct_mat = F2.create_dct(n_mfcc, n_mels, norm)

    def forward(self, mel_specgram):
        mel_specgram = self.amplitude_to_DB(mel_specgram)
        mfcc = torch.matmul(mel_specgram.transpose(-1, -2), self.dct_mat).transpose(-1, -2)
        return mfcc

def main(hparams, logger=None):
    use_cuda = torch.cuda.is_available()
    seed = 7
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if use_cuda else "cpu")

    test_set, train_set, val_set = get_dataset_splits(sample_rate=hparams["sample_rate"])
    train_dataset = Dataset(train_set)
    val_dataset = Dataset(val_set)
    test_dataset = Dataset(test_set)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x, hparams, 'train'),
                                **kwargs)
    train_eval_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, hparams, 'valid'),
                                **kwargs)
    val_loader = data.DataLoader(dataset=val_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, hparams, 'valid'),
                                **kwargs)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, hparams, 'valid'),
                                **kwargs)

    model = SpeechRecognitionModel(hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout'], only_conv=(hparams['arc'] == "only_conv")
        ).to(device)


    load_prev_model = False
    if hparams['test']:
        load_prev_model = True

    os.makedirs(logger.models_dir, exist_ok=True)
    if load_prev_model:
        if os.path.exists(logger.models_dir):

            iference_model_name = f'{hparams["audio_representation"]}_{hparams["augment"]}_{hparams["opt"]}_greedy_{hparams["arc"]}_test'
            iference_model_state = f"{logger.models_dir}/{iference_model_name}"

            # state_path = logger.model_state
            state_path = iference_model_state
            print(f"Loading model state from {state_path} model_state: {logger.model_state}")
            model.load_state_dict(torch.load(state_path, map_location=torch.device(device)))




    # print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))
    print('using device', device)

    if hparams["opt"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    elif hparams["opt"] == "Adam":
        optimizer = optim.Adam(model.parameters(), hparams['learning_rate'])
    elif hparams["opt"] == "SGD":
        optimizer = optim.SGD(model.parameters(), hparams['learning_rate'])


    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], steps_per_epoch=int(len(train_loader)),
                                            epochs=hparams['epochs'],
                                            anneal_strategy='linear')

    criterion = nn.CTCLoss(blank=28).to(device)
    iter_meter = IterMeter()
    sys.stdout.flush()

    if hparams['test']:
        epoch = -1
        test_loss, test_wer, test_cer = test(model, device, test_loader, criterion, epoch, iter_meter, logger, set_name="Test", hparams=hparams)
        sys.stdout.flush()
        exit()
        validation_loss, validation_wer, validation_cer = test(model, device, val_loader, criterion, epoch, iter_meter, logger, set_name="Validation", hparams=hparams)
        sys.stdout.flush()
        train_loss, train_wer, train_cer = test(model, device, train_eval_loader, criterion, epoch, iter_meter, logger, set_name="Train", hparams=hparams)
        sys.stdout.flush()
        torch.save(logger.metrics, f"{logger.logger_path}_test")
    min_wer = {}
    min_wer["train"], min_wer["validation"], min_wer["test"] = None, None, None
    for epoch in range(1, hparams["epochs"] + 1):
        print(f"epoch {epoch} --------------------------------------------------------------")
        train_loss, train_wer, train_cer = test(model, device, train_eval_loader, criterion, epoch, iter_meter, logger, set_name="Train", hparams=hparams)
        validation_loss, validation_wer, validation_cer = test(model, device, val_loader, criterion, epoch, iter_meter, logger, set_name="Validation", hparams=hparams)
        test_loss, test_wer, test_cer = test(model, device, test_loader, criterion, epoch, iter_meter, logger, set_name="Test", hparams=hparams)

        cur_wer = {}
        cur_wer["train"], cur_wer["validation"], cur_wer["test"] = train_wer, validation_wer, test_wer
        for set_name in ["train", "validation", "test"]:
            if min_wer[set_name] is None or cur_wer[set_name] < min_wer[set_name]:
                print(f"{cur_wer[set_name]} < {min_wer[set_name]} -> saving new best {set_name} model")
                min_wer[set_name] = cur_wer[set_name]
                torch.save(model.state_dict(), f"{logger.model_state}_{set_name}")
        torch.save(model.state_dict(), logger.model_state)

        train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, logger)
        torch.save(logger.metrics, logger.logger_path)
        sys.stdout.flush()


hparams = parse_arguments()
hparams["model_name"] = f'{hparams["audio_representation"]}_{hparams["augment"]}_{hparams["opt"]}_{hparams["decoder"]}_{hparams["arc"]}'
print(hparams["model_name"])
logger = Logger(model_name=hparams["model_name"])

# PitchShift_transform = torchaudio.transforms.PitchShift(hparams["sample_rate"], 4)
# SPEED_transform = torchaudio.transforms.SpeedPerturbation(orig_freq=hparams["sample_rate"], factors =[0.9, 1.1, 1.0, 1.0, 1.0])

# n_mels, n_mfcc =128, 13
# MEL2MFCC_transform = MEL2MFCC(n_mels=n_mels, n_mfcc=n_mfcc)

MFCC_transform = torchaudio.transforms.MFCC(sample_rate=hparams["sample_rate"], n_mfcc=13,  melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23}, )
MelSpectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=hparams["sample_rate"], n_mels=128)
if hparams["audio_representation"] == "Mel":
    audio_representation_transform = MelSpectrogram_transform
elif hparams["audio_representation"] == "MFCC":
    audio_representation_transform = MFCC_transform
    hparams["n_feats"] = 14

if "SpecAugment" in hparams["augment"]:
    train_audio_transforms = nn.Sequential(
        audio_representation_transform,
        torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
        torchaudio.transforms.TimeMasking(time_mask_param=100)
    )
else:
    train_audio_transforms = audio_representation_transform

valid_audio_transforms = audio_representation_transform
text_transform = TextTransform()
if "beam" in hparams["decoder"]:
    labels_beam = list(text_transform.int_to_text(range(len(text_transform.index_map.keys())))) + ["-"]
    labels_beam[1] = "|"
    print(f"found {len(labels_beam)} labels - {labels_beam}")
    lm_path =None
    if "trained" in hparams["decoder"]:
        lm_path = 'train_5_gram_data_new.bin'
    decoder = BEAM_Decoder(labels=labels_beam, use_lm=("lm" in hparams["decoder"]), lm_path=lm_path)
else:
    decoder = None
if __name__ == "__main__":
    main(hparams, logger)

#-audio_representation MFCC -augment SpecAugment -opt AdamW -decoder greedy -arc RNN -test test