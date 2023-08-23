import torch


def get_text_words_list(text):
    words = text.lower().split()
    return words
def dump_words(set_name, words):
    lines = [" ".join(([w] + list(w) + ["|"])) for w in words]
    with open(f'{set_name}_lexicon.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

def dump_lexicon():
    sample_rate = 16000
    from ASR.parser import get_dataset_splits
    test_set, train_set, val_set = get_dataset_splits(sample_rate)
    set2df = {"train": train_set, "val": val_set, "test": test_set}
    for set_name in set2df:
        df = set2df[set_name]['text'].map(lambda text: get_text_words_list(text)).explode()
        set_words = df.unique()
        dump_words(set_name, set_words)

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=28):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor):
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])
        return joined.replace("|", " ").strip().split()


class BEAM_Decoder(torch.nn.Module):
    def __init__(self, labels, use_lm=False, lm_path=None):
        super().__init__()
        if use_lm:
            if lm_path is not None:
                lm = lm_path
            else:
                torch_path = "/cs/dataset/Download/adiyoss/sheffer/CLAP/"
                import os
                if os.path.exists(torch_path):
                    os.environ['TRANSFORMERS_CACHE'] = torch_path
                torch_path = "/cs/dataset/Download/adiyoss/sheffer/passt"
                if os.path.exists(torch_path):
                    torch.hub.set_dir(torch_path)
                from torchaudio.models.decoder import download_pretrained_files
                files = download_pretrained_files("librispeech-4-gram")
                # print(f"files.lm {files.lm}")
                lm = files.lm
            # print(f"Using a language model - download_pretrained_files: {files}")
        else:
            lm = None

        LM_WEIGHT = 3.23
        WORD_SCORE = -0.26
        from torchaudio.models.decoder import ctc_decoder
        self.beam_search_decoder = ctc_decoder(
            tokens=labels,
            lexicon="train_lexicon.txt",
            nbest=3,
            lm=lm,
            beam_size=1500,
            lm_weight=LM_WEIGHT,
            word_score=WORD_SCORE,
        )
    def forward(self, emission: torch.Tensor):
        """Given a sequence emissions over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        return self.beam_search_decoder(emission)
    def transcript(self, emission: torch.Tensor):
        beam_search_result = self.forward(emission)
        transcripts = [" ".join(beam_search_result[i][0].words) for i in range(len(beam_search_result))]
        return transcripts


# from ctcdecode import CTCBeamDecoder
# model_path = None #  the path to your external kenlm language model(LM)
# alpha = 0 # alpha Weighting associated with the LMs probabilities. A weight of 0 means the LM has no effect
# beta = 0 # Weight associated with the number of words within our beam.
# cutoff_top_n = 1000 # Cutoff number in pruning. Only the top cutoff_top_n characters with the highest probability in the vocab will be used in beam search.
# cutoff_prob = 1.0 # Cutoff probability in pruning. 1.0 means no pruning.
# beam_width = 1000 # This controls how broad the beam search is. Higher values are more likely to find top beams but they also will make beam search exponentially slower.
# blank_id = 28 # This should be the index of the CTC blank token (probably 0).
# log_probs_input = False # If your outputs have passed through a softmax and represent probabilities, this should be false, if they passed through a LogSoftmax and represent negative log likelihood, you need to pass True. If you don't understand this, run print(output[0][0].sum()), if it's a negative number you've probably got NLL and need to pass True, if it sums to ~1.0 you should pass False. Default False.
# print(f"blank token is {labels[blank_id]}")
# decoder = CTCBeamDecoder(labels, model_path=None, alpha=alpha, beta=beta, cutoff_top_n=cutoff_top_n, cutoff_prob=cutoff_prob, beam_width=beam_width, num_processes=4, blank_id=blank_id, log_probs_input=log_probs_input)
#
# #Your output should be BATCHSIZE x N_TIMESTEPS x N_LABELS so you may need to transpose it before passing it to the decoder.
# beam_results, beam_scores, timesteps, out_lens = decoder.decode(output)
# # print(beam_results[0][0])
# # print(beam_results[0,0].cpu().detach().numpy())
# top_beam = beam_results[0][0][:out_lens[0, 0]].cpu().detach().numpy()
# print(f"top_beam {top_beam}")
#
# # text_beam_results = text_transform.int_to_text(beam_results[0, 0].cpu().detach().numpy())
# text_beam_results = text_transform.int_to_text(top_beam)
# # text_beam_results = [text_transform.int_to_text(r) for r in beam_results]
#
# print(f"text results {text_beam_results}")