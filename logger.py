import torch
import matplotlib.pyplot as plt
import pandas as pd
from ASR.parser import get_dataset_splits, Dataset
import numpy as np
def plot_metric(logger, key, ax, name=None):
    data = logger[key]
    X, Y = [d[0] for d in data], [d[1] for d in data]
    ax.plot(X, Y, label=name)
    if name == "Test":
        print(key, np.min(Y))
        ax.axhline(np.min(Y), color="gray", ls=":")
    if name:
        ax.legend(loc="best")
    return [1, np.min(Y)]
    # if name:
    #     yticks = list(plt.yticks()[0])
    #     j = np.argmin([np.abs(y - np.min(Y)) for y in yticks])
    #     yticks[j] = np.min(Y)
    #     plt.yticks(yticks)


class Logger:
    models_dir = "models/model_files"
    pred_log_num = 2
    def __init__(self, model_name="DS"):
        self.metrics = {}
        self.model_state = f"{self.models_dir}/{model_name}"
        self.logger_path = f"{self.models_dir}/{model_name}_logger"
        self.pred_html_path = f"{self.models_dir}/{model_name}_pred.html"

    def log_metric(self, metric_name, metric_value, step):
        item = (step, metric_value)
        if metric_name in self.metrics:
            self.metrics[metric_name].append(item)
        else:
            self.metrics[metric_name] = [item]

name2title = {
    # "DS": "Using AdamW optimizer",
    # "DS": "Using SpecAugment augmentation",
    "DS": "Using Mel-Spectrogram features",
    "DS_NoSpecAugment": "Without using augmentation",
    "DS_Adam": "Using Adam optimizer",
    "DS_SGD": "Using SGD optimizer",
    "DS_classicAugment": "Using classic augmentation",
    "DS_classicSpecAugment": "Using both classic and SpecAugment augmentation",
    "DS_MFCC": "Using MFCC features",
    "DS_MFCC_classicAugment": "Using MFCC features with classic augmentation",
    "DS_MFCC_SpecAugment": "Using MFCC features with SpecAugment augmentation",
    "DS_MFCC_classicSpecAugment": "Using both classic and SpecAugment augmentation",
}
fontsize = 20
if __name__ == "__main__":
    # models = ["DS", "DS_NoSpecAugment", "DS_Adam", "DS_SGD", "DS_classicAugment", "DS_classicSpecAugment", "DS_MFCC", "DS_MFCC_SpecAugment"]
    # models = ["DS", "Mel_SpecAugment_AdamW"]
    # models = list(name2title.keys())

    # name2title["DS"], models = "Using AdamW optimizer", ["DS", "DS_SGD", "DS_Adam"]
    # name2title["DS"], models = "Using SpecAugment augmentation", ["DS", "DS_classicAugment", "DS_classicSpecAugment", "DS_NoSpecAugment"]
    # name2title["DS"], models = "Using Mel-Spectrogram features", ["DS", "DS_MFCC_SpecAugment"]



    models = []
    # for REPRESENTATION in ["MFCC", "MEL"]:
    for REPRESENTATION in ["MFCC"]:
        # for AUGMENT in ["noAugment", "SpecAugment", "classic", "classicSpecAugment"]:
        for AUGMENT in ["SpecAugment"]:
            # for OPT in ["AdamW", "Adam", "SGD"]:
            for OPT in ["AdamW"]:
                for DECODER in ["greedy"]:
                # for DECODER in ["greedy", "beam", "beam_lm"]:
                    for ARC in ["RNN", "only_conv"]:
                    # for ARC in ["RNN"]:
                        model = f"{REPRESENTATION}_{AUGMENT}_{OPT}_{DECODER}_{ARC}"
                        models.append(model)
                        # model2 = f"{REPRESENTATION}_{AUGMENT}_{OPT}"
                        # model2 = f"{REPRESENTATION}_{AUGMENT}_{OPT}_{DECODER}_{ARC}2"
                        # models.append(model2)
    # models += ["DS_MFCC_SpecAugment", "DS_MFCC"]


    metrics = ["Train", "Validation", "Test"]
    fig, axes = plt.subplots(len(metrics), len(models), figsize=(10 * len(models), 8 * len(metrics)), tight_layout=True,sharex=True)




    for i, model_name in enumerate(models):

        logger = Logger(model_name=model_name)
        path = f"{Logger.models_dir}/{model_name}_logger"
        logger.metrics = torch.load(logger.logger_path)
        # print(logger.metrics.keys())
        print(model_name, "-------------------------")

        for j, metric in enumerate(["wer", "cer", "loss"]):
            ax = axes[j, i]
            yticks = []
            for set_name in metrics:
                key = f"{set_name}_{metric}"
                yticks += plot_metric(logger.metrics, key, ax, name=set_name)
            ax.set_title(f"{model_name} {metric}", fontsize=25)
            ax.set_yticks(yticks[:-1] + [np.max(ax.get_yticks())])

            twin_ax = ax.twinx()
            twin_ax.set_yticks([yticks[-1]])
            twin_ax.set_ylim(ax.get_ylim())

            ax.tick_params(axis='y', labelsize=fontsize)
            ax.tick_params(axis='x', labelsize=fontsize)
            twin_ax.tick_params(axis='y', labelsize=fontsize)
        # key = "learning_rate"
        # plot_metric(logger.metrics, key)
        # plt.title(f"{key}")
        # plt.show()

        test_set, train_set, val_set = get_dataset_splits()
        train_dataset = Dataset(train_set)
        val_dataset = Dataset(val_set)
        test_dataset = Dataset(test_set)

        set_name2dataset = {"Train": train_dataset, "Validation": val_dataset, "Test": test_dataset}
        df_dict = {}
        for m in range(logger.pred_log_num):
            g = {f"{set_name}_{set_name2dataset[set_name][m][1]}": [d[1] for d in logger.metrics[f"{set_name}_pred_{m}"]] for set_name in ["Train", "Validation", "Test"]}

            df_dict = {**df_dict, **g}
        df = pd.DataFrame.from_dict(df_dict)
        df.iloc[0::5, :].to_html(logger.pred_html_path)
    # plt.show()

    fig.savefig("stats.png")



