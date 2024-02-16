import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from math import exp, floor
import matplotlib.pyplot as plt
from dataset import CountSubset, NonCountSubset, split
import clip

import torch
import torch.nn as nn
import torch.distributed
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

loss_total_val = {}
loss_total_train = {}

loss_clip_val = {}
loss_clip_train = {}

loss_count_val = {}
loss_count_train = {}

def ddp_setup():
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.cuda.empty_cache()


class ModelArgs:
    version: str


class TrainArgs:
    epochs: int
    precision: str
    no_save: bool
    accumulate_steps: int
    train_size: float
    batch_size: int
    eval_every: int
    learning_rate: float
    optimizer: str
    weight_decay: float
    path: str
    init: bool
    clip: bool
    c_ratio: float
    lambda_: float = 0.5


class Experiment:

    def __init__(self, model_args: ModelArgs, train_args: TrainArgs):
        """
        Takes (almost) all parameters as command line arguments and runs the experiment.

        Params
        ------
        args
            argparse object containing all arguments.
        clip
            If True, then only run on 64 samples. **(overfitting test)**
        """
        # Load data
        self.count_data = np.load("segments0.npy", allow_pickle=True)
        self.noncount_data = np.load("segments1.npy", allow_pickle=True) #FIXME: Change this to segments1.npy
        # dataframe = dataframe # reduce size for faster training
        if train_args.clip:
            dataframe = dataframe[:50]

        # Dataset
        train_count_data, val_count_data = split(self.count_data, train_args.train_size)
        train_noncount_data, val_noncount_data = split(self.noncount_data, train_args.train_size)

        # Device
        device = int(os.environ["LOCAL_RANK"])
        self.device = device

        train_count_data, val_count_data = CountSubset(train_count_data), CountSubset(val_count_data)
        train_noncount_data, val_noncount_data = NonCountSubset(train_noncount_data), NonCountSubset(val_noncount_data)

        # Instantiate model
        self.model, self.preprocess = clip.load(model_args.version)
        assert type(self.model) == nn.Module, "Model is not a nn.Module."

        train_count_data_loader = DataLoader(
            train_count_data,
            batch_size=train_args.batch_size/train_args.c_ratio,
            shuffle=False,
            sampler=DistributedSampler(train_count_data),
            pin_memory=True,
        )
        val_count_data_loader = DataLoader(
            val_count_data,
            batch_size=train_args.batch_size/train_args.c_ratio,
            shuffle=False,
            sampler=DistributedSampler(val_count_data),
            pin_memory=True,
        )
        train_noncount_data_loader = DataLoader(
            train_noncount_data,
            batch_size=train_args.batch_size,
            shuffle=False,
            sampler=DistributedSampler(train_noncount_data),
            pin_memory=True,
        )
        val_noncount_data_loader = DataLoader(
            val_noncount_data,
            batch_size=train_args.batch_size,
            shuffle=False,
            sampler=DistributedSampler(val_noncount_data),
            pin_memory=True,
        )

        self.train_count_data, self.val_count_data = train_count_data_loader, val_count_data_loader
        self.train_noncount_data, self.val_noncount_data = train_noncount_data_loader, val_noncount_data_loader

        # Model
        if train_args.precision == "bf16":
            self.model = self.model.bfloat16()

        # Print total number of parameters and trainable parameters.
        total_params = sum(p.numel() for p in self.model.parameters())
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total number of parameters: {total_params/1e6 : .1f}M")
        print(
            f"Total number of trainable parameters: {total_trainable_params/1e6 : .1f}M"
        )

        if train_args.optimizer == "SGD":
            optimizer = optim.SGD(
                self.model.parameters(),
                train_args.learning_rate,
                momentum=0.98,
                weight_decay=train_args.weight_decay,
            )
        elif train_args.optimizer == "Adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                train_args.learning_rate,
                weight_decay=train_args.weight_decay,
            )
        elif train_args.optimizer == "AdamW":
            optimizer = optim.AdamW(
                self.model.parameters(),
                train_args.learning_rate,
                weight_decay=train_args.weight_decay,
            )
        else:
            raise NotImplementedError
        self.optimizer = optimizer

        # Dataset Assignments
        self.batch_size = train_args.batch_size
        self.accumulate_steps = train_args.accumulate_steps
        self.version = model_args.version
        self.eval_every = train_args.eval_every
        self.precision = train_args.precision
        self.weight_decay = train_args.weight_decay
        self.lr = train_args.learning_rate
        self.snapshot_path = train_args.path
        self.dataframe = dataframe
        self.device = device
        self.model = self.model.to(self.device)
        self.no_save = train_args.no_save
        self.c_ratio = train_args.c_ratio
        self.lambda_ = train_args.lambda_
        self.epochs_run = 0

        # For save-decisions
        self.cur_min_loss = float("inf")

        global SAVE_PATH
        SAVE_PATH = f"c_ratio_{train_args.c_ratio}_epochs_{train_args.epochs}_lr_{train_args.learning_rate}_bs_{train_args.batch_size}_version_{model_args.version}"

        if (
            os.path.exists(self.snapshot_path)
            or os.path.exists("intermediate/" + SAVE_PATH + ".pth")
        ) and train_args.init == False:  # To resume training in case of a failure
            print("Loading snapshot...")
            temp_path = self.snapshot_path
            if os.path.exists(
                "intermediate/" + SAVE_PATH + ".pth"
            ) and not os.path.exists(self.snapshot_path):
                self.snapshot_path = "intermediate/" + SAVE_PATH + ".pth"
                print("Intermediate weights found!", flush=True)
            self._load_snapshot(self.snapshot_path)
            self.snapshot_path = temp_path

        self.model = DDP(self.model, device_ids=[self.device])

    def _load_snapshot(self, snapshot_path):

        loc = f"cuda:{self.device}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.cur_min_loss = snapshot["MIN_LOSS"]
        assert (
            self.lambda_ == snapshot["LAMBDA_"]
        ), "This snapshot cannot be used with this lambda. Snapshot has {} and you have {}.".format(
            snapshot["LAMBDA_"], self.lambda_
        )
        assert (
            self.c_ratio == snapshot["C_RATIO"]
        ), "This snapshot cannot be used with this c_ratio. Snapshot has {} and you have {}.".format(
            snapshot["C_RATIO"], self.c_ratio
        )
        assert (
            self.version == snapshot["VERSION"]
        ), "This snapshot cannot be used with this version. Snapshot has {} and you have {}.".format(
            snapshot["VERSION"], self.version
        )
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
        return

    def _save_snapshot(self, epoch):

        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "MIN_LOSS": self.cur_min_loss,
            "VERSION": self.version,
        }
        if self.no_save:
            print(f"Epoch {epoch} | Training snapshot not saved.")
            return
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")
        return

    def _save_checkpoint(self, epoch, path=None):
        global SAVE_PATH
        # ckp = self.model.module.state_dict()
        if path == None:
            path = "chkpts/" + SAVE_PATH + ".pth"
        if (
            self.device == 0
        ):  # NOTE : running _evaluate on a single GPU lead to synchoronization issues so I ran it on both GPUs but save only from one.
            snapshot_path = self.snapshot_path
            self.snapshot_path = path
            self._save_snapshot(epoch)
            self.snapshot_path = snapshot_path
        return

    def _run_batch(self, count_imgs, count_texts, counterfactual_count_text, non_count_images, non_count_text, i):

        if i == 10:
            start = time.time()

        count_len = len(count_imgs)
        non_count_len = len(non_count_images)

        imgs = torch.cat((count_imgs, non_count_images), dim=0)
        texts = torch.cat((count_texts, non_count_text, counterfactual_count_text), dim=0)

        encoded_imgs = self.model.encode_image(imgs)
        encoded_texts = self.model.encode_text(texts)

        count_images_encoding = encoded_imgs[:count_len]
        count_texts_encoding = encoded_texts[:count_len]
        counterfactual_text_encoding = encoded_texts[count_len + non_count_len:]

        # Counting loss, dot product of image and text embeddings
        numerator = torch.exp(count_images_encoding * count_texts_encoding)
        assert numerator.dim() == 1, "Numerator has more than 1 dimension."
        # dot product of image and counterfactual text embeddings + numerator, and maximize this
        denominator = numerator + torch.exp(count_images_encoding * counterfactual_text_encoding)
        assert denominator.dim() == 1, "Denominator has more than 1 dimension."

        # Standard CLIP loss
        img_logits = encoded_imgs @ encoded_texts[:count_len + non_count_len].t()
        text_logits = img_logits.t()

        labels = torch.arange(0, count_len + non_count_len).to(dtype=torch.long, device=self.device)

        ce_loss = (F.cross_entropy(img_logits, labels) + F.cross_entropy(text_logits, labels)) / 2
        count_loss = -torch.log(numerator / denominator).mean()

        loss = ce_loss + self.lambda_ * count_loss

        loss.backward()

        if i == 10:
            end = time.time()
            print(f"Time taken for 10th iteration: {end - start} seconds.")

        if (i + 1) % self.accumulate_steps == 0 or i == self.total_steps - 1:  # Accumulate gradients over 4 steps
            self.optimizer.step()
            self.optimizer.zero_grad()

        return (loss.item(), ce_loss.item(), count_loss.item()) # return loss for logging

    def _run_epoch(self, epoch):
        global SAVE_PATH

        b_sz = len(next(iter(self.train_data))[0])
        print(
            f"[GPU{self.device}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}"
        )
        total_steps = len(self.train_data)
        self.total_steps = total_steps
        total_steps_mod_x = total_steps // self.eval_every
        self.train_data.sampler.set_epoch(floor(epoch))
        epoch_total_loss = []
        epoch_ce_loss = []
        epoch_count_loss = []

        cur_i = 0
        if epoch > floor(epoch):
            cur_i = floor((epoch - floor(epoch)) * total_steps)
            epoch = floor(epoch)

        for i, ((count_imgs, count_texts, counterfactual_count_text), (non_count_images, non_count_text)) in tqdm(enumerate(zip(self.train_count_data, self.train_noncount_data)), desc="Training"):

            if i < cur_i:  # helps when training is interrupted in-between epochs
                continue

            if torch.cuda.is_available():
                count_imgs = count_imgs.cuda(self.device, non_blocking=True)
                count_texts = count_texts.cuda(self.device, non_blocking=True)
                counterfactual_count_text = counterfactual_count_text.cuda(self.device, non_blocking=True)
                non_count_images = non_count_images.cuda(self.device, non_blocking=True)
                non_count_text = non_count_text.cuda(self.device, non_blocking=True)

            total_loss, contrastive_loss, counting_loss = self._run_batch(count_imgs, count_texts, counterfactual_count_text, non_count_images, non_count_text, i)
            epoch_total_loss.append(total_loss)
            epoch_ce_loss.append(contrastive_loss)
            epoch_count_loss.append(counting_loss)

            # Evaluate every x iterations
            if (i + 1) % total_steps_mod_x == 0:

                if self.device == 0:
                    print(
                        f"Epoch [{epoch + i/(total_steps)}] | Total loss [{sum(epoch_total_loss)/len(epoch_total_loss) :.5f}].",
                        flush=True,
                    )
                    loss_total_train.update({epoch+(i/total_steps):epoch_total_loss})
                    print(
                        f"Epoch [{epoch + i/(total_steps)}] | Contrastive loss [{sum(epoch_ce_loss)/len(epoch_ce_loss) :.5f}].",
                        flush=True,
                    )
                    loss_clip_train.update({epoch+(i/total_steps):epoch_ce_loss})
                    print(
                        f"Epoch [{epoch + i/(total_steps)}] | Counting loss [{sum(epoch_count_loss)/len(epoch_count_loss) :.5f}].",
                        flush=True,
                    )
                    loss_count_train.update({epoch+(i/total_steps):epoch_count_loss})

                val_total_loss, _, _ = self._evaluate(epoch + i / (total_steps))
                if val_total_loss < self.cur_min_loss:
                    self.cur_min_loss = val_total_loss
                    self._save_checkpoint(epoch + i / (total_steps))

                intermediate_path = "intermediate/" + SAVE_PATH + ".pth"
                self._save_checkpoint(epoch + i / (total_steps), path=intermediate_path)

        epoch_total_loss = sum(epoch_total_loss) / len(epoch_total_loss)
        epoch_ce_loss = sum(epoch_ce_loss) / len(epoch_ce_loss)
        epoch_count_loss = sum(epoch_count_loss) / len(epoch_count_loss)        

        if self.device == 0:
            print(
                f"Epoch [{epoch + 1}] | Total loss [{epoch_total_loss :.5f}].",
                flush=True,
            )
            loss_total_train.update({epoch+1:epoch_total_loss})
            print(
                f"Epoch [{epoch + 1}] | Contrastive loss [{epoch_ce_loss :.5f}].",
                flush=True,
            )
            loss_clip_train.update({epoch+1:epoch_ce_loss})
            print(
                f"Epoch [{epoch + 1}] | Counting loss [{epoch_count_loss :.5f}].",
                flush=True,
            )
            loss_count_train.update({epoch+1:epoch_count_loss})

        return

    def _evaluate(self, epoch):

        with torch.no_grad():

            self.model.module.eval()
            eval_epoch_total_loss = []
            eval_epoch_ce_loss = []
            eval_epoch_count_loss = []
            self.val_data.sampler.set_epoch(floor(epoch))

            for i, ((count_imgs, count_texts, counterfactual_count_text), (non_count_images, non_count_text)) in tqdm(enumerate(zip(self.val_count_data, self.val_noncount_data)), desc="Evaluation"):

                if torch.cuda.is_available():
                    count_imgs = count_imgs.cuda(self.device, non_blocking=True)
                    count_texts = count_texts.cuda(self.device, non_blocking=True)
                    counterfactual_count_text = counterfactual_count_text.cuda(self.device, non_blocking=True)
                    non_count_images = non_count_images.cuda(self.device, non_blocking=True)
                    non_count_text = non_count_text.cuda(self.device, non_blocking=True)

                total_loss, contrastive_loss, counting_loss = self._run_batch(count_imgs, count_texts, counterfactual_count_text, non_count_images, non_count_text, i)
                eval_epoch_total_loss.append(total_loss)
                eval_epoch_ce_loss.append(contrastive_loss)
                eval_epoch_count_loss.append(counting_loss)

            if self.device == 0:
                print(
                    f"Epoch [{epoch}] | Total loss [{sum(eval_epoch_total_loss)/len(eval_epoch_total_loss) :.5f}].",
                    flush=True,
                )
                loss_total_val.update({epoch:sum(eval_epoch_total_loss)/len(eval_epoch_total_loss)})
                print(
                    f"Epoch [{epoch}] | Contrastive loss [{sum(eval_epoch_ce_loss)/len(eval_epoch_ce_loss) :.5f}].",
                    flush=True,
                )
                loss_clip_val.update({epoch:sum(eval_epoch_ce_loss)/len(eval_epoch_ce_loss)})
                print(
                    f"Epoch [{epoch}] | Counting loss [{sum(eval_epoch_count_loss)/len(eval_epoch_count_loss) :.5f}].",
                    flush=True,
                )
                loss_count_val.update({epoch:sum(eval_epoch_count_loss)/len(eval_epoch_count_loss)})

            self.model.module.train()
            total_loss = sum(eval_epoch_total_loss) / len(eval_epoch_total_loss)
            contrastive_loss = sum(eval_epoch_ce_loss) / len(eval_epoch_ce_loss)
            counting_loss = sum(eval_epoch_count_loss) / len(eval_epoch_count_loss)
            return total_loss, contrastive_loss, counting_loss

    def train(self, max_epochs: int):

        print("*" * 50)
        print(f"Training for {max_epochs - self.epochs_run} epochs.")
        print("*" * 50)

        for epoch in tqdm(range(floor(self.epochs_run), max_epochs), desc="Epochs"):

            if (
                self.epochs_run > epoch
            ):  # happens when training is interrupted in-between epochs
                self._run_epoch(self.epochs_run)
            else:
                self._run_epoch(epoch)

            if self.device == 0:
                self._save_snapshot(epoch + 1)

            # I have moved evaluations to _run_epoch to make readings fine-grained.

        if self.device == 0:
            self.no_save = False
            self._save_snapshot(epoch + 1)


def main(model_args: ModelArgs, train_args: TrainArgs):

    ddp_setup()

    expt = Experiment(model_args, train_args)
    torch.distributed.barrier()
    expt.train(train_args.epochs)

    destroy_process_group()


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version", type=str, default="ViT-L/14", help="Version of CLIP model to use."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--precision", type=str, default="bf16", help="Precision, fp32 or bf16 or 8bit."
    )
    parser.add_argument(
        "--no_save",
        default=False,
        dest="no_save",
        action="store_true",
        help="Wether or not to save the model.",
    )
    parser.add_argument(
        "--accumulate_steps",
        type=int,
        default=4,
        help="Number of steps to accumulate gradients over.",
    )
    parser.add_argument(
        "-s",
        "--train_size",
        type=float,
        default=0.5,
        help="Train size, 0 < train_size < 1.",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "-e",
        "--eval_every",
        type=int,
        default=2,
        help="Evaluate every eval_every epochs.",
    )
    parser.add_argument(
        "-l", "--learning_rate", type=float, default=3e-4, help="Learning rate."
    )
    parser.add_argument(
        "-o", "--optimizer", type=str, default="Adam", help="Optimizer, SGD or Adam."
    )
    parser.add_argument(
        "-w", "--weight_decay", type=float, default=1e-4, help="Weight decay."
    )
    parser.add_argument(
        "--init",
        default=False,
        dest="init",
        action="store_true",
        help="Wether or not to initialize training from scratch.",
    )
    parser.add_argument(
        "--path",
        default="/home/f20212582/snapshot_fromage.pth",
        help="Path to save snapshots to.",
    )
    parser.add_argument(
        "--clip",
        default=False,
        dest="clip",
        action="store_true",
        help="Run on 64 samples for overfitting test.",
    )
    parser.add_argument(
        "--c_ratio",
        default=1 / 16,
        type=float,
        help="Ratio of counting subset to non-counting subset.",
    )
    parser.add_argument(
        "--lambda_",
        default=0.5,
        type=float,
        help="Weight of the counting loss in the total loss.",
    )

    args = parser.parse_args()
    # print(args.n_visual_tokens, args.n_ret_tokens)

    model_args = ModelArgs()
    model_args.version = args.version

    train_args = TrainArgs()
    train_args.epochs = args.epochs
    train_args.precision = args.precision
    train_args.no_save = args.no_save
    train_args.accumulate_steps = args.accumulate_steps
    train_args.train_size = args.train_size
    train_args.batch_size = args.batch_size
    train_args.eval_every = args.eval_every
    train_args.learning_rate = args.learning_rate
    train_args.optimizer = args.optimizer
    train_args.weight_decay = args.weight_decay
    train_args.path = args.path
    train_args.init = args.init
    train_args.clip = args.clip
    train_args.c_ratio = args.c_ratio
    train_args.lambda_ = args.lambda_

    # Sanity tests
    assert 0 < args.train_size < 1, "train_size must be between 0 and 1."
    assert (
        args.optimizer == "SGD" or args.optimizer == "Adam" or args.optimizer == "AdamW"
    ), "optimizer must be SGD or Adam or AdamW."

    world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")

    # Run experiment
    main(model_args, train_args)

    print("Training Done!")

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 1]

    ax1.plot(
        list(loss_total_train.keys()),
        list(loss_total_train.values()),
        label="Train",
    )
    ax1.plot(list(loss_total_val.keys()), list(loss_total_val.values()), label="Eval")
    ax1.legend()
    ax1.set_title("Train and Eval total loss vs Epoch-steps")
    ax1.title.set_size(10)

    ax2.plot(
        list(loss_clip_train.keys()),
        list(loss_clip_train.values()),
        label="Train",
    )
    ax2.plot(list(loss_clip_val.keys()), list(loss_clip_val.values()), label="Eval")
    ax2.legend()
    ax2.set_title("Train and Eval clip loss vs Epoch-steps")
    ax2.title.set_size(10)


    # ax3.plot(list(eval_aucs.keys()), list(eval_aucs.values()), label="Eval AUC")
    # ax3.set_title("Eval AUC vs Epoch-steps")
    # ax3.title.set_size(10)

    ax3.plot(
        list(loss_count_train.keys()),
        list(loss_count_train.values()),
        label="Train",
    )
    ax3.plot(
        list(loss_count_val.keys()),
        list(loss_count_val.values()),
        label="Val",
    )
    ax3.legend()
    ax3.set_title("Train and Eval Retrieval and Captioning loss vs Epochs")
    ax3.title.set_size(10)

    global SAVE_PATH
    plt.savefig("plots/" + SAVE_PATH + ".png")
    plt.close("all")
