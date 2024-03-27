import os
import json
import wandb
import pandas as pd
from argparse import ArgumentParser

from tqdm import tqdm
from tqdm import trange

from dataset import imageTitleDataset
from utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import clip
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

list_counts = []
val_image_path = '/content/data/VAL'
val_json_path = '/content/val.json'

num2word = {1 : 'one', 2 : 'two', 3 : 'three', 4 : 'four', 5 : 'five',
            6 : 'six', 7 : 'seven', 8 : 'eight', 9 : 'nine', 10 : 'ten'}

def getDataloader():
    '''
    Returns a dataloader for the training data.
    '''

    global list_counts

    # the following files will exist after the data is downloaded
    json_path = '/content/merged.json'
    image_path = '/content/data/merged'
    faulty_path = '/content/faulty.csv'

    # Load the data
    print("Loading data...")
    with open(json_path, 'r') as f:
        input_data = []
        for line in f:
            obj = json.loads(line)
            input_data.append(obj)

    faulty = pd.read_csv(faulty_path) # Contains image paths known not to exist
    indexes = list(faulty["index"])
    json_strs = list(faulty["json_str"])

    for i in trange(len(indexes)):
        input_data[int(indexes[i])] = json.loads(json_strs[i])

    # Get data
    list_image_path = []
    list_txt = []
    list_txt_cf = []

    for item in input_data:
        img_path = image_path +str('/')+ item['pths'].split('/')[-1]
        caption = item['caps'][:110] # Limit caption length to 110 characters
        counts = item['counts']

        if counts > 0: # if it's a counting image, get the counterfactual captions
            cf_cap = generate_caps(caption.lower(), counts, counterfactual=True)
            list_txt_cf.extend([cf_cap] * 5) # indexing convinience

        list_counts.append(counts)
        list_image_path.append(img_path)
        list_txt.append(caption)

    dataset = imageTitleDataset(list_image_path, list_txt, list_txt_cf)
    train_dataloader = DataLoader(dataset, batch_size=5, shuffle=False) #Define your own dataloader
    # Note, changing batch_size is more involved because we need to maintain a fixed ratio between
    # counting and non-counting images (here 1:4).

    return train_dataloader

class TrainArgs:
    balanced_lambda : bool = True
    save_ckpt : bool = True
    num_epochs : int = 10
    with_tracking : bool = True
    scheduler : str = "original"
    resume_from_checkpoint : bool = False
    checkpoint_path : str = 'model_9.pt'

def parse_args():

    args = TrainArgs()
    parser = ArgumentParser()   
    parser.add_argument("--balanced_lambda", action="store_true", default=args.balanced_lambda,
                        help="Use balanced lambda for counting loss")
    parser.add_argument("--save_ckpt", action="store_true", default=args.save_ckpt,
                        help="Save model checkpoints")
    parser.add_argument("--num_epochs", type=int, default=args.num_epochs,
                        help="Number of epochs")
    parser.add_argument("--with_tracking", action="store_true", default=args.with_tracking,
                        help="Use wandb for tracking")
    parser.add_argument("--scheduler", type=str, default=args.scheduler, choices=["linear", "cosine", "original"],
                        help="Scheduler to use")
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=args.resume_from_checkpoint,
                        help="Resume training from a checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=args.checkpoint_path,
                        help="Path to checkpoint")
    input_args = parser.parse_args()
    args.__dict__.update(input_args.__dict__)

def main(args : TrainArgs):

    # Load model
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # wandb setup
    if args.with_tracking:
        wandb.login()
        wandb.init(project="clip", entity="wandb")

    '''This will download a checkpoint from a previous run
    api = wandb.Api()
    run = api.run("<account>/<project>/<run_id>")
    run.file("model.pt").download()
    '''

    # Load checkpoint
    if args.resume_from_checkpoint:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)

    # Get dataloader
    train_dataloader = getDataloader()

    # Prepare the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6,betas=(0.9,0.98), eps=5e-6, weight_decay=0.2) # the lr is smaller, more safe for fine tuning to new dataset

    # Prepare the scheduler (None by default)
    if isinstance(args.scheduler, str):
        if args.scheduler == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=5e-6, total_iters=25)
        elif args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, last_epoch=50)
        elif args.scheduler == "original":
            linear = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=5e-6, total_iters=25)
            cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, last_epoch=50)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linear,cosine], milestones=[25])
        else:
            raise ValueError("Invalid scheduler")
    else:
        scheduler = None
        print("No scheduler specified, whereas the original paper recommends using both linear and cosine schedulers.")

    if not args.balanced_lambda:
        print("Balanced lambda not specified, using default value of 1. It is recommended to use this setting as \
              it evens out the strong class imbalance in the dataset. (~2000+ images with 2 and ~8 images with 10)")

    # Loss
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    print("Training...")

    # Train the model
    full_arr = []
    loss_arr = []
    val_loss_arr = []

    # Training loop
    for epoch in range(args.num_epochs):
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        loss_arr = []

        for batch in pbar:

            optimizer.zero_grad()
            images, texts, cf_texts = batch

            images = images.to(device)
            texts = texts.to(device)
            cf_texts = cf_texts.to(device)

            encoded_imgs = model.encode_image(images)
            encoded_texts = model.encode_text(texts)
            encoded_cf_texts = model.encode_text(torch.unsqueeze(cf_texts[4], 0)) # use only the last element of the counterfactual captions (all are same, only for convinience)

            c_enc_imgs = encoded_imgs[4:]
            c_enc_texts = encoded_texts[4:]

            ei = c_enc_imgs
            ek = c_enc_texts
            ek_cf = encoded_cf_texts

            counting_loss = count_loss(ei, ek, ek_cf)

            # Forward pass
            logits_per_image, logits_per_text = model(images, texts)

            # get lambda
            if args.balanced_lambda:
                lmbda = get_lambda(texts[4], list_counts, train_dataloader) # last item in each batch is the counting image
            else:
                lmbda = 1

            # Compute loss
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            total_loss = ((loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2) + lmbda*counting_loss
            loss_arr.append(total_loss.item())
            # Backward pass
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            if scheduler is not None:
                scheduler.step()
            
            if args.with_tracking:
                wandb.log({"per_step_train_loss": total_loss.item()})
                wandb.log({"running_train_loss": sum(loss_arr)/len(loss_arr)})

            pbar.set_description(f"Epoch {epoch}/{args.num_epochs}, Loss: {total_loss.item():.4f}")

        if (epoch+1) % 10 == 0:

            if args.save_ckpt:
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, f"model_{epoch}.pt"))
                torch.save(optimizer.state_dict(), os.path.join(wandb.run.dir, f"optimizer_{epoch}.pt"))
                if args.scheduler:
                    torch.save(scheduler.state_dict(), os.path.join(wandb.run.dir, f"scheduler_{epoch}.pt"))

        full_arr.extend(loss_arr)
        pbar.set_description(f"Epoch {epoch}/{args.num_epochs}, Loss: {sum(loss_arr)/len(loss_arr):.4f}")

        y, y_pred, val_loss, val_acc, f1_scores = get_preds(val_json_path, model, preprocess, device, args.balanced_lambda)
        val_loss_arr.append(val_loss)
        store_cf_norm(y,y_pred,epoch)

        print(f"Validation Loss: {val_loss}")
        print(f"Validation Accuracy: {val_acc}")
        if args.with_tracking:
            wandb.log({"per_epoch_val_loss": val_loss})
            wandb.log({"per_epoch_val_acc": val_acc})

        for i in range(len(f1_scores)) and args.with_tracking:
            wandb.log({f"f1_score_class{i}": f1_scores[i]})