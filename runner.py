import os
import json
import wandb
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from tqdm import tqdm
from tqdm import trange

from dataset import image_title_dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import clip
from PIL import Image
from PIL import ImageFile
from transformers import CLIPProcessor, CLIPModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

# wandb setup
''' You can uncomment these lines to use wandb
wandb.login()
wandb.init(project="clip", entity="wandb")
'''

'''This will download a checkpoint from a previous run
api = wandb.Api()
run = api.run("<account>/<project>/<run_id>")
run.file("model.pt").download()
'''

num2word = {1 : 'one', 2 : 'two', 3 : 'three', 4 : 'four', 5 : 'five',
            6 : 'six', 7 : 'seven', 8 : 'eight', 9 : 'nine', 10 : 'ten'}

def generate_caps(cap,
                  count,
                  counterfactual=False):
  val_caps = []
  if counterfactual==False:
    for c in list(set(range(1,11)) - set([1])):
      val_cap = cap.replace(num2word[count],num2word[c])
      val_caps.append(val_cap)

  if counterfactual==True:
    l = list(set(range(1,11)) - set([1,count]))
    n = random.choice(l)
    val_caps = cap.replace(num2word[count],num2word[n])

  return val_caps

def get_dataloader():
    
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

    faulty = pd.read_csv(faulty_path)
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
        caption = item['caps'][:110]
        # caption = item['caps']
        counts = item['counts']

        if counts>0:
            cf_cap = generate_caps(caption.lower(),counts,counterfactual=True)
            list_txt_cf.extend([cf_cap] * 5)

        list_image_path.append(img_path)
        list_txt.append(caption)

    dataset = image_title_dataset(list_image_path, list_txt, list_txt_cf)
    train_dataloader = DataLoader(dataset, batch_size=5, shuffle=False) #Define your own dataloader

    return train_dataloader

def count_loss(ei,
               ek,
               ek_cf):

    ei = torch.squeeze(ei).to(torch.float64)
    ek = torch.squeeze(ek).to(torch.float64)
    ek_cf = torch.squeeze(ek_cf).to(torch.float64)

    loss = -torch.log(torch.exp(torch.dot(ei,ek))/(torch.exp(torch.dot(ei,ek))+torch.exp(torch.dot(ei,ek_cf))))

    return loss

# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

def validateAndPlot(model,
                    preprocess,
                    device):
    
    val_image_path = '/content/data/VAL'
    val_json_path = '/content/val.json'
    
    with open(val_json_path, 'r') as f:
        val_input_data = []
        for line in f:
            obj = json.loads(line)
            val_input_data.append(obj)
    all_sims = []
    c = 0

    with torch.no_grad():
        for i in trange(len(val_input_data)):
            try:
                sims = []
                img = Image.open(val_input_data[i]['pths'])
                val_caps = generate_caps(val_input_data[i]['caps'].lower(),val_input_data[i]['counts'])
                x = preprocess(img).to(device)
                encoded_image = model.encode_image(torch.unsqueeze(x, 0))
                encoded_image = encoded_image.to(device)

                for j in range(9):
                    tokenized_text = clip.tokenize(val_caps[j]).to(device)
                    encoded_text = model.encode_text(tokenized_text)
                    encoded_text = encoded_text.to(device)

                    similarity = torch.cosine_similarity(encoded_text, encoded_image)
                    sims.append(float(similarity))
                all_sims.append((sims,val_input_data[i]['counts']))
            except:
                c=c+1
                pass
    
    y = []
    y_pred = []

    for i in range(len(all_sims)):
        y_pred.append(all_sims[i][0].index(max(all_sims[i][0]))+2)
        y.append(all_sims[i][1])

    lbls = []
    for i in range(len(np.unique(y))):
        lbls.append(num2word[np.unique(y)[i]])

    labels = lbls

    cf_matrix = confusion_matrix(y, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = labels)

    cm_display.plot()
    # plt.savefig(os.path.join(wandb.run.dir, f"count.pdf"))
    plt.show()

def main():

    # Load model
    print("Loading model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # Load checkpoint
    # checkpoint = torch.load('model_9.pt')
    # model.load_state_dict(checkpoint)

    # Get dataloader
    train_dataloader = get_dataloader()

    # Prepare the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6,betas=(0.9,0.98),eps=5e-6,weight_decay=0.2) # the lr is smaller, more safe for fine tuning to new dataset

    # Prepare the scheduler (None by default)
    # linear = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=5e-6, total_iters=25)
    # cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,last_epoch=50)
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linear,cosine], milestones=[25])

    # Loss
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    print("Training...")

    # Training loop
    # Train the model
    full_arr = []
    loss_arr = []
    num_epochs = 10

    for epoch in range(num_epochs):
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        loss_arr = []
        for batch in pbar:

            optimizer.zero_grad()
            images,texts,cf_texts = batch

            images = images.to(device)
            texts = texts.to(device)
            cf_texts = cf_texts.to(device)

            encoded_imgs = model.encode_image(images)
            encoded_texts = model.encode_text(texts)
            encoded_cf_texts = model.encode_text(torch.unsqueeze(cf_texts[4], 0))

            c_enc_imgs = encoded_imgs[4:]
            c_enc_texts = encoded_texts[4:]

            ei = c_enc_imgs
            ek = c_enc_texts
            ek_cf = encoded_cf_texts

            counting_loss = count_loss(ei,ek,ek_cf)

            # Forward pass
            logits_per_image, logits_per_text = model(images, texts)

            # Compute loss
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            total_loss = ((loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2) + counting_loss
            loss_arr.append(total_loss.item())
            # Backward pass
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
            # wandb.log({"per_step_loss": total_loss.item()})
            pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")

        if (epoch+1) % 10 == 0:
            # torch.save(model.state_dict(), os.path.join(wandb.run.dir, f"model_{epoch}.pt"))
            with open(f'loss_data_{epoch}.npy', 'wb+') as f:
                np.save(f, np.array(full_arr))

        # wandb.log({"per_epoch_loss": np.mean(loss_arr)})
        full_arr.append(loss_arr)

    # Validate
    validateAndPlot(model, preprocess, device)