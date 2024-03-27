import json
import clip
import random
import numpy as np
from PIL import Image
from tqdm import trange
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

num2word = {1 : 'one', 2 : 'two', 3 : 'three', 4 : 'four', 5 : 'five',
            6 : 'six', 7 : 'seven', 8 : 'eight', 9 : 'nine', 10 : 'ten'}

# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model : nn.Module):
    '''
    Utility function to convert model's parameters to FP32 format,
    before backprop.
    '''
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

def get_lambda(text : str,
               list_counts : list,
               train_dataloader : DataLoader):
    '''
    Balances out the dataset by adjusting the lambda value.
    
    Arguments
    ---------
        text : str
            A string containing the text of the caption.
        list_counts : list
            A list of the number of captions in each batch.
        train_dataloader : torch.utils.data.DataLoader
            A DataLoader object containing the training data.
    '''
    print(text)
    lmbda = 1
    l = list_counts
    n_count = len(train_dataloader)


    if "two" in text: lmbda = 1-(list_counts.count(2)/n_count)
    if "three" in text: lmbda = 1-(list_counts.count(3)/n_count)
    if "four" in text: lmbda = 1-(list_counts.count(4)/n_count)
    if "five" in text: lmbda = 1-(list_counts.count(5)/n_count)
    if "six" in text: lmbda = 1-(list_counts.count(6)/n_count)
    if "seven" in text: lmbda = 1-(list_counts.count(7)/n_count)
    if "eight" in text: lmbda = 1-(list_counts.count(8)/n_count)
    if "nine" in text: lmbda = 1-(list_counts.count(9)/n_count)
    if "ten" in text: lmbda = 1-(list_counts.count(10)/n_count)

    return lmbda

def count_loss(ei : torch.Tensor,
               ek : torch.Tensor,
               ek_cf : torch.Tensor):
    '''
    Convention from the paper
    ei: image embedding
    ek: true caption embedding
    ek_cf: counterfactual caption embedding
    '''

    ei = torch.squeeze(ei).to(torch.float64)/ei.norm(dim=1)
    ek = torch.squeeze(ek).to(torch.float64)/ek.norm(dim=1)
    ek_cf = torch.squeeze(ek_cf).to(torch.float64)/ek_cf.norm(dim=1)

    loss = -torch.log(torch.exp(torch.dot(ei,ek))/(torch.exp(torch.dot(ei,ek))+torch.exp(torch.dot(ei,ek_cf))))

    return loss

def generate_caps(cap : str,
                  count : int,
                  counterfactual : bool =False):
    '''
    Generates counterfactual captions for counting images.
    '''
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

def get_preds(pth : str, 
              model : nn.Module,
              preprocess : Compose,
              device : torch.device,
              balanced_lambda : bool = True):
    '''
    Runs the validation loop.

    Arguments
    ---------
    pth
        Path to validation file
    model
        CLIP model
    preprocess
        a Compose object returned when loading the model
    device
        current device
    balanced_lambda
        whether the current run is using the balanced lambda scheme
    '''
    #opening validation data
    with open(pth, 'r') as f:
        val_input_data = []
        for line in f:
            obj = json.loads(line)
            val_input_data.append(obj)

    all_sims = []
    c = 0
    all_val_loss = []
    y = []
    y_pred = []
    lbls = []
    lmbda = 1

    with torch.no_grad():
        for i in trange(len(val_input_data)):
            try:
                sims = []
                img = Image.open(val_input_data[i]['pths'])
                cap = val_input_data[i]['caps'].lower()
                cap = cap[0:77]
                count = val_input_data[i]['counts']

                cf_cap = generate_caps(cap, count, counterfactual=True)
                val_caps = generate_caps(cap, count)

                # Preprocess the image
                img = preprocess(img).to(device)
                encoded_image = model.encode_image(torch.unsqueeze(img, 0))
                encoded_image = encoded_image.to(device)

                # Encode the text
                tokenized_f_text = clip.tokenize(cap).to(device)
                encoded_f_text = model.encode_text(tokenized_f_text)
                encoded_f_text = encoded_f_text.to(device)

                # Generate counterfactual captions
                cf_cap = generate_caps(cap, count, counterfactual=True)
                tokenized_cf_text = clip.tokenize(cf_cap).to(device)
                encoded_cf_text = model.encode_text(tokenized_cf_text)
                encoded_cf_text = encoded_cf_text.to(device)

                ei = encoded_image
                ek = encoded_f_text
                ek_cf = encoded_cf_text

                for j in range(9):
                    tokenized_text = clip.tokenize(val_caps[j]).to(device)
                    encoded_text = model.encode_text(tokenized_text)
                    encoded_text = encoded_text.to(device)

                    similarity = torch.cosine_similarity(encoded_text, encoded_image)
                    sims.append(float(similarity))

                all_sims.append((sims, count))
                logits_per_image, logits_per_text = model(torch.unsqueeze(img, 0), tokenized_f_text)

                ground_truth = torch.arange(len(torch.unsqueeze(img, 0)), dtype=torch.long, device=device)

                if balanced_lambda:
                    lmbda = get_lambda(cap)

                counting_loss = count_loss(ei, ek, ek_cf)
                val_loss = ((F.cross_entropy(logits_per_image, ground_truth) + F.cross_entropy(logits_per_text, ground_truth)) / 2) + (lmbda * counting_loss)
                all_val_loss.append(val_loss.item())

            except:
                c = c + 1
                pass

    print(f"Got {c} faulty images.")

    for i in range(len(all_sims)):
        y_pred.append(all_sims[i][0].index(max(all_sims[i][0])) + 2)
        y.append(all_sims[i][1])

    val_acc = accuracy_score(y, y_pred)
    f1_scores = f1_score(y, y_pred, average=None)

    return y, y_pred, np.mean(all_val_loss), val_acc, f1_scores

def store_cf_norm(y, 
                  y_pred, 
                  epoch):
    '''
    Plots and stores the normalized Confusion matrix.
    Arguments
    ---------
    y
        actual outcomes
    y_pred
        predicted outcomes
    epoch
        Current epoch for saving
    '''
    lbls = []
    for i in range(len(np.unique(y))):
        lbls.append(num2word[np.unique(y)[i]])

    labels = lbls
    cf_matrix = confusion_matrix(y, y_pred,normalize='true')
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = labels)
    cm_display.plot()

    plt.savefig(f"plots/cfmat_norm_{epoch}.pdf")

    plt.show()