from PIL import Image
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import trange
import clip
from transformers import CLIPProcessor, CLIPModel
from PIL import ImageFile
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import wandb
import pandas as pd
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

# WanB login - Make sure to change the login call to wandb.login(key='<yourapikey') the first time you run.
wandb.login()
wandb.init(
      project="mlrc",
      name=f"dgx_test",
      config={"learning_rate": 5e-6,"epochs": 2,})

print('WandB login successful.')

# Settings for the model
ckpt_pth = '' # path to a checkpoint file 
en_scheduler = False #enable scheduler
en_balanced_lambda = False #enable lambda balancing as decscribed in Section X.X of our paper
en_data_check = False #iterate over the dataloader once to check if there are any faulty images present (for other datasets)
fix_faulty = True #Required to be true for our dataset

#paths for the images and their respective data
json_path = './merged.json'
image_path = './data/merged'
val_image_path = './data/VAL'
val_json_path = './val.json'
faulty_path = './faulty.csv'

## Variables for logging

list_image_path = []
list_txt = []
list_txt_cf = []
list_counts = []
train_arr = []
val_arr = []

## Model configuration settings
num_epochs = 10
lmbda = 1
lr = 5e-6


num2word = {1 : 'one', 2 : 'two', 3 : 'three', 4 : 'four', 5 : 'five',
               6 : 'six', 7 : 'seven', 8 : 'eight', 9 : 'nine', 10 : 'ten'}


# Loading and fixing raw training data
with open(json_path, 'r') as f:
    input_data = []
    for line in f:
        obj = json.loads(line)
        input_data.append(obj)

if fix_faulty: 
  faulty = pd.read_csv(faulty_path)
  indexes = list(faulty["index"])
  json_strs = list(faulty["json_str"])

  for i in trange(len(indexes)):
    input_data[int(indexes[i])] = json.loads(json_strs[i])


# class for the dataset, handles image preprocessing and text tokenization
class image_title_dataset():
    def __init__(self, list_image_path,list_txt,list_txt_cf):
        # Initialize image paths and corresponding texts
        self.image_path = list_image_path
        # Tokenize text using CLIP's tokenizer
        self.title  = clip.tokenize(list_txt)
        self.cf_title = clip.tokenize(list_txt_cf)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        # Preprocess image using CLIP's preprocessing function
        image = preprocess(Image.open(self.image_path[idx]))
        title = self.title[idx]
        cf_title = self.cf_title[idx]
        return image, title, cf_title
    
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False) #Loading model



def generate_caps(cap,count,counterfactual=False):
  '''
  generate_caps() - Function to generate captions by replacing the number in the current caption. Only for counting images.

  inputs:
    - cap:str = The caption of the image
    - count:int = The count of the image
    - counterfactual:bool = If true a random number is chosen for replacing the current number otherwise produces all possible valid combinations.

  outputs:
    - val_caps:str or list() of str = List containing all combinations of the caption, or str containing caption with count replaced with a random number,
      depending on value of counterfactual.
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

def get_lambda(text):
  '''
  get_lambda() - Function implementing Lambda Balancing as described in Section X.X of our paper.
  Applies when en_lambda is set to True. Recommended to use for small, class-imbalanced training data.

  inputs:
    - text:str = caption of the (counting) image

  outputs:
    - lmbda:float = lambda corresponding to the given caption
  '''

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

## Initalising the Dataloader for the training data

for item in input_data:
  img_path = image_path +str('/')+ item['pths'].split('/')[-1]
  caption = item['caps'][:110] 
  # caption = item['caps']
  counts = item['counts']

  if counts>0:
    cf_cap = generate_caps(caption.lower(),counts,counterfactual=True) #adding counterfactual captions for each counting image.
    list_txt_cf.extend([cf_cap] * 5)

  list_image_path.append(img_path)
  list_txt.append(caption)
  list_counts.append(counts)

dataset = image_title_dataset(list_image_path, list_txt, list_txt_cf)
train_dataloader = DataLoader(dataset, batch_size=5, shuffle=False) #Define your own dataloader

# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

if device == "cpu":
  model.double()

# Initalising optimizers and schedulers (if enabled).
optimizer = torch.optim.Adam(model.parameters(), lr=5e-6,betas=(0.9,0.98),eps=5e-6,weight_decay=0.2) # the lr is smaller, more safe for fine tuning to new dataset

if en_scheduler:
  linear = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=5e-6, total_iters=5)
  cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,last_epoch=10)
  scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linear,cosine], milestones=[5])

if en_data_check: #checking data 
  pbar = tqdm(train_dataloader, total=len(train_dataloader))
  for batch in pbar:
    pass

# Loading from checkpoint (if enabled)
if len(ckpt_pth)>0:
  api = wandb.Api()
  run = api.run(ckpt_pth)

  run.file("model_9.pt").download()
  model_ckpt = torch.load('model_9.pt')
  model.load_state_dict(model_ckpt)

  run.file("optimzier_9.pt").download()
  optimizer_ckpt = torch.load('optimizer_9.pt')
  optimizer.load_state_dict(optimizer_ckpt)

  run.file("scheduler_9.pt").download()
  scheduler_ckpt = torch.load('scheduler_9.pt')
  scheduler.load_state_dict(scheduler_ckpt)

print("\nData and model ready.")



def store_cf(y,y_pred,epoch):
  '''
  store_cf() - Function saving unnormalised confusion matrix to wandb (if enabled) or locally

  inputs:
    - y:[int] = ground truth
    - y_pred:[int] = model predictions
    - epoch:int = current epoch

  outputs:
    - unnormalised confusion matrix as a pdf
  '''


  lbls = []
  for i in range(len(np.unique(y))):
    lbls.append(num2word[np.unique(y)[i]])

  labels = lbls
  cf_matrix = confusion_matrix(y, y_pred)
  cm_display = ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = labels)

  cm_display.plot()

  try:
    plt.savefig(os.path.join(wandb.run.dir, f"cfmat_{epoch}.pdf"))
  except:
    plt.savefig(f"cfmat_{epoch}.pdf")

    return
#   plt.show()
  
def store_cf_norm(y,y_pred,epoch):
  '''
  store_cf_norm() - Function saving normalised confusion matrix to wandb (if enabled) or locally

  inputs:
    - y:[int] = ground truth
    - y_pred:[int] = model predictions
    - epoch:int = current epoch

  outputs:
    - normalised confusion matrix as a pdf
  '''

  lbls = []
  for i in range(len(np.unique(y))):
    lbls.append(num2word[np.unique(y)[i]])

  labels = lbls
  cf_matrix = confusion_matrix(y, y_pred,normalize='true')
  cm_display = ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = labels)
  cm_display.plot()

  try:
    plt.savefig(os.path.join(wandb.run.dir, f"cfmat_norm_{epoch}.pdf"))
  except:
    plt.savefig(f"cfmat_norm_{epoch}.pdf")

    return
#   plt.show()


# Normal Cross Entropy for L_{clip}
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

def count_loss(ei,ek,ek_cf):

    '''
    count_loss() - Function computing the counting loss as described in section 3.2 of the original paper

    inputs:
        - ei: Image embedding
        - ek: Factual Caption Embedding
        - ek_cf: Counterfactual Caption Embedding

    outputs:
        - loss:float = The value of the counting loss
    '''

    ei = torch.squeeze(ei).to(torch.float64)/ei.norm(dim=1)
    ek = torch.squeeze(ek).to(torch.float64)/ek.norm(dim=1)
    ek_cf = torch.squeeze(ek_cf).to(torch.float64)/ek_cf.norm(dim=1)

    loss = -torch.log(torch.exp(torch.dot(ei,ek))/(torch.exp(torch.dot(ei,ek))+torch.exp(torch.dot(ei,ek_cf))))

    return loss

def get_preds(pth,model):
  '''
    get_preds() - Runs the validation loop, obtains predictions for the zero-shot classification task and reports validation metrics.

    inputs:
        - pth: Path to validation data
        - model:CLIPModel
    
    outputs:
        - y:[int] = Ground Truth 
        - y_pred:[int] = Model Predictions
        - np.mean(all_val_loss):float = mean validation loss on the validation set
        - val_acc: Zero-Shot Classifiaction Accuracy
        - f1_scores: F1 Scores for each class
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

        cf_cap = generate_caps(cap,count,counterfactual=True)
        val_caps = generate_caps(cap,count)

        img = preprocess(img).to(device)
        encoded_image = model.encode_image(torch.unsqueeze(img, 0))
        encoded_image = encoded_image.to(device)

        tokenized_f_text = clip.tokenize(cap).to(device)
        encoded_f_text = model.encode_text(tokenized_f_text)
        encoded_f_text = encoded_f_text.to(device)

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

        all_sims.append((sims,count))
        logits_per_image, logits_per_text = model(torch.unsqueeze(img,0), tokenized_f_text)

        ground_truth = torch.arange(len(torch.unsqueeze(img,0)),dtype=torch.long,device=device)

        if en_balanced_lambda:
          lmbda = get_lambda(cap)


        counting_loss = count_loss(ei,ek,ek_cf)
        val_loss = ((loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2) + (lmbda * counting_loss)
        all_val_loss.append(val_loss.item())

        # print(val_loss.item())
      except:
         c=c+1
         pass
    print(c)
  # return np.mean(all_val_loss),all_sims
  for i in range(len(all_sims)):
    y_pred.append(all_sims[i][0].index(max(all_sims[i][0]))+2)
    y.append(all_sims[i][1])

  val_acc = accuracy_score(y, y_pred)
  f1_scores = f1_score(y, y_pred, average=None)

  return y,y_pred,np.mean(all_val_loss),val_acc,f1_scores



## Training loop, trains for num_epochs
for epoch in range(num_epochs):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))

    train_loss_arr = []
    val_loss_arr = []

    for batch in pbar:

        optimizer.zero_grad()
        images,texts,cf_texts = batch

        images = images.to(device)
        texts = texts.to(device)
        cf_texts = cf_texts.to(device)

        encoded_imgs = model.encode_image(images)
        encoded_texts = model.encode_text(texts)
        encoded_cf_texts = model.encode_text(torch.unsqueeze(cf_texts[4], 0))

        nc_enc_imgs = encoded_imgs[0:4]
        nc_enc_texts = encoded_texts[0:4]

        c_enc_imgs = encoded_imgs[4:]
        c_enc_texts = encoded_texts[4:]

        ei = c_enc_imgs
        ek = c_enc_texts
        ek_cf = encoded_cf_texts

        counting_loss = count_loss(ei,ek,ek_cf)

        if en_balanced_lambda:
          lmbda = get_lambda(str(texts[4]))

        # Forward pass
        logits_per_image, logits_per_text = model(images, texts)

        # Compute loss
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        total_loss = ((loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2) + (lmbda * counting_loss)
        train_loss_arr.append(total_loss.item())
        # Backward pass

        total_loss.backward()
        if device == "cpu":
            optimizer.step()
            if en_scheduler:
              scheduler.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            if en_scheduler:
              scheduler.step()
            clip.model.convert_weights(model)

        try:
          wandb.log({"per_step_train_loss": total_loss.item()})
          wandb.log({"running_train_loss": sum(train_loss_arr)/len(train_loss_arr)})
        except:
           pass

        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {sum(train_loss_arr)/len(train_loss_arr):.4f}")

    y,y_pred,val_loss,val_acc,f1_scores = get_preds(val_json_path,model)
    val_loss_arr.append(val_loss)
    store_cf(y,y_pred,epoch)
    store_cf_norm(y,y_pred,epoch)

    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_acc}")
    try:
      wandb.log({"per_epoch_val_loss": val_loss})
      wandb.log({"per_epoch_val_acc": val_acc})

      for i in range(len(f1_scores)):
        wandb.log({f"f1_score_class{i}": f1_scores[i]})
    except:
       pass

    if (epoch+1) % 10 == 0:
      try:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, f"model_{epoch}.pt"))
        torch.save(optimizer.state_dict(), os.path.join(wandb.run.dir, f"optimizer_{epoch}.pt"))
        if en_scheduler:
          torch.save(scheduler.state_dict(), os.path.join(wandb.run.dir, f"scheduler_{epoch}.pt"))
      except:
         pass

    try:
      wandb.log({"per_epoch_loss": np.mean(train_loss_arr)})
    except:
       pass
    train_arr.append(train_loss_arr)
