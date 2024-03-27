'''
Dataset File.
'''

import clip
from PIL import Image
from torch.utils.data import Dataset

class imageTitleDataset(Dataset):
    
    def __init__(self, 
                 list_image_path,
                 list_txt,
                 list_txt_cf,
                 clip_version="ViT-B/32"):
        '''
        Arguments
        ---------
        list_image_path
            A list of image paths
        list_txt
            A list of true captions
        list_txt_cf
            A list of counterfactual captions

        Preprocesses images and tokenizes texts using CLIP's 
        preprocessing function and tokenizer.
        '''
        
        # Initialize image paths and corresponding texts
        self.image_path = list_image_path
        self.text = list_txt
        # Tokenize text using CLIP's tokenizer
        self.caption  = clip.tokenize(list_txt)
        self.cf_caption = clip.tokenize(list_txt_cf)

        _, self.preprocess = clip.load(clip_version)

    def __len__(self):

        return len(self.caption)

    def __getitem__(self, idx):

        # Preprocess image using CLIP's preprocessing function
        image = self.preprocess(Image.open(self.image_path[idx]))
        text = self.text[idx]
        caption = self.caption[idx]
        cf_caption = self.cf_caption[idx]

        return image, caption, cf_caption, text