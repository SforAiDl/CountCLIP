'''
Dataset File.
'''

import clip
from PIL import Image

class image_title_dataset():
    def __init__(self, 
                 list_image_path,
                 list_txt,
                 list_txt_cf,
                 clip_version="ViT-B/32"):
        
        # Initialize image paths and corresponding texts
        self.image_path = list_image_path
        # Tokenize text using CLIP's tokenizer
        self.title  = clip.tokenize(list_txt)
        self.cf_title = clip.tokenize(list_txt_cf)

        _, self.preprocess = clip.load(clip_version)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        # Preprocess image using CLIP's preprocessing function
        image = self.preprocess(Image.open(self.image_path[idx]))
        title = self.title[idx]
        cf_title = self.cf_title[idx]
        return image, title, cf_title