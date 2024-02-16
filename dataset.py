import numpy as np

import torch
from torch.utils.data import Dataset

import clip
import requests
from datetime import datetime

_, preprocess = clip.load('ViT-L/14')

def split(data, split_ratio=0.8):
    '''
    Split the data into training and validation sets
    '''
    np.random.shuffle(data)
    split_idx = int(len(data) * split_ratio)
    return data[:split_idx], data[split_idx:]

def verify_url(url, timeout=1):
    try:
        start_time = datetime.now()
        response = requests.get(url, timeout=timeout)
        end_time = datetime.now()
        time_taken = (end_time - start_time).total_seconds()

        if response.status_code == 200 and time_taken <= timeout:
            print(f"URL {url} is reachable and took {time_taken} seconds to fetch.")
            return True
        else:
            print(f"URL {url} is unreachable or took more than {timeout} seconds to fetch.")
            return False
    except Exception as e:
        return False

NUM_TO_TEXT = {1 : 'one', 2 : 'two', 3 : 'three', 4 : 'four', 5 : 'five', 
               6 : 'six', 7 : 'seven', 8 : 'eight', 9 : 'nine', 10 : 'ten'}

class CountSubset(Dataset):

    def __init__(self, data) -> None:
        '''
        Structure - [(
            [list of preds] has length = number of objects and each pred is an integer denoting the class of the object, 
            string URL, 
            string caption, 
            int count), ...]
        '''
        super().__init__()
        self.data = data
        self.true_data, self.false_data = self.process()

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        while True:
            valid_or_not = verify_url(self.data[idx][0])
            if valid_or_not:
                break
            else:
                idx = np.random.randint(0, len(self.data))
        img = preprocess(self.true_data[idx][0])
        caption = self.true_data[idx][1]
        counterfactual_caption = self.false_data[idx][1]

        return img, caption, counterfactual_caption
    
    def process(self) -> tuple:
        new_true_data = []
        new_false_data = []
        for i in range(len(self.data)):
            assert type(self.data[i][1]) == str
            img_url = self.data[i][1]
            assert type(self.data[i][2]) == str
            caption = self.data[i][2].lower()
            # put a false number in the caption
            counterfactual_caption = caption.copy().replace(NUM_TO_TEXT[self.data[i][3]], NUM_TO_TEXT[np.random.randint(1, 11)])

            new_true_data.append((img_url, clip.tokenize(caption)))
            new_false_data.append((img_url, clip.tokenize(counterfactual_caption)))

        return new_true_data, new_false_data
    
class NonCountSubset(Dataset):

    def __init__(self, data) -> None:
        '''
        Structure - [(
            [list of preds] has length = number of objects and each pred is an integer denoting the class of the object, 
            string URL, 
            string caption, 
            int count), ...]
        '''
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        while True:
            valid_or_not = verify_url(self.data[idx][0])
            if valid_or_not:
                break
            else:
                idx = np.random.randint(0, len(self.data))
        img = preprocess(self.true_data[idx][0])
        caption = self.true_data[idx][1]

        return img, caption
  
if __name__ == '__main__':
    dataset = CountSubset()
    print(dataset[0])