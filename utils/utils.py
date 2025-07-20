import cv2
import numpy as np
import torch
import torch.nn as nn
import os


def read_image(image_path:str, image_size:int=224):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    return image

def infinite_iterator(loader):
    while True:
        for batch in loader:
            yield batch


def save_state(save_path: str, model: nn.Module, optim: torch.optim.Optimizer, info: dict | None = None):
    state_dict = {
        'model_state': model.state_dict(),
        'optim_state': optim.state_dict(),
        'info': info
    }
    try:
        os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
        torch.save(state_dict, save_path)
        print(f'State saved.')
        # print('Info:', info)
    except Exception as e:
        print(f'ERROR: {e}')


def load_state(path: str, model: nn.Module=None, optim: torch.optim.Optimizer=None)-> tuple[nn.Module, torch.optim.Optimizer]:
    obj = torch.load(path)
    if model is not None:
        model.load_state_dict(obj['model_state'])
        print('Model State Loaded')
    if optim is not None:
        optim.load_state_dict(obj['optim_state'])
        print('Optimizer State Loaded')
    print(f'Loaded state.')
    print(obj['info'])
    return model, optim