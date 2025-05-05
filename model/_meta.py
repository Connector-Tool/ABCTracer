import os
import time

import torch
import torch.nn as nn


class MetaModel(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, save_dir='', save_name=None):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if save_name is None:
            save_name = f"{self.__class__.__name__}_{int(time.time())}.pth"

        model_save_path = os.path.join(save_dir, save_name)
        torch.save(self.state_dict(), model_save_path)

    def load(self, load_dir='', load_name='model.pth'):
        model_load_path = os.path.join(load_dir, load_name)
        if not os.path.exists(model_load_path):
            raise FileNotFoundError(f"No model found at {model_load_path}")

        self.load_state_dict(torch.load(model_load_path))

    def save_checkpoint(self, optimizer, scheduler, epoch, loss, save_dir='', save_name=None):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if save_name is None:
            save_name = f"{self.__class__.__name__}_{int(time.time())}.pth"

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }
        save_path = os.path.join(save_dir, save_name)
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, optimizer, scheduler, load_dir='', load_name='model.pth'):
        load_path = os.path.join(load_dir, load_name)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No model found at {load_path}")

        checkpoint = torch.load(load_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']

    def save_weights(self, save_dir='', save_name='model.pth'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_save_path = os.path.join(save_dir, save_name)
        torch.save(self.state_dict(), model_save_path)
