import torch
import random
import numpy as np

# Function to set seeds for reproducibility.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='/workspace/models/best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, dcf, model):
        score = -dcf

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(dcf, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(dcf, model)
            self.counter = 0

    def save_checkpoint(self, dcf, model):
        if self.verbose:
            print("DCF improved, saving model ...")
        torch.save(model.state_dict(), self.path)
