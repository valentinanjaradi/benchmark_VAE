import torchvision 
import numpy as np
import os

os.makedirs("examples/scripts/data/mnist", exist_ok=True)

full_train = torchvision.datasets.MNIST("examples/scripts/data/mnist", train=True,  download=True, transform=None)
full_eval = torchvision.datasets.MNIST("examples/scripts/data/mnist", train=False,  download=True, transform=None)
train_dataset = full_train.data.reshape(-1, 1, 28, 28)
eval_dataset = full_eval.data.reshape(-1, 1, 28, 28)
np.savez('examples/scripts/data/mnist/train_data.npz', data=train_dataset)
np.savez('examples/scripts/data/mnist/eval_data.npz', data=eval_dataset)