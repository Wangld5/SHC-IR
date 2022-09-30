# served for generate semantic category
# from dataset.online_loader_cls import *
# served for retrieval
from dataset.online_loader import *

from torchvision import models
from model.Net import Binary_hash, ResNet, MoCo, ResNetClass
import os
import torch.optim as optim
import time
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from torch.nn.modules import BCELoss
from loguru import logger
import torch
torch.multiprocessing.set_sharing_strategy('file_system')  # multiprocessing to read files
import torch.nn as nn
import random
import shutil
from torch.utils.tensorboard import SummaryWriter

# served for generate semantic category
# from scripts.utils_cls import *
# served for retrieval
from scripts.utils import *
import pdb

