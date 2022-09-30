from tkinter.tix import Tree
from scripts.head import *
import torch
import torch.nn.functional as F



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_config():
    config = {
        # "remarks": "OurLossWithPair",
        "seed": 60,
        "optim_parms":{
            "lr": 5e-3,
            "momentum": 0.9,
            "weight_decay": 1e-4,
        },
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "dataset": "car_ims",
        "test_map": 1,
        "stop_iter": 20,
        "epoch": 1000,
        "device": torch.device('cuda'),
        "n_gpu": torch.cuda.device_count(),
        "max_norm": 5.0,
        "resnet_url": '../../Pretrain/resnet50-19c8e357.pth',
        "url": '../../Pretrain/moco_v2_800ep_pretrain.pth',
        "txt_path": '../../data/glove/glove.6B.50d.txt',
        "save_path": './results/class_model',
    }
    config = config_dataset(config)
    return config