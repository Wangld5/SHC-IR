from scripts.train_classifier import *
import numpy as np

if __name__ == '__main__':
    config = get_config()
    best_result = 0
    config['cls_model'] = 'ResNet'
    logger.add(f'logs/{config["dataset"]}.log',
                rotation='500 MB',
                level='INFO',
                )
    setup_seed(config['seed'])
    train_val(config)



