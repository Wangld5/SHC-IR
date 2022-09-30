from scripts.train_CUB import *
import numpy as np

if __name__ == '__main__':
    config = get_config()
    best_result = 0
    for bit in config['bit_list']:
        config['order_seed'] = 80
        flag = True
        if bit == 48:
            flag = False
        l = list(range(config['n_class']))
        random.seed(config['order_seed'])
        random.shuffle(l)
        setup_seed(config['seed'])
        config["center_path"] = f"./centersDzero/CSQ_init_{flag}_{config['n_class']}_{bit}_L2_alpha1.npy"
        config["CSQ_center"] = f"./centersCSQ/CSQ_{config['n_class']}_{bit}.npy"
        config["remarks"] = "OurLossDzero"
        config["save_path"] = f"./results/{config['dataset']}/{config['remarks']}"
        config["save_center"] = f"./results/{config['dataset']}/{config['remarks']}/CSQ_{bit}.npy"
        config["info"] = f"[{config['remarks']}]"
        config["loss_way"] = f"{config['remarks']}"
        logger.add(f'logs/{config["dataset"]}/{config["info"]}/{bit}.log',
                   rotation='500 MB',
                   level='INFO',
                   )
        train_val(config, bit, l)



