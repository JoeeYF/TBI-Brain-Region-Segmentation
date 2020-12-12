import os
import argparse
import torch
import torch.optim.lr_scheduler

from core.utils import get_logger, load_config_yaml, increment_dir

from core.TrainerKeyPoint import TrainerKeyPoint
from core.TrainerKeyPointMorph import TrainerKeyPointMorph
from core.TrainerSegment import TrainerSegment

def main():
    parser = argparse.ArgumentParser(description='UNet3D')
    parser.add_argument('-cfg', '--config', type=str, help='Path to the YAML config file', default='./config_kp.yaml')
    args = parser.parse_args()

    config = load_config_yaml(args.config)
    base_dir = increment_dir(f"output/{config['base_dir_prev']}_", config['description'])
    os.system(f'cp {args.config} {base_dir}')
    os.system(f'cp -r core {base_dir}')
    config['base_dir'] = base_dir

    # config['base_dir'] = 'output/KpResidualUNet3D_006-baseline'

    logger = get_logger('UNet3DTrain', config['base_dir']+'/file.log')

    gpus = ','.join([str(i) for i in config['GPUs']])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    trainer = eval(config['trainer']['trainer_name'])(config)
    trainer.fit()
    trainer.predict()


if __name__ == '__main__':
    main()
