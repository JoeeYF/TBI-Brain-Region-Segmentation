import os
import argparse
from glob import glob
import torch
import torch.optim.lr_scheduler

from core.utils import get_logger, load_config_yaml, increment_dir

from core.TrainerKeyPoint import TrainerKeyPoint
from core.TrainerKeyPointMorph import TrainerKeyPointMorph
from core.TrainerSegment import TrainerSegment


def main():
    parser = argparse.ArgumentParser(description='UNet3D')
    parser.add_argument('-d', '--base_dir', type=str, help='', default='output/KpResidualUNet3D_006-baseline')
    parser.add_argument('-g', '--gpu', type=str, help='', default='0')
    args = parser.parse_args()

    config_path = glob(os.path.join(args.base_dir, '*.yaml'))[0]
    config = load_config_yaml(config_path)
    config['base_dir'] = args.base_dir
    logger = get_logger('UNet3DTrain', config['base_dir'] + '/file.log')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    trainer = eval(config['trainer']['trainer_name'])(config)
    # trainer.fit()
    trainer.predict()


if __name__ == '__main__':
    main()
