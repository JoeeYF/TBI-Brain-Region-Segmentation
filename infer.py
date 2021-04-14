import os
import argparse
from glob import glob
import torch
import torch.optim.lr_scheduler

from core.utils import get_logger, load_config_yaml, increment_dir

from core.TrainerKeyPoint import TrainerKeyPoint
from core.TrainerKeyPointMorph import TrainerKeyPointMorph
from core.TrainerSegment import TrainerSegment
from core.TrainerSegmentMorph import TrainerSegmentMorph
from core.TrainerSegmentMP import TrainerSegmentMP

def main():
    parser = argparse.ArgumentParser(description='UNet3D')
    parser.add_argument('-d', '--base_dir', type=str, help='', default='output/seg/SegResidualUNet3D_112-fold1_mp5_group4')
    parser.add_argument('-g', '--gpu', type=str, help='', default='0')
    parser.add_argument('-f', '--fold', type=int, help='fold', default=1)
    args = parser.parse_args()

    config_path = glob(os.path.join(args.base_dir, '*.yaml'))[0]
    config = load_config_yaml(config_path)
    config['base_dir'] = args.base_dir
    logger = get_logger('UNet3DTrain', config['base_dir'] + '/file.log')
    config['loaders']['val_fold'] = args.fold
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    trainer = eval(config['trainer']['trainer_name'])(config)
    # trainer.fit()
    # trainer.predict()
    trainer.predict_heatmap()


if __name__ == '__main__':
    main()
