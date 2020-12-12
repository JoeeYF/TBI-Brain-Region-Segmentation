import importlib
import torch

from .loss.losses import compute_per_channel_dice
from .utils import get_distance_2point

class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and theTn simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def __call__(self, input, target):
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice(input, target, epsilon=self.epsilon))


class PointDistance:
    def __init__(self, distance_thre, **kwargs) -> None:
        self.distance_thre = distance_thre

    def __call__(self, input, target):
        distance_list = []
        for classindex in range(17):
            input_c = input[0, classindex]
            target_c = target[0, 0]
            coord = torch.where(input_c == torch.max(input_c))
            a = coord[0].float().mean()
            b = coord[1].float().mean()
            c = coord[2].float().mean()
            input_coord = torch.tensor([a, b, c]).float()
            target_coord = torch.tensor(torch.where(target_c == classindex+1)).float()
            dis = get_distance_2point(input_coord, target_coord)
            distance_list.append(dis)
        # PCP = (torch.tensor(self.distance_list) < self.distance_thre).float().mean()
        return torch.tensor(distance_list).mean(),distance_list


def get_evaluation_metric(metric_config):
    """
    Returns the evaluation metric function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'eval_metric' key
    :return: an instance of the evaluation metric
    """

    def _metric_class(class_name):
        m = importlib.import_module('core.metrics')
        clazz = getattr(m, class_name)
        return clazz

    assert 'name' in metric_config, 'Could not find evaluation metric configuration'
    
    metric_class = _metric_class(metric_config['name'])
    return metric_class(**metric_config)
