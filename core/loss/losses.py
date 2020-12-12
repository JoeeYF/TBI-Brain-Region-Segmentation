from .segment_loss import *
from .keypoint_loss import *
from .voxmorph_loss import *


def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    assert 'name' in config, 'Could not find loss function configuration'
    loss_config = config
    name = loss_config.pop('name')

    ignore_index = loss_config.pop('ignore_index', None)
    skip_last_target = loss_config.pop('skip_last_target', False)
    weight = loss_config.pop('weight', None)

    if weight is not None:
        # convert to cuda tensor if necessary
        weight = torch.tensor(weight).to(config['device'])

    pos_weight = loss_config.pop('pos_weight', None)
    if pos_weight is not None:
        # convert to cuda tensor if necessary
        pos_weight = torch.tensor(pos_weight).to(config['device'])

    loss = _create_loss(name, loss_config, weight, ignore_index, pos_weight)

    if not (ignore_index is None or name in ['CrossEntropyLoss', 'WeightedCrossEntropyLoss']):
        # use MaskingLossWrapper only for non-cross-entropy losses, since CE losses allow specifying 'ignore_index' directly
        loss = _MaskingLossWrapper(loss, ignore_index)

    if skip_last_target:
        loss = SkipLastTargetChannelWrapper(loss, loss_config.get('squeeze_channel', False))

    return loss


SUPPORTED_LOSSES = ['BCEWithLogitsLoss', 'BCEDiceLoss', 'CrossEntropyLoss', 'WeightedCrossEntropyLoss',
                    'PixelWiseCrossEntropyLoss', 'GeneralizedDiceLoss', 'DiceLoss', 'TagsAngularLoss', 'MSELoss',
                    'SmoothL1Loss', 'L1Loss', 'WeightedSmoothL1Loss','NCC','Grad']


def _create_loss(name, loss_config, weight, ignore_index, pos_weight):
    if name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif name == 'BCEDiceLoss':
        alpha = loss_config.get('alphs', 1.)
        beta = loss_config.get('beta', 1.)
        sigmoid_normalization = loss_config.get('sigmoid_normalization', False)
        return BCEDiceLoss(alpha, beta, sigmoid_normalization)
    elif name == 'CrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif name == 'WeightedCrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return WeightedCrossEntropyLoss(ignore_index=ignore_index)
    elif name == 'DiceLoss':
        sigmoid_normalization = loss_config.get('sigmoid_normalization', True)
        return DiceLoss(weight=weight, sigmoid_normalization=sigmoid_normalization)
    elif name == 'MSELoss':
        return MSELoss()
    elif name == 'SmoothL1Loss':
        return SmoothL1Loss()
    elif name == 'L1Loss':
        return L1Loss()
    elif name == 'WeightedSmoothL1Loss':
        return WeightedSmoothL1Loss(threshold=loss_config['threshold'], initial_weight=loss_config['initial_weight'],
                                    apply_below_threshold=loss_config.get('apply_below_threshold', True))

    ## keypoint
    elif name == 'KeyPointBCELoss':
        return KeyPointBCELoss()
    elif name == 'KeyPointMSELoss':
        return KeyPointMSELoss()
    elif name == 'KeyPointFocalLoss':
        return KeyPointFocalLoss()

    ## voxmorph
    elif name == 'NCC':
        return NCC()
    elif name == 'MSE':
        return MSE()
    elif name == 'Grad':
        return Grad('l2', loss_mult=2)
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'. Supported losses: {SUPPORTED_LOSSES}")


class _MaskingLossWrapper(nn.Module):
    """
    Loss wrapper which prevents the gradient of the loss to be computed where target is equal to `ignore_index`.
    """

    def __init__(self, loss, ignore_index):
        super(_MaskingLossWrapper, self).__init__()
        assert ignore_index is not None, 'ignore_index cannot be None'
        self.loss = loss
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = target.clone().ne_(self.ignore_index)
        mask.requires_grad = False

        # mask out input/target so that the gradient is zero where on the mask
        input = input * mask
        target = target * mask

        # forward masked input and target to the loss
        return self.loss(input, target)
