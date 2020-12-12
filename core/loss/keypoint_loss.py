import torch
import torch.nn as nn


class KeyPointFocalLoss(nn.Module):
    def __init__(self):
        super(KeyPointFocalLoss, self).__init__()
        self.normalization = nn.Sigmoid()

    def forward(self, pred, label):
        pred = self.normalization(pred)
        label = label.to(pred.dtype)
        loss = _neg_loss(pred, label)
        return loss


class KeyPointBCELoss(nn.Module):
    def __init__(self):
        super(KeyPointBCELoss, self).__init__()
        # self.normalization = nn.Sigmoid()

    def forward(self, pred, label):
        # label = dist < self.max_dist
        # pred = self.normalization(pred)
        label = label.to(pred.dtype)
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=1 / label.mean())
        return loss(pred, label)


class KeyPointMSELoss(nn.Module):
    def __init__(self):
        super(KeyPointMSELoss, self).__init__()
        self.normalization = nn.Sigmoid()
    def forward(self, pred, label):
        pred = self.normalization(pred)
        label = label.to(pred.dtype)
        l = ((pred - label)**2)
        return l.mean()


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    alpha = 2
    beta = 4
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, beta)

    loss = 0
    pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


# pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
# pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
# neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
