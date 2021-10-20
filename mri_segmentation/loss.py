import torch
from torch import nn, einsum, Tensor
from .utils import one_hot, simplex


class DiceLoss(nn.Module):
    """ Multiclass loss"""
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def dice_loss(self, score, target, dim=(-3, -2, -1)):
        assert len(score.shape) == 4
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target, dim=dim)
        z_sum = torch.sum(score * score, dim=dim)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss.mean()

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self.one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes

        assert inputs.size() == target.size(), \
            'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(self.n_classes):
            dice = self.dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]

        return loss / self.n_classes


class BoundaryLoss:
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = kwargs["idc"]

    def __call__(self, probs, dist_maps):
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bkxyz,bkxyz->bkxyz", pc, dc)

        loss = multipled.mean()

        return loss
