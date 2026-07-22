"""Custom losses for RTMDet fine-tuning.

Registered into mmdet's MODELS registry so they can be referenced by name
from the auto-generated training config — see
`train_rtmdet.pipeline.generate_mmdet_config`, which emits a
`custom_imports` entry pointing at this module whenever
`training.class_weights` is set in hyperparameter_config.yaml.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.losses.utils import weighted_loss
from mmdet.registry import MODELS


@weighted_loss
def quality_focal_loss_class_weighted(pred, target, beta=2.0, class_weight=None):
    """Quality Focal Loss (QFL) with an added per-class weight.

    Identical to mmdet's `quality_focal_loss`, except the (N, C) elementwise
    loss is scaled per class-column before being summed over classes. QFL
    supervises every class channel independently (each column is its own
    sigmoid binary classifier — positives get the IoU-quality target, all
    other columns are pushed to 0), so weighting per column reweights both
    the positive and negative contribution of that class, not just positives
    as a plain per-sample `class_weight` would.
    """
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    label, score = target

    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes - 1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos],
        reduction='none') * scale_factor.abs().pow(beta)

    if class_weight is not None:
        loss = loss * class_weight

    loss = loss.sum(dim=1, keepdim=False)
    return loss


@MODELS.register_module()
class QualityFocalLossClassWeighted(nn.Module):
    """QualityFocalLoss with a fixed per-class weight vector.

    Drop-in replacement for mmdet's `QualityFocalLoss`: same constructor
    args plus `class_weight`, a list of length `num_classes` that scales the
    loss contribution of each class. Useful for imbalanced datasets — see
    `train_rtmdet/balancer.py::compute_class_weights()` for a formula to
    derive it from per-class instance counts.
    """

    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0,
                 class_weight=None):
        super().__init__()
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        assert isinstance(target, tuple), (
            'QualityFocalLossClassWeighted only supports the (label, score) '
            'tuple target form used by RTMDet, not the one-hot tensor form.')

        class_weight = None
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)

        loss_cls = self.loss_weight * quality_focal_loss_class_weighted(
            pred,
            target,
            weight,
            beta=self.beta,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_cls
