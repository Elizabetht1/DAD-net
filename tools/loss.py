import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    """ Masked CE Loss
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor,
                y_true: torch.Tensor,
                mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be
                  ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with
            gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred
        # and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)
        return self.loss(masked_pred, masked_true)


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy:
        1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """
    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(),
                               weight=self.weight,
                               ignore_index=self.ignore_index,
                               reduction=self.reduction)


def compute_loss(preds, targets, masks):
    if masks is not None:
        loss_fn = MaskedMSELoss(reduction='none')
        return loss_fn(preds, targets, masks)
    else:
        # modified version of pytorch's crossentropy loss to handle shape
        # of targets
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        return loss_fn(preds, targets.long())
