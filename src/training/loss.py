import torch
import torch.nn as nn

# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py


class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, 2, L)` where N is the batch size and L is the sequence length
        - Target: :math:`(N, L)` where each value is :math:`0 ≤ targets[i] ≤ 1`.
    """

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError(
                "Input type is not a torch.Tensor. Got {}".format(type(input))
            )

        if not len(input.shape) == 3:
            raise ValueError(
                "Invalid input shape, we expect (N, 2, L). Got: {}".format(input.shape)
            )

        if not input.shape[-1] == target.shape[-1]:
            raise ValueError(
                "input and target shapes must be the same. Got: {}".format(
                    input.shape, input.shape
                )
            )

        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}".format(
                    input.device, target.device
                )
            )

        # get positive prediction only
        positive_preds = input[:, 1, :]

        # compute the actual dice score
        intersection = torch.sum(positive_preds * target, (1))
        union = torch.sum(positive_preds + target, (1))

        dice_score = 2.0 * intersection / (union + self.eps)

        return torch.mean(1.0 - dice_score)
