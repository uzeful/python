import torch

'''
Refered to the project of Zengyu714
'''


def dice_loss(output, target):
    """Compute dice among **positive** labels to avoid unbalance.

    Arguments:
        output: [batch_size * height * width,  2 ] (torch.cuda.FloatTensor)
        target: [batch_size * height * width, (1)] (torch.cuda.FloatTensor)

    Returns:
        tuple contains:
        + dice loss: for back-propagation
        + accuracy: (pred - true) / true
        + dice overlap:  2 * pred * true / (pred - true) * 100
        + pred: FloatTensor {0.0, 1.0} with shape [batch_size * height * width, (1)]
        + true: FloatTensor {0.0, 1.0} with shape [batch_size * height * width, (1)]
    """
    y_pred = output.max(1)[1]  # {0, 1}
    output = output[:, 1]
    target = target.float()

    # Loss
    # `dim = 0` for Tensor result
    intersection = torch.sum(output * target, 0)
    union = torch.sum(output * output, 0) + torch.sum(target * target, 0)
    dice = 2.0 * intersection / union

    # Overlap
    pred = y_pred.eq(1).float().data    # FloatTensor 0.0 / 1.0
    true = target.data                  # FloatTensor 0.0 / 1.0
    overlap = 2 * (pred * true).sum() / (pred.sum() + true.sum()) * 100

    # Accuracy
    acc = pred.eq(true).float().mean()
    return 1 - torch.clamp(dice, 0.0, 1.0 - 1e-7), acc, overlap, pred, true

