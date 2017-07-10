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
        + accuracy: (predtrue - true) / true
        + dice overlap:  2 * predtrue * true / (predtrue - true) * 100
        + predtrue: FloatTensor {0.0, 1.0} with shape [batch_size * height * width, (1)]
        + true: FloatTensor {0.0, 1.0} with shape [batch_size * height * width, (1)]
    """
    predict = output.max(1)[1]  # {0, 1}. 0 for the original output, 1 for the binary mask
    target = target.float()

    # Loss
    intersection = torch.sum(predict * target, 0)
    union = torch.sum(predict * target, 0) + torch.sum(target * target, 0)
    dice = 2.0 * intersection / union

    # Overlap
    predtrue = predict.eq(1).float().data    # FloatTensor 0.0 / 1.0
    true = target.data                  # FloatTensor 0.0 / 1.0
    overlap = 2 * (predtrue * true).sum() / (predtrue.sum() + true.sum()) * 100

    # Accuracy
    acc = predtrue.eq(true).float().mean()
    return 1 - torch.clamp(dice, 0.0, 1.0 - 1e-7), acc, overlap, predtrue, true

