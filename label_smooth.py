import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCELoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCELoss, self).__init__()
        self.smoothing = smoothing

    def _onehot_and_smooth(self, indexes, num_classes):
        batch_size = indexes.shape[0]
        onehot = torch.full(
            (batch_size, num_classes),
            fill_value=self.smoothing/(num_classes-1),
            device=indexes.device
        )
        onehot.scatter_(-1, indexes.unsqueeze(1), 1-self.smoothing)

        return onehot

    def forward(self, logits, target):
        pred = F.log_softmax(logits, dim=-1)
        num_classes = logits.shape[-1]
        smooth_target = self._onehot_and_smooth(target, num_classes)
        return (-smooth_target * pred).sum(dim=-1).mean()


# if __name__ == "__main__":
#     loss = LabelSmoothingCELoss()
#     input = torch.randn(3, 100, requires_grad=True)
#     target = torch.empty(3, dtype=torch.long).random_(100)
#     print(target)
#     output = loss(input, target)
#     output.backward()
