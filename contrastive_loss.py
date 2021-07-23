import torch
import torch.nn as nn


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=5.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-6

    def forward(self, out_1, out_2, Y):
        # torch.FloatTensor(out_1)
        euclidean_distance = nn.functional.pairwise_distance(out_1, out_2)
        loss_contrastive = torch.mean(Y * torch.pow(euclidean_distance, 2) +
                                      (1 - Y) * torch.pow
                                      (torch.clamp(self.margin - euclidean_distance.float(), min=0.0), 2))
        return loss_contrastive



