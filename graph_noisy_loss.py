import torch


def graph_noisy_loss(labels, adj):
    ones = torch.ones([len(labels), len(labels)]).long()
    zeros = torch.zeros([len(labels), len(labels)]).long()
    labels = labels.unsqueeze(0)
    labels_expand = labels.expand(labels.shape[1], labels.shape[1])
    labels_matrix = (labels_expand - labels_expand.T).long()  # label相同则为0，label不同则不为0
    labels_matrix = torch.where(labels_matrix == 0, ones, zeros)
    un_correct_matrix = torch.where(labels_matrix == adj, zeros, ones)
    un_correct = torch.sum(un_correct_matrix)
    loss = float(un_correct) / float(len(un_correct_matrix) ** 2)
    return loss
















