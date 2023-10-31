import torch
from torch import nn

class KappaLoss(nn.Module):
    def __init__(self, beta=0):
        super(KappaLoss, self).__init__()
        self.loss = nn.CosineSimilarity(dim=0)
        self.beta = beta

    def forward(self, inputs, targets, w=None):
        base_loss = 0.5*torch.mean(1-self.loss(inputs, targets))

        if w is not None:
            w_hat = torch.sum(torch.abs(torch.fft.fft(w,dim=1))**2,dim=0)
            B = torch.max(w_hat,dim=0).values
            A = torch.min(w_hat,dim=0).values

            loss = base_loss + self.beta*B/A
        else:
            loss = base_loss

        return base_loss, loss