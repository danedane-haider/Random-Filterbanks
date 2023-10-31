import torch
import torch.nn.functional as F

class TDFilterbank(torch.nn.Module):
    def __init__(self, spec, w):
        super().__init__()
        
        self.psi = torch.nn.Conv1d(
            in_channels=1,
            out_channels=spec["J"],
            kernel_size=spec["T"],
            stride=spec["stride"],
            padding=0,
            bias=False)
        
        self.spec = spec

        self.psi.weight.data = w[:, :spec["T"]].unsqueeze(1).float()        
    
    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[-1])
        x = F.pad(x, (0, self.spec["T"]-1), mode='circular',)
        Wx = torch.abs(self.psi(x))
        # hann = torch.hann_window(spec["N"]//spec["stride"]).unsqueeze(0).unsqueeze(0)
        # phi = torch.ones(spec["J"], spec["J"], spec["N"]//spec["stride"])*hann
        # Ux = F.conv1d(Wx, phi, bias=None, stride=1, padding=0)

        Ux = F.avg_pool1d(Wx, kernel_size=self.spec["N"]//self.spec["stride"], stride=1)
        return Ux