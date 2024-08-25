import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50
import copy


# 2. BYOL Network Architecture
class MLPHead(nn.Module):
    def __init__(self, in_channels, hidden_size=4096, projection_size=128):
        super(MLPHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x):
        return self.net(x)


class EncoderNetwork(nn.Module):
    def __init__(self):
        super(EncoderNetwork, self).__init__()
        resnet = resnet50(pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projector = MLPHead(2048)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projector(h)


class BYOLNetwork(nn.Module):
    def __init__(self, projection_size=128):
        super(BYOLNetwork, self).__init__()
        self.online_encoder = EncoderNetwork()
        self.target_encoder = None
        self.predictor = MLPHead(
            projection_size, hidden_size=4096, projection_size=projection_size
        )
        self._init_target_encoder()

    def _init_target_encoder(self):
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(self, x1, x2):
        online_proj_1 = self.predictor(self.online_encoder(x1))
        online_proj_2 = self.predictor(self.online_encoder(x2))

        with torch.no_grad():
            target_proj_1 = self.target_encoder(x1)
            target_proj_2 = self.target_encoder(x2)

        return (
            online_proj_1,
            online_proj_2,
            target_proj_1.detach(),
            target_proj_2.detach(),
        )
