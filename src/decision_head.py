import torch
import torch.nn as nn

class DecisionHead(nn.Module):
    def __init__(self, input_length: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_length, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def get_early_latent(self, x):
        return self.layers[0](x)

    def get_latent(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
        return x

    def forward(self, x):
        z: torch.Tensor = self.layers(x)
        return z.squeeze()
