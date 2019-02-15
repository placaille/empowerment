import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    The model architecture is a typical MLP with dense layers and ReLUs.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        logits = self.model(x)
        return logits

    def save(self, fname):
        torch.save(self.state_dict(), fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname, map_location='cpu'))
