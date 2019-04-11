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

class MLPShallow(nn.Module):
    """
    The model architecture is a typical MLP with dense layers and ReLUs.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPShallow, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
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


class LinearModel(nn.Module):
    """
    The model architecture is a typical MLP with dense layers and ReLUs.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, output_size),
        )

    def forward(self, x):
        logits = self.model(x)
        return logits

    def save(self, fname):
        torch.save(self.state_dict(), fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname, map_location='cpu'))


class MLPDeeper(nn.Module):
    """
    The model architecture is a typical MLP with dense layers and ReLUs.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPDeeper, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
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

class MLPDropout(nn.Module):
    """
    The model architecture is a typical MLP with dense layers and ReLUs.
    """
    def __init__(self, input_size, hidden_size, output_size, p=0.5):
        super(MLPDropout, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Dropout(p),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p),
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


class MLPBatchNorm(nn.Module):
    """
    The model architecture is a typical MLP with dense layers and ReLUs.
    """
    def __init__(self, input_size, hidden_size, output_size, p=0.5):
        super(MLPBatchNorm, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        logits = self.model(x)
        return logits

    def save(self, fname):
        torch.save(self.state_dict(), fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname, map_location='cpu'))
