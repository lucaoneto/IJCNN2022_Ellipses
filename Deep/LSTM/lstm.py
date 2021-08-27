from utilities import *


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_size=14, drop=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop = drop
        self.output_size = output_size

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=drop)
        self.linear = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_size),
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)  # *2

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hidden, _) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.linear(hidden[-1])
        return out
