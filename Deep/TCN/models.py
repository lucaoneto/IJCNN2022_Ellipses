from utilities import *


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.Tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)
        self.batch1 = nn.BatchNorm1d(n_outputs)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.Tanh2 = nn.Tanh()
        self.dropout2 = nn.Dropout(dropout)
        self.batch2 = nn.BatchNorm1d(n_outputs)
        self.max_pool = nn.MaxPool1d(2)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.Tanh1, self.dropout1,
                                 self.conv2, self.chomp2, self.Tanh2, self.dropout2, )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.Tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.Tanh(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_layers, num_filters, kernel_sizes, dropout):
        """
        @param num_layers: numbers of convolutional layers
        @param num_filters: numbers of filters for each convolutional layer
        @param kernel_sizes: size of convolutional filter for each signal
        @param dropout: dropout to be added after each tcn block
        """

        super(TemporalConvNet, self).__init__()
        layers = []

        for signal in range(2):
            tcn_blocks = []
            stride = 1
            for i in range(0, num_layers):
                dilation_size = 2 ** i
                in_channel = 1 if i == 0 else num_filters
                out_channel = num_filters

                tcn_blocks += [
                    TemporalBlock(in_channel, out_channel, kernel_sizes[signal], stride=stride, dilation=dilation_size,
                                  padding=(kernel_sizes[signal] - 1) * dilation_size, dropout=dropout)]
                stride = 1

            layers.append(tcn_blocks)

        self.feature1 = nn.Sequential(*layers[0])
        self.feature2 = nn.Sequential(*layers[1])

    def forward(self, x):
        dim = x.size(2)

        x1, x2 = x[:, 0, :].view(-1, 1, dim), x[:, 1, :].view(-1, 1, dim)
        x1 = self.feature1(x1)
        x2 = self.feature2(x2)

        return x1, x2


class TCN(nn.Module):
    def __init__(self, num_layers, num_filters, kernel_sizes, dropout, output_size=14):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(num_layers, num_filters, kernel_sizes, dropout=dropout)
        self.fc = nn.Linear(num_filters * 2, output_size)

    def forward(self, x):
        x1, x2 = self.tcn(x)

        x1 = x1[:, :, -1]
        x2 = x2[:, :, -1]

        x = torch.cat((x1.view(x1.size(0), -1),
                       x2.view(x2.size(0), -1)), dim=1)

        x = self.fc(x)
        return x


class TCN_auto(nn.Module):
    def __init__(self, num_layers, num_filters, kernel_sizes, dropout, output_size=111):
        super(TCN_auto, self).__init__()
        self.tcn = TemporalConvNet(num_layers, num_filters, kernel_sizes, dropout=dropout)
        self.fc1 = nn.Sequential(nn.Linear(num_filters, output_size))
        self.fc2 = nn.Sequential(nn.Linear(num_filters, output_size))

    def forward(self, x):
        x1, x2 = self.tcn(x)

        x1 = x1[:, :, -1]
        x2 = x2[:, :, -1]

        x1 = self.fc1(x1)
        x2 = self.fc2(x2)

        x = torch.cat((x1.view(x1.size(0), 1, -1),
                       x2.view(x2.size(0), 1, -1)), dim=1)

        return x
