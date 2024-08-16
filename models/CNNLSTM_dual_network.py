import torch
from torch import nn


class CNNLSTM_dn(nn.Module):
    def __init__(self, input_dim, time_steps, num_filters, hidden_size,
                 classes):
        '''
        SIMPLE CNN ARCHITECTURE
        [INPUT - [(CONV -> RELU)*N -> POOL] -> LSTM - FC]
        '''
        super(CNNLSTM_dn, self).__init__()

        '''
        CONVLUTIONAL LAYER
        '''
        self.activation = nn.ReLU()

        self.sCNN1 = nn.Conv1d(in_channels=input_dim,
                               out_channels=num_filters,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.dropout1 = nn.Dropout(0.2)
        # NOTE : we do integer division to ensure types are correct, but if
        # STRIDE_DIM != 1, may have rounding issues

        self.gmp = nn.MaxPool1d(kernel_size=time_steps, stride=1)
        self.gap = nn.AvgPool1d(kernel_size=time_steps, stride=1)
        self.lmp = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lap = nn.AvgPool1d(kernel_size=2, stride=2)

        '''
        LSTM LAYER
        '''

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_size,
                            batch_first=True)
        self.dropout2 = nn.Dropout(0.5)

        # lstm hidden size must match num convultional filters in order to
        # ensure concatentation is posisble

        '''
        FULLY CONNECTED LAYER
        '''

        self.fc1 = nn.Linear(num_filters + hidden_size, classes)
        self.dropout3 = nn.Dropout(0.5)

        # once we flatten output, it will have the length of time series
        # (from lstm) + 1 (from pooled conv)
        # multiplied by hidden size, the nubmer of dimesnions

    def forward(self, x):
        '''run conv network '''
        conv_out = self.activation(self.bn1(self.sCNN1(x)))
        conv_out = self.dropout1(conv_out)
        conv_out = self.gmp(conv_out).squeeze()

        '''run lstm network '''
        # dim shuffling:
        # (batches, channels, timesteps) -> (batches, timesteps, channels)
        lstm_out = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(lstm_out)
        # Only use the output of the last time step
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout2(lstm_out)

        # concatenate row-by-row in each batch
        x = torch.cat((conv_out, lstm_out), dim=1)

        x = self.fc1(x)
        x = self.dropout3(x)

        return x
