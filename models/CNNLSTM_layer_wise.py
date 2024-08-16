from torch import nn


class CNNLSTM_lw(nn.Module):
    def __init__(self, input_dim, num_filters, hidden_size, classes):
        '''
        SIMPLE CNN ARCHITECTURE
        [INPUT - [(CONV -> RELU)*N -> POOL] -> LSTM - FC]

        '''
        super(CNNLSTM_lw, self).__init__()

        '''
        CONVLUTIONAL LAYER
        '''
        self.activation = nn.ReLU()

        self.sCNN1 = nn.Conv1d(in_channels=input_dim,
                               out_channels=num_filters,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters, momentum=0.99, eps=1e-3)
        self.dropout1 = nn.Dropout(0.2)

        self.sCNN2 = nn.Conv1d(in_channels=num_filters,
                               out_channels=num_filters,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm1d(num_filters, momentum=0.99, eps=1e-3)
        self.dropout2 = nn.Dropout(0.2)

        # NOTE : we do integer division to ensure types are correct, but if
        # STRIDE_DIM != 1, may have rounding issues

        # need to distinguish b/w global vs local pooling
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        '''
        LSTM LAYER
        '''

        self.lstm = nn.LSTM(input_size=num_filters,
                            hidden_size=hidden_size,
                            batch_first=True)
        self.dropout3 = nn.Dropout(0.5)

        '''
        FULLY CONNECTED LAYER
        '''
        self.fc1 = nn.Linear(hidden_size, classes)

    def forward(self, x):
        '''
        input: (num_batches, input_dim, time_steps)
        '''

        '''first convulational layer'''
        x = self.activation(self.bn1(self.sCNN1(x)))
        x = self.dropout1(x)

        '''second convultional layer'''
        x = self.activation(self.bn2(self.sCNN2(x)))
        x = self.dropout2(x)

        '''local pooling layer'''
        x = self.pool(x)

        # Reshape for LSTM layer
        x = x.permute(0, 2, 1)  # to (batch_size, time_steps, num_filters)

        '''lstm layer'''
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Only use the output of the last time step

        '''fully connected layer'''
        x = self.fc1(x)
        x = self.dropout3(x)

        return x
