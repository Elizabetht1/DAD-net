from torch import nn


class simpleCNN(nn.Module):
    def __init__(self, input_dim, time_steps, num_filters, classes):
        '''
        SIMPLE CNN ARCHITECTURE
        [INPUT - [(CONV -> RELU)*1 -> POOL] - FC]
        '''
        super(simpleCNN, self).__init__()

        '''
        CONVLUTIONAL LAYER
        '''
        self.activation = nn.ReLU()

        # input [nBatch, nChannels, time_steps]
        self.sCNN1 = nn.Conv1d(in_channels=input_dim,
                               out_channels=num_filters,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        # channels are given by filters added in conv layer
        self.bn1 = nn.BatchNorm1d(num_filters, momentum=0.99, eps=1e-3)
        self.dropout1 = nn.Dropout(0.2)

        self.sCNN2 = nn.Conv1d(in_channels=num_filters,
                               out_channels=num_filters,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm1d(num_filters, momentum=0.99, eps=1e-3)
        self.dropout2 = nn.Dropout(0.2)

        # need to distinguish b/w global vs local pooling vs. average vs. max
        self.pool = nn.AvgPool1d(time_steps)

        # since this is a multivariate time sereis, need to flatten
        self.flat = nn.Flatten()

        '''
        FULLY CONNECTED LAYER
        '''
        self.fc = nn.Linear(num_filters, classes)
        self.dropout3 = nn.Dropout(0.5)

    def forward(self, x):
      
        '''first convultional layer'''
        x = self.activation(self.bn1(self.sCNN1(x)))
        x = self.dropout1(x)

        '''second convultional layer'''
        x = self.activation(self.bn2(self.sCNN2(x)))
        x = self.dropout2(x)

        x = self.pool(x)
        x = self.flat(x)

        '''fully connected layer'''
        x = self.fc(x)
        x = self.dropout3(x)

        return x
