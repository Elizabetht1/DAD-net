import torch
from torch import nn


def get_pool(avg = True):
    '''
    avg = True -> Average Pooling, avg = FAlse -> Max pooling
    '''
    if avg:
        return nn.AvgPool1d
    return nn.MaxPool1d
    


class conv_block(nn.Module):
    def __init__(self, in_channels,out_channels,pool_dim,groups,dropout=0.2,momentum=0.99,eps = 1e-3,average_pooling=True):
        super(conv_block,self).__init__()

        self.activation = nn.ReLU()

        self.CNN = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=5,
                               stride=1,
                               padding=0,
                               groups=groups) 
        
        self.bn = nn.BatchNorm1d(out_channels, momentum=momentum, eps=eps)
        self.dropout = nn.Dropout(dropout)

        self.pool = get_pool(average_pooling)(pool_dim,stride=pool_dim) 
        # if pool_dim = out_channels, it is max pooling
        # stride = pool_dim to ensure no overlap


    
    def forward(self,X):
        ''' 
        input: X - (batch,C,L) a univariate series / feature maps derived from univariate 
        output: 
        - conv_out –> (batch,out_channels,(L-K)+1 ) 
        - pool_out –> (batch,out_channels,(L-pool_dim)/2+1)  [length is 1 if global]

        '''

        conv_out = self.activation(self.bn(self.CNN(X))) ##Filter Layer & Activiation Layer
        conv_out = self.dropout(conv_out) ##drouput
        pool_out = self.pool(conv_out) ##pooling layer

        return pool_out





class CNN_cw(nn.Module):
    def __init__(self, input_dim, time_steps, classes, num_filters1 = 8, num_filters2=4, pool1=2,pool2=2,dim_ff=64):

        super(CNN_cw,self).__init__()
        '''
        CNN Channel Wise ARCHITECTURE
        [INPUT -> [ ( CONV -> RELU -> POOL) * 2 ] x Channels -> MLP ]
        '''

        '''
        CONVLUTIONAL LAYER
        '''
        self.activation = nn.ReLU()

        # input [nBatch, nChannels, time_steps]
        self.sCNN1 = conv_block(in_channels=input_dim,
                                out_channels = input_dim*num_filters1,
                                pool_dim = pool1,
                                groups = input_dim) ##perform a depthwise convultion using the torch groups param


                            
        # channels are given by filters added in conv layer
        self.sCNN2 = conv_block(in_channels=num_filters1*input_dim,
                                out_channels = num_filters2*input_dim,
                                pool_dim = pool2,
                                groups = input_dim)
    
        self.dropout1 = nn.Dropout(0.2)

        # since this is a multivariate time sereis, need to flatten
        self.flat = nn.Flatten()

        '''
        MLP 
        '''
        ##result of flatting (batch,C,L) to (batch,C*L) per dimension 
        self.per_channel_dim = int((((time_steps - 4 - pool1) \
                                   / 2 +1) \
                                    -4 - pool2) \
                                   /2 +1) \
                                   * num_filters2
        

        ##length of input after concatenating dimensions 
        self.dim_after_conv = int(self.per_channel_dim*input_dim)


        self.fc1 = nn.Linear(self.dim_after_conv, dim_ff)
        self.fc2 = nn.Linear(dim_ff, classes)
        self.dropout2 = nn.Dropout(0.5)


    def forward(self, X):

        ''' 
        input: X - (batch,C,L) a multivariate series with C channels 
        '''

        batch_size,dim,_ = X.shape
      
        '''first convultional layer'''
        out = self.sCNN1(X)
            
        '''second convultional layer'''
        out = self.sCNN2(out) ##(batch, out_filters, mod_seq_len)
    
        
        out = self.flat(out) ##(batch, out_filters * mod_seq_len)

        '''
        MLP
        '''

        out = self.fc2(self.fc1(out))
        out = self.dropout2(out)

        return out
