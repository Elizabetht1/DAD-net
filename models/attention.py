import torch
from torch import nn

import os 
import math


class LearnablePosEnc(nn.Module):
    def __init__(self,series_len,embedding_dim,dropout=0.1):
        super(LearnablePosEnc,self).__init__()
        self.pos_enc = nn.Parameter(torch.empty(series_len,1,embedding_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.uniform_(self.pos_enc, -0.02, 0.02)
    
    def forward(self,x):
        '''
        x: (seq_len,batchs,channels)
        out: (seq_len,batchs,channels)
        '''
        return self.dropout(x + self.pos_enc)


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPosEnc(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.

    Args:
        series_len: the max. length of the incoming sequence
        embedding_dim: the embed dim (required).
        dropout: the dropout value (default=0.1).
    """

    def __init__(self, series_len, embedding_dim, dropout=0.1, scale_factor=1.0):
        super(FixedPosEnc, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(series_len, embedding_dim)  # positional encoding
        position = torch.arange(0, series_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_norm(is_batch):
    if is_batch:
        return nn.BatchNorm1d
    return nn.LayerNorm

def get_pos_enc(learnable):
    if learnable:
        return LearnablePosEnc
    return FixedPosEnc


class encoderMTSC(nn.Module):
    def __init__(self,d_model,heads,dropout=0.1,dim_ff=128,is_batch =True,debug=False):
        super(encoderMTSC,self).__init__()
        self.debug =True
        self.self_attn = nn.MultiheadAttention(d_model,heads,dropout=dropout,batch_first=False)
        self.norm1 = nn.BatchNorm1d(d_model,eps=1e-5)
        self.dropout1 = nn.Dropout(dropout) ##assumes attention score dimnesions are equivlanet to embedding dims
        
        self.fc1 = nn.Linear(d_model,dim_ff) 
        self.dropout2 = nn.Dropout(dropout)
       
        self.fc2 = nn.Linear(dim_ff,d_model) 
        self.norm2 = nn.BatchNorm1d(d_model,eps=1e-5)
        self.activation = nn.ReLU()
        self.displayed = False

    
    def forward(self,X,src_mask,src_key_padding_mask,is_causal):
        ''' 
        input: word embeddings with postion info. (seq_len,batch,channels)

        -  fufills interface for pytorch TransformEncoderLayer -

        src_mask: optional, default None 
        src_key_padding_mask : optional, default None 
        is_casual: optional, default None 

        '''

        ''' self attention '''
        x_pe_perm = X.permute(1,0,2) #(batch size, seq len, embed dim)
        x_attention_score,_ = self.self_attn(x_pe_perm,x_pe_perm,x_pe_perm) ##need to be of the format query key value
        
        x_attention_score = x_attention_score.permute(1,0,2) #restore to (seq_len,batch_size,embedding dim)
        x_attention_score = self.dropout1(x_attention_score)


        ''' residual connection '''
        x_rc1 = X + self.dropout1(x_attention_score) ##add in residual connection 
        x_rc1 = self.norm1(x_rc1.permute(1,2,0)) # (batch_Size,embedding_dim,seq_len)
        x_rc1 = x_rc1.permute(2,0,1) # (seq_len,batch_Size,embedding_dim)
        if self.debug and not self.displayed:
            print("rc 1 output:", x_rc1.shape)


        ''' feed forward network '''
    
        lin1_out = self.dropout2(self.activation(self.fc1(x_rc1)))
        if self.debug and not self.displayed:
            print("Fully-connected layer 1 output:", lin1_out.shape)

        lin2_out = self.fc2(lin1_out) # (seq_len,batch_size,embedding_dim)
        if self.debug and not self.displayed:
            print("Fully-connected layer 2 output:", lin2_out.shape)
            self.displayed = True
    

        '''residual connection'''
        x_rc2 = x_rc1 + self.dropout2(lin2_out)
        x_rc2 = x_rc2.permute(1,2,0) # (batch_size,embedding_dim,seq_len)
        
        x_rc2 = self.norm2(x_rc2)

        return x_rc2.permute(2,0,1)  ##(seq_len,batch_size,embedding_dim)

        

class attentionMTSC(nn.Module):
    def __init__(self,series_len,input_dim,dataset_name,learnable_pos_enc=True,d_model=128,heads=8,classes=6,dropout=0.1,dim_ff=256,is_batch =True,num_layers=1,task="classification",weights_fp="",debug=False):
        super(attentionMTSC,self).__init__()

        if len(weights_fp):
            self.load_pretrained(weights_fp)
        
        self.debug=True
        self.d_model = d_model
        self.tok_embed =  nn.Linear(input_dim, d_model)

     
        self.pos_enc = get_pos_enc(learnable_pos_enc)(series_len,d_model) #leaving dropout to 0.1
        self.task = task
    
        encoder_layer = encoderMTSC(d_model=d_model,
                                    heads=heads,
                                    dim_ff=dim_ff,
                                    is_batch=is_batch,debug=self.debug)

        ##instead of using predefined TransformerEncoderLayer, user-defined class allows us to specify batch or layer noramlization
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        
    
    
        self.dropout = nn.Dropout(dropout)

       
        self.output = nn.Linear(d_model,input_dim)

        self.activation = nn.ReLU()

        if self.debug:
            self.displayed = False

        if task == "classification":
            self.output = nn.Linear(d_model*series_len, classes)
            torch.nn.init.uniform_(self.output.weight, -0.02, 0.02)

    def load_pretrained(self, weights_fp):
        assert os.path.exists(weights_fp), \
            "No pre-trained state dictionary available"
        pre_trained_state_dict = torch.load(weights_fp)
        self.load_state_dict(pre_trained_state_dict)
        print(f"Pre-trained weights from {weights_fp} areloaded successfully")

    def forward(self, X):  # assume X is scaled
        '''
        input: X - (batch,channels,seq_len)
        '''

        X = X.to(torch.float32)

        '''   project into model dim space (embeddings)   '''
        if self.debug and not self.displayed:
            print("INPUT SHAPE:", X.shape)

        x_embed = self.tok_embed(X.permute(2,0,1)) * math.sqrt(self.d_model) ##(seq_len,batch,channels)
       
        if self.debug and not self.displayed:
            print("EMBEDDING:" , x_embed.shape)
    

        '''   positional encoding   '''
        x_pos_enc = self.pos_enc(x_embed) ##add in positional information

        if self.debug and not self.displayed:
            print("POSITONAL ENCODING:", x_pos_enc.shape)
            

        '''   attention module   '''
        attention_output = self.encoder(x_pos_enc) #(seq_len,batch_size,channels)
        attention_output = self.activation(attention_output) ##not included in user defined encoder class
        attention_output = attention_output.permute(1,0,2) #(batch_size,seq_len,channels)
        attention_output = self.dropout(attention_output)
        
        if self.debug and not self.displayed:
            print("ATTENTION SHAPE:", attention_output.shape)

        '''   output layer   '''
        if self.task == "classification":
            out = self.output(attention_output.reshape(attention_output.shape[0],-1)) ##elimates one dimenison
        else:
            out = self.output(attention_output) 
            out = out.permute(0,2,1)

        if self.debug and not self.displayed:
            print("OUTPUT SHAPE:",out.shape)
            self.displayed = True

        return out
