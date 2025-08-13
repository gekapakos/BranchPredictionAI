import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.utils.data import DataLoader
from math import ceil, floor

## GPU/CPU ##
device_idx = 0
device = torch.device("cuda:"+str(device_idx) if torch.cuda.is_available() else "cpu")

# class that encodes n LSB PC bits with Branch Path
class Encoder():
    def __init__(self, pc_bits=7, concatenate_dir=False):
        self.pc_bits = pc_bits
        self.concatenate_dir = concatenate_dir
    
    def encode(self, pc, direction):
        pc = int(pc)
        pc = (pc & (2**self.pc_bits - 1))<< 1
        if(self.concatenate_dir):
            return pc+int(direction)
        else:
            return pc

    

class TwoLayerFullPrecisionBPCNN(nn.Module):
    def __init__(self, tableSize=256, numFilters=2, history_length=200):
        super().__init__()
        
        #embed via identity matrix:
        self.E = torch.eye(tableSize, dtype=torch.float32).to(device)
        self.Embedding = nn.Embedding(2**8, tableSize)
        self.EncoderLayer = nn.TransformerEncoderLayer(d_model=tableSize, nhead=8,dim_feedforward=512, dropout=0.2)        
        #convolution layer, 1D in effect; Linear Layer
        self.c1 = nn.Conv2d(tableSize, numFilters, (1,1))        
        self.tahn = nn.Tanh()
        self.l2 = nn.Linear(history_length*numFilters, 1)        
        self.sigmoid2 = nn.Tanh()        

    def forward(self, seq):
        #Embed by expanding sequence of ((IP << 7) + dir ) & (255)
        # integers into 1-hot history matrix during training
        
        #xFull = self.E[seq.data.type(torch.long)]
        #xFull = self.E[seq.data.long()]        
        xFull = self.Embedding(seq)        
        xFull = self.EncoderLayer(xFull)        
        xFull = torch.unsqueeze(xFull, 1)        
        xFull = xFull.permute(0,3,1,2)#.to(device)
        #xFull = xFull.reshape(len(xFull),16,16,200).to(device)
        
        h1 = self.c1(xFull)
        h1a = self.tahn(h1)        
        h1a = h1a.reshape(h1a.size(0),h1a.size(1)*h1a.size(3))
        out = self.l2(h1a)        
        out = self.sigmoid2(out)
        return out.squeeze().squeeze()

def init_lstm(lstm, lstm_hidden_size, forget_bias=2):
    for name,weights in lstm.named_parameters():
        if "bias_hh" in name:
            #weights are initialized 
            #(b_hi|b_hf|b_hg|b_ho), 
            weights[lstm_hidden_size:lstm_hidden_size*2].data.fill_(forget_bias)
        elif 'bias_ih' in name:
            #(b_ii|b_if|b_ig|b_io)
            pass
        elif "weight_hh" in name:
            torch.nn.init.orthogonal_(weights)
        elif 'weight_ih' in name:
            torch.nn.init.xavier_normal_(weights)
    return lstm


class RNNLayer(nn.Module):
    def __init__(self,  input_dim, out_dim, num_layers, pc_hash_bits=12, init_weights = True, batch_first=True, rnn_layer = 'gru', normalization = False, history_length=582):
        super().__init__()        

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.rnn_layer = rnn_layer
        self.pc_hash_bits = pc_hash_bits
        self.history_length = history_length
        
        self.embedding1 = nn.Embedding(2**self.pc_hash_bits+1,self.input_dim)
        #self.embedding2 = nn.Embedding(2,64)

        if rnn_layer == 'lstm':
            self.conv1 = nn.Conv1d(in_channels=self.input_dim, out_channels=64, kernel_size=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
            self.relu2 = nn.ReLU()
            self.conv3 = nn.Conv1d(in_channels=128, out_channels=self.out_dim, kernel_size=1)                        
            self.tanh = nn.ReLU()
            self.rnn = nn.LSTM(input_size = self.out_dim, hidden_size=512,\
                          num_layers= self.num_layers, 
                          batch_first=self.batch_first,                          
                          bidirectional=False)
            self.fc1 = nn.Linear(512, 128)
            self.relufc = nn.ReLU()
            self.fc2 = nn.Linear(128, 1)
            self.finalActivation = nn.Tanh()
        elif rnn_layer == 'gru':
            self.rnn = nn.GRU(input_size= self.input_dim, hidden_size=self.out_dim,\
                          num_layers= self.num_layers, batch_first=self.batch_first)
        elif rnn_layer =='transformer':
            #self.inpLinear = nn.Linear(2, self.input_dim)
            EncoderLayer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=self.out_dim,dim_feedforward=2048)
            self.rnn = nn.TransformerEncoder(EncoderLayer,self.num_layers)            
            #self.rnn = nn.Transformer(d_model=self.input_dim, nhead=self.out_dim, num_encoder_layers=self.num_layers, num_decoder_layers=self.num_layers, dim_feedforward=512, dropout=0.5)
            self.fc = nn.Linear(input_dim*200, 2) # nn.Linear(out_dim*200, 2)
        else:
            raise NotImplementedError(rnn_layer)
        
        if init_weights:
            self.rnn = init_lstm(self.rnn, self.out_dim)

        if (normalization):
            self.normalization = nn.BatchNorm1d(self.input_dim*200)
        else:
            self.normalization = None
        
       # self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, seq):
        
        #data = self.E[seq.data.type(torch.long)]
        #seq = data.to(torch.device("cuda:0"))
        #lay1=self.embedding1(seq.data[:,:,0].type(torch.long))
        #lay2=self.embedding2(seq.data[:,:,1].type(torch.long))
        #seq = torch.cat((lay1,lay2),2)

        if self.rnn_layer == 'transformer':
            #for transformer change batch first
            seq = seq.unsqueeze(dim=2)                      
            seq = seq.reshape(seq.size(1), seq.size(0), seq.size(2))            
            #OneHotVector = self.E[seq.data.long()].squeeze().to(device)
            EmbeddingVector = self.embedding1(seq.long()).squeeze()
            #seq = self.inpLinear(seq)
            self.rnn_out = self.rnn(EmbeddingVector)
            # self.rnn_out = self.rnn_out[:,-1,:]  
            #place again batch size first
            self.rnn_out = self.rnn_out.reshape(self.rnn_out.size(1),self.rnn_out.size(0)*self.rnn_out.size(2))                        
        elif self.rnn_layer =='lstm':               
            # seq = seq.unsqueeze(dim=2)             
            x = self.embedding1(seq.long())
            # print(x.shape)
            x = x.transpose(1,2)
            # print(x.shape)
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            x = self.tanh(self.conv3(x))            
            # print(x.shape)
            x=x.squeeze(dim=2)
            # print(x.shape)
            x = x.transpose(1,2)
            _, (x, _) = self.rnn(x)
            # print(x.shape)
            x = x.transpose(0,1)
            x = x.flatten(1)
            # print(x.shape)
            x = self.relufc(self.fc1(x))
            # print(x.shape)
            x = self.fc2(x)
            # print(x.shape)
            return self.finalActivation(x).squeeze(dim=1)

        elif self.rnn_layer == 'gru': 
            self.rnn_out, self.h = self.rnn(seq)

        if self.normalization:
            self.rnn_out = self.normalization(self.rnn_out)

        out = self.fc(self.rnn_out)                   
        
        return self.softmax(out)

HISTORY_LENGTHS = [42, 78, 150, 294, 582]
CONV_FILTERS    = [32, 32, 32, 32, 32]
CONV_WIDTHS     = [7, 7, 7, 7, 7]
POOL_WIDTHS     = [3, 6, 12, 24, 48]

PC_HASH_BITS    = 12            # bits of hashed PC value
HASH_DIR_WITH_PC = True         # include branch direction bit
EMBED_DIM       = 32

USE_LSTM        = True
LSTM_HIDDEN     = 128
BIDIRECTIONAL   = True

# MLP
HIDDEN_NEURONS  = [128, 128]

# --------------------------
# Helper
# --------------------------

def hash_width():
    """Return number of distinct indices for the embedding table."""
    total_bits = PC_HASH_BITS + (1 if HASH_DIR_WITH_PC else 0)
    return 1 << total_bits  # 2^(bits)

# --------------------------
# Slice module
# --------------------------

class Kladi(nn.Module):
    """One history slice → embedding → Conv → BN/ReLU → Pool → (optional Bi‑LSTM)."""
    def __init__(self, history_len:int, conv_width:int, pool_width:int, n_filters:int):
        super().__init__()

        self.embedding = nn.Embedding(hash_width(), EMBED_DIM)
        self.conv      = nn.Conv1d(EMBED_DIM, n_filters, conv_width)
        self.bn_conv   = nn.BatchNorm1d(n_filters)
        self.act       = nn.ReLU(inplace=True)
        self.pool      = nn.AvgPool1d(pool_width, stride=pool_width)
        self.bn_pool   = nn.BatchNorm1d(n_filters)  # "bn_only" activation in original cfg

        self.use_lstm  = USE_LSTM
        if self.use_lstm:
            self.bilstm = nn.LSTM(
                input_size=n_filters,
                hidden_size=LSTM_HIDDEN,
                num_layers=1,
                bidirectional=BIDIRECTIONAL,
                batch_first=True,
            )
            lstm_out_size = LSTM_HIDDEN * (2 if BIDIRECTIONAL else 1)
            self.out_dim  = lstm_out_size
        else:
            # After pooling, sequence length = history_len/ pool_width (integer division)
            self.out_dim = n_filters

    def forward(self, x:torch.Tensor):
        """
        x  : (batch, time)  — sequence of hashed PCs
        out: (batch, out_dim)
        """
        x = self.embedding(x)                 # (B,T,EMBED)
        x = x.transpose(1,2)                  # (B,EMBED,T)
        x = self.act(self.bn_conv(self.conv(x)))
        x = self.pool(x)                      # (B, C, T_out)
        x = self.bn_pool(x)

        if self.use_lstm:
            x = x.permute(0,2,1)              # (B,T_out,C)
            _, (h, _) = self.bilstm(x)        # h: (dir*layers,B,H)
            if BIDIRECTIONAL:
                h = torch.cat([h[0],h[1]], dim=1)  # (B,2H)
            else:
                h = h.squeeze(0)                  # (B,H)
            return h
        else:
            # Global sum (similar to original sum‑pooling when pool_width == -1)
            x = x.sum(dim=2)
            return x            # (B,C)

# --------------------------
# BranchNet main
# --------------------------

class KladosNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.slices = nn.ModuleList([
            Kladi(h, w, p, f)
            for h, w, p, f in zip(HISTORY_LENGTHS, CONV_WIDTHS, POOL_WIDTHS, CONV_FILTERS)
        ])
        flattened = sum(s.out_dim for s in self.slices)

        layers = []
        in_dim = flattened
        for hid in HIDDEN_NEURONS:
            layers += [nn.Linear(in_dim, hid), nn.BatchNorm1d(hid), nn.ReLU(inplace=True)]
            in_dim = hid
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    @torch.no_grad()
    def extract_history_slice(self, hist:torch.Tensor, length:int):
        """Take the last *length* entries from global history."""
        return hist[:, -length:]

    def forward(self, history:torch.Tensor):
        """
        history: (batch, total_history_len)  hashed PC values (ints)
        returns: (batch,)  probability/logit of branch taken
        """
        slice_outputs = []
        for slice_mod, length in zip(self.slices, HISTORY_LENGTHS):
            slice_hist = self.extract_history_slice(history, length)
            slice_outputs.append(slice_mod(slice_hist))
        x = torch.cat(slice_outputs, dim=1)
        return self.mlp(x).squeeze(1)