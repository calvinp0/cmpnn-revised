import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru  = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, 
                           bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size), 
                                1.0 / math.sqrt(self.hidden_size))


    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        # padding
        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))
            
            cur_message = torch.nn.ZeroPad2d((0,0,0,MAX_atom_len-cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))
            
        message_lst = torch.cat(message_lst, 0)
        hidden_lst  = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2,1,1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)
        
        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2*self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)
        
        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1), 
                             cur_message_unpadding], 0)
        return message