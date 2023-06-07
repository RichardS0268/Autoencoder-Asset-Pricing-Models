import torch
from torch import nn
import torch.nn.functional as F
    

class CrossAttention(nn.Module):
    """
    Cross Attention posted at https://towardsdatascience.com/illustrated-difference-between-mlp-and-transformers-for-tensor-reshaping-52569edaf89
    """
    def __init__(self, input_size, embedding_size=4, dropout=0.0, device='cuda'):
        super(CrossAttention, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.device = device
        
        # Query : (batch_size, input_size, 1) -> (batch_size, input_size, embedding_size)
        # Key : (batch_size, input_size, 1) -> (batch_size, input_size, embedding_size)
        self.query = nn.Linear(1, embedding_size)
        self.key = nn.Linear(1, embedding_size)

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

        # Value : (batch_size, input_size, 1) -> (batch_size, input_size, embedding_size)
        self.value = nn.Linear(1, embedding_size)

    def forward(self, q, x):
        # print('q: ', q.shape)
        # print('x: ', x.shape)
        Q = self.query(q)
        # print('Q: ', Q.shape)
        K = self.key(x)
        # print('K: ', K.shape)
        V = self.value(x)
        # print('V: ', V.shape)


        # Q : (batch_size, input_size, embedding_size)
        # K : (batch_size, input_size, embedding_size)
        # V : (batch_size, input_size, embedding_size)
        # QK^T : (batch_size, input_size, input_size)
        # print("DEBUG:", Q.shape, K.transpose(1,2).shape)
        QK = torch.matmul(Q, K.transpose(1, 2))
        # print("matmul(Q, K.T): ", QK.shape)
        # softmax(QK^T) : (batch_size, input_size, input_size)
        QK = self.softmax(QK)
        # print("softmax: ", QK.shape)
        # QK^TV : (batch_size, input_size, embedding_size)
        QKV = torch.matmul(QK, V)
        # print("matmul(QK, V): ", QKV.shape)
        # QK^TV : (batch_size, input_size, embedding_size)
        QKV = self.dropout(QKV)
        return QKV.squeeze(1)

class UniversalCrossAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.out = nn.Linear(dim, dim)

    def forward(self, query, context):
        # b: batch_size
        # n: sequence length 
        # h: number of heads
        # d_k:  the dimension of the key vectors per attention head 
        b, n, _, h, d_k = *query.shape, self.heads, query.shape[-1] // self.heads

        # Apply linear transformation
        q = self.query(query)  # Shape: (batch_size, seq_len, dim)
        k = self.key(context)  # Shape: (batch_size, seq_len, dim)
        v = self.value(context)  # Shape: (batch_size, seq_len, dim)

        # Reshape and transpose for multi-head attention computation
        q = q.view(b, n, h, d_k).transpose(1, 2)  # Shape: (batch_size, heads, seq_len, dim_per_head)
        k = k.view(b, n, h, d_k).transpose(1, 2)  # Shape: (batch_size, heads, seq_len, dim_per_head)
        v = v.view(b, n, h, d_k).transpose(1, 2)  # Shape: (batch_size, heads, seq_len, dim_per_head)

        # Compute scaled dot-product attention
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # Shape: (batch_size, heads, seq_len, seq_len)

        # Apply softmax to get attention weights
        attn = dots.softmax(dim=-1)  # Shape: (batch_size, heads, seq_len, seq_len)

        # Compute weighted sum of values
        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # Shape: (batch_size, heads, seq_len, dim_per_head)

        # Reshape and apply final linear transformation
        out = out.transpose(1, 2).contiguous().view(b, n, -1)  # Shape: (batch_size, seq_len, dim)
        out = self.out(out)  # Shape: (batch_size, seq_len, dim)

        return out
