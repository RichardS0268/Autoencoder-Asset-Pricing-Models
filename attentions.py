# In the context of asset pricing, attention mechanisms can be used to identify and focus on
# the most relevant features or time periods for predicting asset prices. This can be 
# particularly useful in situations where there are many potential features that could be used
# to predict asset prices, but only a subset of them are relevant at any given time.

# **Temporal Attention**
# If your data is a time series (which is often the case in asset pricing),
# you could use an attention mechanism that focuses on different time steps. For example, you 
# could use an LSTM with an attention mechanism that allows the model to pay more attention to
# certain time steps when making its predictions.

# **Feature Attention**
# If you have many potential features and you want your model to learn which
# ones are most important, you could use an attention mechanism that focuses on different 
# features. This could be implemented as a fully connected layer that assigns an attention
# weight to each feature.

# **Self-Attention**
# The self-attention mechanism, which is used in the Transformer model, could 
# also be useful in this context. Self-attention allows the model to consider the relationships
# between all pairs of time steps or features, which could help it identify complex patterns in 
# the data.

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_weights = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x):
        # x should be a 3D tensor with shape (batch_size, sequence_length, hidden_dim)
        batch_size, sequence_length, _ = x.size()

        # Compute the attention scores
        attention_scores = torch.matmul(x, self.attention_weights)
        # attention_scores should have shape (batch_size, sequence_length)

        # Apply softmax to get the attention distribution
        attention_distribution = F.softmax(attention_scores, dim=1)
        # attention_distribution should have shape (batch_size, sequence_length)

        # Compute the weighted sum of the inputs
        weighted_sum = torch.sum(x * attention_distribution.unsqueeze(-1), dim=1)
        # weighted_sum should have shape (batch_size, hidden_dim)

        return weighted_sum

class FeatureAttention(nn.Module):
    def __init__(self, input_dim):
        super(FeatureAttention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        # Apply softmax to the attention weights
        attention_weights = F.softmax(self.attention_weights, dim=0)

        # Multiply each feature by its attention weight
        x = x * attention_weights

        return x

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query_weight = nn.Parameter(torch.randn(hidden_dim))
        self.key_weight = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x):
        # x should be a 3D tensor with shape (batch_size, sequence_length, hidden_dim)
        batch_size, sequence_length, _ = x.size()

        # Compute the query and key
        query = torch.matmul(x, self.query_weight)
        key = torch.matmul(x, self.key_weight)
        # query and key should have shape (batch_size, sequence_length)

        # Compute the attention scores
        attention_scores = torch.bmm(query.unsqueeze(2), key.unsqueeze(1))
        # attention_scores should have shape (batch_size, sequence_length, sequence_length)

        # Apply softmax to get the attention distribution
        attention_distribution = F.softmax(attention_scores, dim=-1)
        # attention_distribution should have shape (batch_size, sequence_length, sequence_length)

        # Compute the weighted sum of the inputs
        weighted_sum = torch.bmm(attention_distribution, x)
        # weighted_sum should have shape (batch_size, sequence_length, hidden_dim)

        return weighted_sum
