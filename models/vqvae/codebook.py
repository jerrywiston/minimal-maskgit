"""
taken from: https://github.com/dome272/VQGAN-pytorch/blob/main/codebook.py
"""

import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch
import torch.nn as nn

class VqQuantizer(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, beta=10):
        super(VqQuantizer, self).__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_embeddings, 1.0 / self.n_embeddings)

    def forward(self, z):
        """
        Input:
            z (torch.FloatTensor): [batch_size, ..., embedding_dim]
        Output:
            z_q (torch.FloatTensor): [batch_size, ..., embedding_dim]
            loss (torch.FloatTensor)
            indices (torch.FloatTensor): [batch_size, indices_size]
        """
        #z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.reshape(-1, self.embedding_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))

        indices = torch.argmin(d, dim=-1)
        indices = indices.reshape(z.shape[:-1])
        z_q = self.embedding(indices).reshape(z.shape)
        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)
        z_q = z + (z_q - z).detach()

        #z_q = z_q.permute(0, 3, 1, 2)
        return z_q, loss, indices

    def indices_to_codes(self, indices):
        return self.embedding(indices)
