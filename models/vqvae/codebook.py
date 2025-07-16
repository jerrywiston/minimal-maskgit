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
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, loss, min_encoding_indices

class RqQuantizer(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, depth=4, beta=10):
        super(RqQuantizer, self).__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.depth = depth

        self.embedding = nn.Embedding(self.n_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_embeddings, 1.0 / self.n_embeddings)

    def quantize(self, z): # z shape: (B, c, h, w)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))

        code = torch.argmin(d, dim=1)
        z_q = self.embedding(code).view(z.shape).permute(0, 3, 1, 2)
        return z_q, code

    def compute_commitment_loss(self, z, quant_list):
        """
        Compute the commitment loss for the residual quantization.
        The loss is iteratively computed by aggregating quantized features.
        """
        loss_list = []
        loss = 0.
        
        for idx, z_q in enumerate(quant_list):
            partial_loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)
            loss_list.append(partial_loss)
            loss = loss + partial_loss

        return loss 

    def z_to_code(self, z):
        residual_feature = z.detach().clone()
        quant_list = []
        code_list = []
        aggregated_quants = torch.zeros_like(z)
        for _ in range(self.depth):
            quant, code = self.quantize(residual_feature)

            residual_feature.sub_(quant)
            aggregated_quants.add_(quant)

            quant_list.append(aggregated_quants.clone())
            code_list.append(code.unsqueeze(-1))
        
        codes = torch.cat(code_list, dim=-1).reshape(-1, z.shape[2], z.shape[3], self.depth)
        return quant_list, codes

    @torch.no_grad()
    def code_to_z(self, codes): # (B, h, w, d)
        z_q_agg = self.embedding(codes) # (B, h, w, d, e)
        z_q = z_q_agg.sum(3).permute(0,3,1,2)
        return z_q
    
    def forward(self, z):
        quant_list, codes = self.z_to_code(z)
        loss = self.compute_commitment_loss(z, quant_list)
        z_q = z + (quant_list[-1] - z).detach()

        return z_q, loss, codes
