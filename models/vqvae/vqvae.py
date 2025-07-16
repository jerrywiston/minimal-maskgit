import torch
from torch import nn, einsum
import torch.nn.functional as F
from backbone import Encoder, Decoder
from codebook import VqQuantizer, RqQuantizer

class VQVAE(nn.Module):
    def __init__(self, h_dim, n_embeddings, embedding_dim, input_channels=3):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_channels=input_channels, embedding_dim=embedding_dim, ch=h_dim)
        self.quantizer = VqQuantizer(n_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim=embedding_dim, ch=h_dim, output_channels=input_channels)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, latent_loss, ind = self.quantizer(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, latent_loss, ind
    
    def x_to_code(self, x):
        z_e = self.encoder(x)
        quant_list, codes = self.quantizer.z_to_code(z_e)
        return quant_list, codes

    def x_to_code_and_zq(self, x):
        z_e = self.encoder(x)
        quant_list, codes = self.quantizer.z_to_code(z_e)
        z_q = self.quantizer.code_to_z(codes)
        return z_q, quant_list, codes

    def code_to_x(self, codes):
        z_q = self.quantizer.code_to_z(codes)
        x_re = self.decoder(z_q)
        return x_re

class RQVAE(nn.Module):
    def __init__(self, h_dim, n_embeddings, embedding_dim, downsample_steps=2, input_channels=3, depth=4):
        super(RQVAE, self).__init__()
        self.encoder = Encoder(input_channels=input_channels, embedding_dim=embedding_dim, ch=h_dim, downsample_steps=downsample_steps)
        self.quantizer = RqQuantizer(n_embeddings, embedding_dim, depth)
        self.decoder = Decoder(embedding_dim=embedding_dim, ch=h_dim, output_channels=input_channels, downsample_steps=downsample_steps)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, latent_loss, ind = self.quantizer(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, latent_loss, ind

    def x_to_code(self, x):
        z_e = self.encoder(x)
        quant_list, codes = self.quantizer.z_to_code(z_e)
        return quant_list, codes
    
    def x_to_code_and_zq(self, x):
        z_e = self.encoder(x)
        quant_list, codes = self.quantizer.z_to_code(z_e)
        z_q = self.quantizer.code_to_z(codes)
        return z_q, quant_list, codes

    def code_to_x(self, codes):
        z_q = self.quantizer.code_to_z(codes)
        x_re = self.decoder(z_q)
        return x_re
