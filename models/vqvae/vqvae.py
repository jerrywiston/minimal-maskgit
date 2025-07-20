import torch
from torch import nn, einsum
import torch.nn.functional as F
from codebook import VqQuantizer
from fsq import FsQuantizer
from .backbone import Encoder, Decoder, EncoderAtt, DecoderAtt, EncoderSimple, DecoderSimple
import einops

class VQBase(nn.Module):
    def __init__(self, encoder, quantizer, decoder, is_img_or_video=True):
        super(VQBase, self).__init__()
        self.is_img_or_video = is_img_or_video
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
    
    def forward(self, x):
        z_e = self.encoder(x)

        if self.is_img_or_video:
            z_e = einops.rearrange(z_e, "b d ... -> b ... d")

        z_q, latent_loss, indices = self.quantizer(z_e)
        
        if self.is_img_or_video:
            z_q = einops.rearrange(z_q, "b ... d -> b d ...")
        
        x_hat = self.decoder(z_q)
        return x_hat, latent_loss, indices

    def x_to_indices(self, x, flatten=True):
        z_e = self.encoder(x)
        if self.is_img_or_video:
            z_e = einops.rearrange(z_e, "b d ... -> b ... d")

        z_q, latent_loss, indices = self.quantizer(z_e)
        if flatten:
            z_q = z_q.reshape(z_q.shape[0], -1, z_q.shape[-1])
            indices = indices.reshape(z_q.shape[0], -1)
        return z_q, indices
    
    def indices_to_x(self, indices, size=None):
        z_q = self.quantizer.indices_to_codes(indices)
        if self.is_img_or_video:
            z_q = einops.rearrange(z_q, "b ... d -> b d ...")
            if size is not None:
                z_q = z_q.reshape(z_q.shape[0], z_q.shape[1], *size)
        
        x_hat = self.decoder(z_q)
        return x_hat

class VQVaeSimple(VQBase):
    def __init__(self, h_dim, n_embeddings=1024, embedding_dim=8, input_channels=3):
        encoder = EncoderSimple(input_channels=input_channels, embedding_dim=embedding_dim, ch=h_dim)
        quantizer = VqQuantizer(n_embeddings, embedding_dim)
        decoder = DecoderSimple(embedding_dim=embedding_dim, ch=h_dim, output_channels=input_channels)
        is_img_or_video = True
        super().__init__(encoder, quantizer, decoder, is_img_or_video)

class VQVae(VQBase):
    def __init__(self, h_dim, n_embeddings=1024, embedding_dim=8, input_channels=3):
        encoder = Encoder(input_channels=input_channels, embedding_dim=embedding_dim, ch=h_dim)
        quantizer = VqQuantizer(n_embeddings, embedding_dim)
        decoder = Decoder(embedding_dim=embedding_dim, ch=h_dim, output_channels=input_channels)
        is_img_or_video = True
        super().__init__(encoder, quantizer, decoder, is_img_or_video)

class VQVaeFsqSimple(VQBase):
    def __init__(self, h_dim, levels=[8,5,5,5], embedding_dim=8, input_channels=3):
        encoder = EncoderSimple(input_channels=input_channels, embedding_dim=embedding_dim, ch=h_dim)
        quantizer = FsQuantizer(levels=levels, dim=embedding_dim)
        decoder = DecoderSimple(embedding_dim=embedding_dim, ch=h_dim, output_channels=input_channels)
        is_img_or_video = True
        super().__init__(encoder, quantizer, decoder, is_img_or_video)

class VQVaeFsq(VQBase):
    def __init__(self, h_dim, levels=[8,5,5,5], embedding_dim=8, input_channels=3):
        encoder = Encoder(input_channels=input_channels, embedding_dim=embedding_dim, ch=h_dim)
        quantizer = FsQuantizer(levels=levels, dim=embedding_dim)
        decoder = Decoder(embedding_dim=embedding_dim, ch=h_dim, output_channels=input_channels)
        is_img_or_video = True
        super().__init__(encoder, quantizer, decoder, is_img_or_video)
