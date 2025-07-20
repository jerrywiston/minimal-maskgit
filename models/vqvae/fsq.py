import torch
import torch.nn as nn
from einops import rearrange, pack, unpack

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def round_ste(z):
    """Round with straight through gradients."""
    zhat = torch.round(z)
    return z + (zhat - z).detach()

class FsQuantizer(nn.Module):
    def __init__(
            self, 
            levels: list[int],
            dim = None,
            num_codebooks = 1,
            keep_num_codebooks_dim = None,
        ):
        super().__init__()

        _levels = torch.tensor(levels, dtype=torch.long)
        self.register_buffer("_levels", _levels, persistent=False)
        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.long)
        self.register_buffer("_basis", _basis, persistent=False)

        self.codebook_dim = len(levels)
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = self.codebook_dim * num_codebooks
        self.keep_num_codebooks_dim = keep_num_codebooks_dim if keep_num_codebooks_dim is not None else num_codebooks > 1
        self.dim = dim if dim is not None else len(_levels) * num_codebooks
        self.codebook_size = self._levels.prod().item()
        
        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        # Projection in and out if dim is not matched
        has_projections = self.dim != self.effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, self.effective_codebook_dim)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(self.effective_codebook_dim, self.dim)
            if has_projections
            else nn.Identity()
        )
        self.has_projections = has_projections
        
    def bound(self, z: torch.Tensor):
        """Bound 'z', an array of shape (..., d)."""
        eps = 1e-3
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 1, 0.0, 0.5)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset
    
    def quantize(self, z: torch.Tensor):
        """Quanitzes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2 # Renormalize to [-1, 1].
        return quantized / half_width
    
    def _scale_and_shift(self, zhat_normalized: torch.Tensor):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat: torch.Tensor):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width
    
    def codes_to_indices(self, zhat: torch.Tensor):
        """Converts a 'code' to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat).float()
        return (zhat * self._basis).sum(dim=-1).to(torch.long)
    
    def indices_to_codes(self, indices: torch.Tensor, project_out=True):
        """Inverse of 'indexes_to_codes'."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")
        
        if project_out:
            codes = self.project_out(codes)
        
        #is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        #if is_img_or_video:
        #    codes = rearrange(codes, "b ... d -> b d ...")
        
        return codes

    def forward(self, z: torch.Tensor):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """
        #is_img_or_video = z.ndim >= 4

        #if is_img_or_video:
        #    z = rearrange(z, "b d ... -> b ... d")
        #    z, ps = pack_one(z, "b * d")
        z, ps = pack_one(z, "b * d")

        z = self.project_in(z)
        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)
        codes = rearrange(codes, "b n c d -> b n (c d)")
        out = self.project_out(codes)

        #if is_img_or_video:
        #    out = unpack_one(out, ps, "b * d")
        #    out = rearrange(out, "b ... d -> b d ...")
        #    indices = unpack_one(indices, ps, "b * c")
        out = unpack_one(out, ps, "b * d")
        indices = unpack_one(indices, ps, "b * c")
        
        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")
        
        dummy_loss = torch.zeros(1).to(z.device)
        return out, dummy_loss, indices