import math
import numpy as np
import torch
import torch.nn as nn
from vqvae.vqvae import VQVAE

class Bert(nn.Module):
    def __init__(self, vocab_size, additional_vocab_size, embedding_dim, nhead=8, num_layers=6, max_block_size=257):
        super().__init__()
        self.vocab_size = vocab_size
        self.additional_vocab_size = additional_vocab_size # [SOS] and [Mask] token
        self.embedding_dim = embedding_dim
        self.max_block_size = max_block_size # Image tokens and 1 SOS token

        EncBlock = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True, norm_first=True)
        self.wte = nn.Embedding(vocab_size + additional_vocab_size, embedding_dim)
        self.wpe = nn.Embedding(max_block_size, embedding_dim) 
        self.transformer = nn.TransformerEncoder(EncBlock, num_layers=num_layers)
        self.proj_out = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, x):
        device = x.device
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.wte(x) + self.wpe(positions)
        x = self.transformer(x)
        logits = self.proj_out(x)
        return logits

# Masked Visual Token Modeling (MVTM) for MaskGIT
class Mvtm(nn.Module):
    def __init__(self, transformer_config, vqmodel, vq_size=(16,16), n_embeddings=256, embedding_dim=3):
        super().__init__()
        self.vqmodel = vqmodel.eval()

        self.sos_token_idx = transformer_config["vocab_size"]  # Assuming the SOS token is the last index in the vocabulary
        self.mask_token_idx = transformer_config["vocab_size"] + 1  # Assuming the mask token is the second last index in the vocabulary
        self.transformer_config = transformer_config
        self.transformer = Bert(transformer_config["vocab_size"], 
                                transformer_config["additional_vocab_size"],
                                transformer_config["n_embd"],
                                nhead=transformer_config["n_head"],
                                num_layers=transformer_config["n_layer"],
                                max_block_size=transformer_config["block_size"])
        
        self.vq_size = vq_size
        self.num_image_tokens = vq_size[0] * vq_size[1]
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
    
    def gamma(self, r):
        return np.cos(r * np.pi / 2) # Use cosine function for best performacne reference to the original MaskGIT paper
        #return 1 - r # Linear function
    
    @torch.no_grad()
    def encode_to_z(self, x):
        z_e = self.vqmodel.encoder(x)
        z_q, _, indices = self.vqmodel.quantizer(z_e)
        indices = indices.view(z_q.shape[0], -1)
        return z_q, indices
    
    @torch.no_grad()
    def z_to_image(self, indices):
        ix_to_vectors = self.vqmodel.quantizer.embedding(indices)
        ix_to_vectors= ix_to_vectors.reshape(indices.shape[0], self.vq_size[0], self.vq_size[1], self.embedding_dim)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqmodel.decoder(ix_to_vectors)
        return image

    def _construct_tokens(self, batch_size, token_idx, num_tokens=1):
        tokens = torch.ones(batch_size, num_tokens, dtype=torch.long, device=self.device) 
        tokens = tokens * token_idx
        return tokens
    
    def _get_training_mask(self, batch_size, device):
        r = math.floor(self.gamma(np.random.uniform()) * self.num_image_tokens)
        if r == 0:
            r = 1 # Make sure at least one token is masked
        sample = torch.rand(batch_size, self.num_image_tokens, device=self.device).topk(r, dim=1).indices
        mask = torch.zeros(batch_size, self.num_image_tokens, dtype=torch.bool, device=device)
        mask.scatter_(dim=1, index=sample, value=True)
        return mask

    def forward(self, x, mask=None, only_mask_token_loss=True):
        self.device = next(self.transformer.parameters()).device
        if mask is None:
            mask = self._get_training_mask(x.shape[0], self.device)

        _, indices = self.encode_to_z(x)

        mask_token = self._construct_tokens(indices.shape[0], self.mask_token_idx, num_tokens=self.num_image_tokens)
        indices_masked = mask * indices + (~mask) * mask_token
        indices_sos = self._construct_tokens(x.shape[0], self.sos_token_idx)
        indices_masked = torch.cat((indices_sos, indices_masked), dim=1)
        logits = self.transformer(indices_masked)[:,1:] # Get results without SOS token

        if only_mask_token_loss: # Only compute loss for masked tokens
            mask = mask.reshape(-1)
            loss = nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1))[~mask], indices.reshape(-1)[~mask])
        else:
            loss = nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), indices.reshape(-1))

        return logits, loss

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample(self, batch_size=64, temperature=1.0, top_k=100, n_steps=10):
        self.transformer.eval()
        self.device = next(self.transformer.parameters()).device
        indices_curr = self._construct_tokens(batch_size, self.mask_token_idx, num_tokens=self.num_image_tokens)
        indices_sos = self._construct_tokens(batch_size, self.sos_token_idx)
        num_masked_tokens_last = self.num_image_tokens
        
        self.gumbel = torch.distributions.Gumbel(0, 1) # Gumbel distribution for sampling
        self.base_gumbel_temp = 4.5

        for step in range(n_steps):
            # Compute num of tokens to unmask
            num_masked_tokens_curr = math.floor(self.gamma((step+1) / n_steps) * self.num_image_tokens)
            num_unmask = num_masked_tokens_last - num_masked_tokens_curr
            num_masked_tokens_last = num_masked_tokens_curr

            # Get Logits
            indices_input = torch.cat((indices_sos, indices_curr), dim=1)
            logits_raw = self.transformer(indices_input)[:, 1:]  # [batch_size, seq_len, vocab_size]
            probs = nn.Softmax(dim=-1)(logits_raw)
            logits = self.top_k_logits(logits_raw, top_k)
            logits = logits_raw / temperature
            
            # Sample and compute scores
            indices_sampled = torch.distributions.categorical.Categorical(logits=logits).sample()  # [batch_size, seq_len]
            randomness = self.gumbel.sample(probs.shape).to(self.device)
            gumbel_temp = self.base_gumbel_temp * (1 - (step + 1) / n_steps) # Heursitic part
            confidence = torch.log(probs) + gumbel_temp * randomness
            scores = torch.gather(confidence, 2, indices_sampled.unsqueeze(2)).reshape(batch_size, -1) # [batch_size, seq_len]
            unknown_map = (indices_curr == self.mask_token_idx)
            scores = torch.where(unknown_map, scores, torch.ones_like(scores) * -float("inf")) # Masking the knwon tokens
            
            # Select the indices with highest scores to unmask
            _, unmask_idx = torch.topk(scores, num_unmask, dim=1, largest=True, sorted=False)
            values_to_scatter = torch.gather(input=indices_sampled, dim=1, index=unmask_idx) # Gather the sampled indices
            indices_curr = indices_curr.scatter(dim=1, index=unmask_idx, src=values_to_scatter) # Fill to the indices_curr

        x = self.z_to_image(indices_curr)
        self.transformer.train()
        return x
