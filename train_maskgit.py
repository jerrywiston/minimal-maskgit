import os
from tqdm import tqdm
import torch
import torchvision.utils as vutils

from models.vqvae import vqvae
from models.transformer.maskgit import Mvtm
from dataset import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Environment
dataset, dataloader = load_dataset(image_size=64, batch_size=64, dataroot="datasets/celeba_hq")

# Create Model
config_vq = {
    "h_dim": 128,
    "n_embeddings": 1024, 
    "embedding_dim": 8, 
    "input_channels": 3, # RGB input
}
vqnet = vqvae.VQVae(**config_vq).to(device)
vqmodel_path = "vqvae.pt"

transformer_config = {
    "vocab_size": 1024, 
    "additional_vocab_size": 2,  # [SOS] / [MASK] token 
    "block_size": 257, 
    "n_layer": 8,
    "n_head": 16,
    "n_embd": 512,
    "is_causal": False,
    "use_condition": False
}
model_name = "maskgit_vqvae"

""" Use Finite Scalar Quantization
config_fsq = {
    "h_dim": 128,
    "levels": [8,5,5,5], 
    "embedding_dim": 8, 
    "input_channels": 3, # RGB input
}
vqnet = vqvae.VQVaeFsq(**config_fsq).to(device)
vqmodel_path = "fsqvae.pt"

transformer_config = {
    "vocab_size": 1000, # 8*5*5*5
    "additional_vocab_size": 2,  # [SOS] / [MASK] token 
    "block_size": 257, 
    "n_layer": 8,
    "n_head": 16,
    "n_embd": 512,
    "is_causal": False,
    "use_condition": False
}
model_name = "maskgit_fsqvae"
"""

print("Loading VQ model from:", os.path.join("checkpoints", vqmodel_path))
vqnet.load_state_dict(torch.load(os.path.join("checkpoints",vqmodel_path)))
mvtm = Mvtm(vqnet, [16, 16], transformer_config).to(device)

# Create Optimizer
optimizer = torch.optim.Adam(mvtm.parameters(), lr=1e-4, amsgrad=True)

# Set Parameters of Experiment
save_path = "checkpoints"
exp_path = "experiments"
results_path = os.path.join(exp_path, model_name)

if not os.path.exists(exp_path):
    os.makedirs(exp_path)
if not os.path.exists(results_path):
    os.mkdir(os.path.join(results_path))
if not os.path.exists(save_path):
    os.mkdir(save_path)

# Load trained weight
if os.path.exists(os.path.join(save_path, model_name+".pt")):
    print("Load trained weights ...")
    mvtm.load_state_dict(torch.load(os.path.join(save_path, model_name+".pt")))

# Training Iteration
for epoch in range(1, 100):
    iter = 0
    for x, _ in tqdm(dataloader):
        optimizer.zero_grad()

        x = x.to(device)
        logits, loss = mvtm(x)
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            print("Epoch " + str(epoch).zfill(4) + ", Iter " + str(iter).zfill(4) +\
                " | ce_loss: " + str(loss.item()))

            # Generate
            with torch.no_grad():
                x_fig = mvtm.sample(batch_size=32)
                path = os.path.join(results_path, str(epoch).zfill(4) + "_" + str(iter).zfill(4)+".jpg")
                vutils.save_image(x_fig, path, padding=2, normalize=False)

                # Save model
                torch.save(mvtm.state_dict(), os.path.join(save_path, model_name+".pt"))
        
        iter += 1      