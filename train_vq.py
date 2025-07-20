import os
import torch
import torch.optim as optim
import torchvision.utils as vutils

from models.vqvae import losses, vqvae
from dataset import load_dataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Environment
dataset, dataloader = load_dataset(image_size=64, batch_size=64, dataroot="datasets/celeba_hq")

# Model Parameters
config_vq = {
    "h_dim": 128,
    "n_embeddings": 1024, 
    "embedding_dim": 8, 
    "input_channels": 3, # RGB input
}

vqnet = vqvae.VQVae(**config_vq).to(device)
""" Use Finite Scalar Quantization
config_fsq = {
    "h_dim": 128,
    "levels": [8,5,5,5], 
    "embedding_dim": 8, 
    "input_channels": 3, # RGB input
}
vqnet = vqvae.VQVaeFsq(**config_fsq).to(device)
"""

vggnet = losses.Vgg16().to(device)
disc = losses.Discriminator().to(device)
optimizer = optim.Adam(vqnet.parameters(), lr=1e-4, amsgrad=True)
optimizer_disc = optim.Adam(disc.parameters(), lr=1e-4, amsgrad=True)

save_path = "checkpoints"
exp_path = "experiments"
model_name = "vqvae"
results_path = os.path.join(exp_path, model_name)

if not os.path.exists(exp_path):
    os.makedirs(exp_path)
if not os.path.exists(results_path):
    os.mkdir(os.path.join(results_path))
if not os.path.exists(save_path):
    os.mkdir(save_path)

# Training Iteration
for epoch in tqdm(range(1, 100)):
    iter = 0
    for x, _ in tqdm(dataloader):
        optimizer.zero_grad()
        optimizer_disc.zero_grad()

        x = x.to(device)
        x_hat, latent_loss, ind = vqnet(x)
        recon_loss = torch.mean((x_hat - x)**2)
        perceptual_loss = losses.perceptual_loss(vggnet, x_hat, x)
        d_loss, g_loss = losses.patch_discriminator_loss(disc, x_hat, x)
        loss = recon_loss + perceptual_loss + latent_loss + g_loss

        loss.backward()
        optimizer.step()

        d_loss.backward()
        optimizer_disc.step()

        if iter % 100 == 0:
            print("Epoch " + str(epoch).zfill(4) + ", Iter " + str(iter).zfill(4) \
                + " | Rec_loss: " + str(recon_loss.item())[:8] + " | Emb_loss: " + str(latent_loss.item())[:8] \
                + " | Perceptual_loss: " + str(perceptual_loss.item())[:8] \
                + " | G_loss: " + str(g_loss.item())[:8] + " | D_loss: " + str(d_loss.item())[:8] )

            # Generate reconstructed samples
            x_rec = x_hat.detach()
            x_fig = torch.cat([x[0:8], x_rec[0:8], x[8:16], x_rec[8:16]], 0)
            path = os.path.join(results_path, str(epoch).zfill(4) + "_" + str(iter).zfill(4)+".jpg")
            vutils.save_image(x_fig, path, padding=2, normalize=False)

            # Save model
            torch.save(vqnet.state_dict(), os.path.join(save_path, model_name+".pt"))
        
        iter += 1
