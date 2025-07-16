import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import random
import os

def load_dataset(image_size=128, batch_size=16, dataroot="datasets/celeba_hq"):
    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Root directory for dataset
    #dataroot = "datasets/celeba"
    #dataroot = "datasets/celeba_hq"
    #dataroot = "datasets/birds_525_species"
    #dataroot = "datasets/birds_subset"
    #dataroot = "datasets/church"
    if not os.path.exists(dataroot):
        os.makedirs(dataroot)

    ngpu = 1

    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader