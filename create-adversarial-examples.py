import torch
import json
import os
import clip
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random
import torchvision.transforms as transforms
import torch.nn.functional as F
from skimage import exposure
import cv2

device = "cuda:0" if torch.cuda.is_available() else "cpu"

mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]

os.makedirs('adv_PGD', exist_ok=True)
os.makedirs('adv_plots', exist_ok=True)
os.makedirs('adv_steps', exist_ok=True)

class ImageTextDataset(Dataset):
    def __init__(self, image_folder, annotations_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_paths = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        labels = self.annotations[self.image_paths[idx]]
        
        # Pro tip: Just specify number greater than len(labels) to always use a specific label
        # elif: label = labels[0] uses first label, label = labels[1] uses second label, etc.
        # Set to >=2 in order to make a random choice from your labels.
        if len(labels) >= 20:
            label = random.choice([labels[0], labels[1]])
        elif labels:
            label = labels[0]  # Fallback to first label if less than [len] are available
        else:
            label = ''  # Fallback if no labels are available

        text = clip.tokenize([label])

        return image, text.squeeze(0), image_path, label

def unnormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean

def save_intermediate_image(image_tensor, mean, std, step, image_path):
    image_np = unnormalize(image_tensor.squeeze().cpu(), mean, std).detach().numpy().transpose(1, 2, 0)
    image_np = np.clip(image_np, 0, 1)
    image_np = exposure.rescale_intensity(image_np, in_range=(0, 1))    
    image_np = (image_np * 255).astype(np.uint8)
    
    # Convert RGB image to LAB color space
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    
    # Split LAB channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L-channel back with a and b channels
    limg = cv2.merge((cl, a, b))
    
    # Convert LAB image back to RGB
    image_np_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)  
   
    # Convert numpy array to PIL Image
    image_pil = Image.fromarray(image_np_clahe.astype(np.uint8), mode="RGB")

    # Upscale the adjusted image using Lanczos resampling
    image = upscale_image(image_pil, scale_factor=3)
        
    base_name = os.path.basename(image_path).split('.')[0]
    image.save(f'adv_steps/{base_name}_step{step}.png')
    
def adjust_black_white_points(image, black_points, white_points):
    """Adjust the black and white points for each channel of the image.
    
    Args:
        image: Input image as a numpy array.
        black_points: List of black points for each channel [R, G, B].
        white_points: List of white points for each channel [R, G, B].
    
    Returns:
        Adjusted image as a numpy array.
    """
    channels = cv2.split(image)
    adjusted_channels = []
    for ch, bp, wp in zip(channels, black_points, white_points):
        adjusted_ch = exposure.rescale_intensity(ch, in_range=(bp, wp), out_range=(0, 255))
        adjusted_channels.append(adjusted_ch)
    return cv2.merge(adjusted_channels)

def upscale_image(image, scale_factor):
    """Upscale the image using the Lanczos resampling method."""
    width, height = image.size
    new_width, new_height = int(width * scale_factor), int(height * scale_factor)
    upscaled_image = image.resize((new_width, new_height), Image.LANCZOS)
    return upscaled_image

def pgd_attack(model, image, target_embedding, epsilon, alpha, iters, save_every=False, save_steps=25, image_path='', use_momentum=False, lr_schedule=None, reg_factor=1e-4, swa_start=0.75, swa_frequency=5):
    perturbed_image = image.clone().detach().requires_grad_(True)
    swa_image = perturbed_image.clone().detach()
    momentum = torch.zeros_like(perturbed_image).to(device)
    
    for i in range(iters):
        if lr_schedule:
            alpha = lr_schedule(i, iters)
        
        output = model.encode_image(perturbed_image)
        loss = torch.nn.functional.cosine_similarity(output, target_embedding).mean()
        
        # Adding L2 regularization
        l2_reg = reg_factor * torch.norm(perturbed_image)
        loss += l2_reg
        
        tv_loss = torch.sum(torch.abs(perturbed_image[:, :, :, :-1] - perturbed_image[:, :, :, 1:])) + \
                  torch.sum(torch.abs(perturbed_image[:, :, :-1, :] - perturbed_image[:, :, 1:, :]))
        loss += 0.0000001 * tv_loss

        model.zero_grad()
        loss.backward(retain_graph=True)
        
        if use_momentum:
            grad = perturbed_image.grad
            if grad is not None:
                grad = grad / torch.norm(grad, p=1)
                momentum = 0.9 * momentum + grad
                perturbed_image = perturbed_image + alpha * momentum.sign()
                # SWA logic
                if i > int(swa_start * iters) and i % swa_frequency == 0:
                    swa_image = ((swa_image + perturbed_image) / 2).requires_grad_(True)
                    perturbed_image = swa_image
            else:
                raise ValueError("Gradient is None. Ensure requires_grad is set to True and backward() is called.")
        else:
            grad = perturbed_image.grad
            if grad is not None:
                perturbed_image = perturbed_image + alpha * grad.sign()
            else:
                raise ValueError("Gradient is None. Ensure requires_grad is set to True and backward() is called.")
        
        perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon).detach().requires_grad_(True)
        perturbed_image = perturbed_image.detach().requires_grad_(True)
        
        if save_every and (i + 1) % save_steps == 0:
            save_intermediate_image(perturbed_image, mean, std, i + 1, image_path)
            print(loss)
        
   
    return perturbed_image


def evaluate_adversarial(model, images, target_embedding, epsilon, alpha, iters, save_every=False, save_steps=25, image_path='', use_momentum=False):
    model.eval()
    images.requires_grad = True
    output = model.encode_image(images)
    clean_similarity = torch.nn.functional.cosine_similarity(output, target_embedding).mean().item()

    pgd_data = pgd_attack(model, image, target_embedding, epsilon, alpha, iters, save_every, save_steps, image_path=image_path, use_momentum=use_momentum, lr_schedule=cosine_lr_schedule)
    output = model.encode_image(pgd_data)
    pgd_similarity = torch.nn.functional.cosine_similarity(output, target_embedding).mean().item()

    return clean_similarity, pgd_similarity, pgd_data




## SETTINGS ##

# Learning rate schedule
def cosine_lr_schedule(iteration, total_iterations, initial_lr=0.08):
    return initial_lr * (1 + np.cos(np.pi * iteration / total_iterations)) / 2

epsilon = 0.3
alpha = 0.08
iters = 500
sample_size = 10 # How many images from the dataset to use. Set "shuffle=False" below to disable random choice.

# --- alpha (α) - The step size for each iteration in PGD
# --- epsilon (ε) - The maximum perturbation allowed for each pixel:
# Good adversarial Value: ε ≤ 0.1 - This provides moderate perturbations that can be visible but not overly distortive.
# Likely Insufficient: ε ≤ 0.01 - This value results in very slight (potentially insufficient) perturbations.
# --- iters - Iterations:
# Iterations ≥ 100 - A large number of iterations can create highly refined (and visible) adversarial examples.
# Iterations = 50 for more 'normal' (invisible to human, but visible to model -> adversarial attack).

use_momentum = True  # Enable momentum for MI-FGSM. Disable for classic PGD.

save_every = True  # If True, save every save_steps. If False, save only final image.
save_steps = 10

use_single = True  # If True, use target_text as-is. If False, use invidual text for image, as returned by dataloader.

model, preprocess = clip.load("ViT-L/14", device=device, jit=False)

# Example dataset provided, insert your own here:
dataset = ImageTextDataset("images", "images-labels-bestnoise.json", transform=preprocess)
datasetloader = DataLoader(dataset, batch_size=1, shuffle=True) # Set False to disable random choice.

# Set a single target_text here, if desired:
if use_single:
    target_text = clip.tokenize(["darthvader"]).to(device)
    with torch.no_grad():
        target_embedding = model.encode_text(target_text).expand(sample_size, -1)



sampled_data = []
for i, (image, text, image_path, label) in enumerate(datasetloader):
    if i >= sample_size:
        break
    sampled_data.append((image.to(device), text.to(device), image_path, label))

results = []
for idx, (image, text, image_path, label) in enumerate(sampled_data):
    if not use_single:
        target_text = text
        with torch.no_grad():
            target_embedding = model.encode_text(target_text)
    clean_similarity, pgd_similarity, pgd_data = evaluate_adversarial(model, image, target_embedding, epsilon, alpha, iters, save_every, save_steps, image_path=image_path[0], use_momentum=use_momentum)
    results.append((image, pgd_data, clean_similarity, pgd_similarity, image_path))


for idx, (image, pgd_data, clean_similarity, pgd_similarity, image_path) in enumerate(results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    image_np = unnormalize(image.squeeze().cpu(), mean, std).detach().numpy().transpose(1, 2, 0)
    pgd_np = unnormalize(pgd_data.squeeze().cpu(), mean, std).detach().numpy().transpose(1, 2, 0)

    # Apply histogram stretching to pgd_np
    pgd_np = exposure.rescale_intensity(pgd_np, in_range=(0, 1))

    # Convert pgd_np to uint8 for CLAHE
    pgd_np_uint8 = (pgd_np * 255).astype(np.uint8)

    # Convert RGB image to LAB color space
    lab = cv2.cvtColor(pgd_np_uint8, cv2.COLOR_RGB2LAB)
    
    # Split LAB channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L-channel back with a and b channels
    limg = cv2.merge((cl, a, b))
    
    # Convert LAB image back to RGB
    pgd_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    axes[0].imshow(image_np)
    axes[0].set_title(f"Original Similarity: {clean_similarity:.2f}")
    axes[0].axis('off')

    axes[1].imshow(pgd_np)
    axes[1].set_title(f"PGD Similarity: {pgd_similarity:.2f}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"adv_plots/adv{idx}_e{epsilon}-a{alpha}-i{iters}.png", pad_inches=0.1)
    plt.close()
    
    pgd_image = Image.fromarray(pgd_np, mode="RGB")
    pgd_image.save(os.path.join('adv_PGD', os.path.basename(image_path[0])))
