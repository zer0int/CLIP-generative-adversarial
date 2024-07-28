## CLIP-generative-adversarial
### Who needs a Diffusion U-Net or Transformer if you can just use the 'text encoder'...? 😉

![examples](https://github.com/user-attachments/assets/e9d233f6-3690-4ee9-b6ea-52db40e85577)

- Uses Projected Gradient Descent (PGD), which is commonly used in adversarial robustness training
- Inverts objective and amplifies perturbations for human perception
- Perturbation towards (!) the prompt by default, making CLIP a self-generative AI 🙃
- Also plots results 'success' (original vs. perturbated cosine similarity)
- You can change the default behavior to "classic" adversarial example generation
- See code comments for details and instructions
------
- If you ever wondered why some strange word in a prompt for e.g. Stable Diffusion works - now you can find out!
- For Stable Diffusion V1, the sole text encoder "guide" is CLIP ViT-L/14 (the model set by default in my code).
- The diff between *this* and feature activation max visualization: We're using the whole model (output) to guide towards a text prompt.
- To visualize indivual 'neurons' (features) in CLIP ViT, see my other repo: [zer0int/CLIP-ViT-visualization](https://github.com/zer0int/CLIP-ViT-visualization)
- To get a CLIP opinion of what CLIP 'thinks' of an image, see: [zer0int/CLIP-XAI-GUI](https://github.com/zer0int/CLIP-XAI-GUI)
------
Requires: torch, torchvision, numpy, PIL, matplotlib, OpenCV, (skimage).
Requires [OpenAI/CLIP](https://github.com/openai/CLIP).
