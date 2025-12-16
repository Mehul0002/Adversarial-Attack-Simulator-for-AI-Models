import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np

def tensor_to_image(tensor):
    """
    Convert a tensor to a PIL Image.
    """
    image = tensor.squeeze().cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image, mode='L')  # Grayscale for MNIST

def plot_images(original, adversarial, title1="Original", title2="Adversarial"):
    """
    Plot original and adversarial images side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original.squeeze().cpu().numpy(), cmap='gray')
    axes[0].set_title(title1)
    axes[0].axis('off')
    axes[1].imshow(adversarial.squeeze().cpu().numpy(), cmap='gray')
    axes[1].set_title(title2)
    axes[1].axis('off')
    plt.show()

def plot_perturbation(original, adversarial):
    """
    Plot the perturbation (difference between original and adversarial).
    """
    perturbation = adversarial - original
    plt.imshow(perturbation.squeeze().cpu().numpy(), cmap='RdBu', vmin=-1, vmax=1)
    plt.title("Perturbation")
    plt.colorbar()
    plt.axis('off')
    plt.show()

def plot_confidences(model, original, adversarial):
    """
    Plot confidence scores for original and adversarial images.
    """
    with torch.no_grad():
        orig_outputs = model(original)
        adv_outputs = model(adversarial)
        orig_probs = torch.softmax(orig_outputs, dim=1).squeeze().cpu().numpy()
        adv_probs = torch.softmax(adv_outputs, dim=1).squeeze().cpu().numpy()

    classes = [str(i) for i in range(10)]
    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, orig_probs, width, label='Original', alpha=0.8)
    ax.bar(x + width/2, adv_probs, width, label='Adversarial', alpha=0.8)
    ax.set_xlabel('Class')
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Confidences')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    plt.show()
