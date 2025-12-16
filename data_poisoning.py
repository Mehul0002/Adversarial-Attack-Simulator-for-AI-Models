import torch
import numpy as np

def data_poisoning_attack(images, labels, poison_rate=0.1, noise_level=0.1):
    """
    Simple data poisoning attack by adding noise and flipping labels.

    Args:
        images: Input images (batch).
        labels: True labels.
        poison_rate: Fraction of data to poison.
        noise_level: Amount of noise to add.

    Returns:
        Poisoned images and labels.
    """
    num_poison = int(poison_rate * images.size(0))
    indices = np.random.choice(images.size(0), num_poison, replace=False)

    poisoned_images = images.clone()
    poisoned_labels = labels.clone()

    # Add noise to images
    noise = torch.randn_like(images[indices]) * noise_level
    poisoned_images[indices] += noise
    poisoned_images = torch.clamp(poisoned_images, 0, 1)

    # Flip labels randomly
    poisoned_labels[indices] = torch.randint(0, 10, (num_poison,))

    return poisoned_images, poisoned_labels
