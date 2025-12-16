import torch
import torch.nn.functional as F

def fgsm_attack(model, images, labels, epsilon):
    """
    Fast Gradient Sign Method (FGSM) attack.

    Args:
        model: The target model.
        images: Input images (batch).
        labels: True labels.
        epsilon: Perturbation magnitude.

    Returns:
        Adversarial images.
    """
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    data_grad = images.grad.data
    sign_data_grad = data_grad.sign()
    perturbed_images = images + epsilon * sign_data_grad
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images
