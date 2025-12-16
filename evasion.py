import torch
import torch.nn.functional as F

def evasion_attack(model, images, labels, epsilon, num_iter=10):
    """
    Simple evasion attack (similar to iterative FGSM).

    Args:
        model: The target model.
        images: Input images (batch).
        labels: True labels.
        epsilon: Perturbation magnitude.
        num_iter: Number of iterations.

    Returns:
        Adversarial images.
    """
    perturbed_images = images.clone().detach()
    perturbed_images.requires_grad = True

    for _ in range(num_iter):
        outputs = model(perturbed_images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        data_grad = perturbed_images.grad.data
        perturbed_images = perturbed_images + epsilon * data_grad.sign()
        perturbed_images = torch.clamp(perturbed_images, 0, 1).detach_()
        perturbed_images.requires_grad = True

    return perturbed_images
