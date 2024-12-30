import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import Optional

def visualize_result(original_image: torch.Tensor,
                    adversarial_image: torch.Tensor,
                    original_class: str,
                    adversarial_class: str,
                    epsilon: float,
                    output_path: Optional[str] = None,
                    show: bool = True):
    """
    Visualize original and adversarial images side by side.
    
    Args:
        original_image: Original input image tensor
        adversarial_image: Generated adversarial image tensor
        original_class: Original classification
        adversarial_class: Adversarial classification
        epsilon: Epsilon value used
        output_path: Path to save visualization (optional)
        show: Whether to display the plot (default: True)
    """
    # Denormalize images
    denorm = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    ])
    
    original_img = denorm(original_image.squeeze(0).cpu())
    adversarial_img = denorm(adversarial_image.squeeze(0).cpu())
    
    # Convert to displayable format
    to_display = transforms.ToPILImage()
    
    plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(to_display(original_img))
    plt.title(f'Original\nClass: {original_class}')
    plt.axis('off')
    
    # Adversarial image
    plt.subplot(1, 2, 2)
    plt.imshow(to_display(adversarial_img))
    plt.title(f'Adversarial (ε={epsilon:.4f})\nClass: {adversarial_class}')
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    
    if show:
        plt.show()
    else:
        plt.close() 