import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class AdversarialGenerator:
    def __init__(self):
        # Load pretrained ResNet model
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        self.model = models.resnet50(weights=weights)
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Standard ImageNet preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Get ImageNet class labels from the weights
        self.categories = weights.meta["categories"]

    def load_image(self, image_path):
        """Load and preprocess image"""
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        return input_batch.to(self.device)

    def generate_adversarial(self, image, epsilon, target_class=None):
        """Generate adversarial example using FGSM"""
        image.requires_grad = True
        
        # Forward pass
        output = self.model(image)
        
        if target_class is None:
            # Untargeted attack - use original prediction
            initial_pred = output.max(1)[1]
            loss = nn.CrossEntropyLoss()(output, initial_pred)
        else:
            # Targeted attack
            target = torch.tensor([target_class]).to(self.device)
            loss = nn.CrossEntropyLoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Generate adversarial example
        data_grad = image.grad.data
        perturbed_image = image + epsilon * data_grad.sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        return perturbed_image.detach()

    def test_epsilon_range(self, image_path, epsilon_range=(0.01, 0.1), steps=10):
        """Test different epsilon values to find minimum effective perturbation"""
        image = self.load_image(image_path)
        
        # Get original prediction
        with torch.no_grad():
            original_output = self.model(image)
            original_pred = original_output.max(1)[1].item()
            original_class = self.categories[original_pred]
        
        results = []
        epsilons = np.linspace(epsilon_range[0], epsilon_range[1], steps)
        
        for epsilon in tqdm(epsilons):
            # Generate adversarial example
            perturbed_image = self.generate_adversarial(image.clone(), epsilon)
            
            # Get prediction for adversarial example
            with torch.no_grad():
                adv_output = self.model(perturbed_image)
                adv_pred = adv_output.max(1)[1].item()
                adv_class = self.categories[adv_pred]
            
            # Check if classification changed
            success = (adv_pred != original_pred)
            
            results.append({
                'epsilon': epsilon,
                'success': success,
                'original_class': original_class,
                'adversarial_class': adv_class,
                'example': perturbed_image.cpu()
            })
        
        return results

    def visualize_result(self, original_image, adversarial_image, original_class, 
                        adversarial_class, epsilon, output_path):
        """Visualize original and adversarial images side by side"""
        # Denormalize images
        denorm = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
        ])
        
        original_img = denorm(original_image.squeeze(0).cpu())
        adversarial_img = denorm(adversarial_image.squeeze(0))
        
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
        plt.savefig(output_path)
        plt.close()

def main():
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Initialize generator
    generator = AdversarialGenerator()
    
    # Process each image in the images folder
    for image_file in os.listdir('images'):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join('images', image_file)
            print(f"\nProcessing {image_file}...")
            
            # Test epsilon range
            results = generator.test_epsilon_range(image_path)
            
            # Find minimum effective epsilon
            success_results = [r for r in results if r['success']]
            if success_results:
                min_successful = min(success_results, key=lambda x: x['epsilon'])
                
                # Visualize result
                original_image = generator.load_image(image_path)
                output_path = os.path.join('output', f'adversarial_{os.path.splitext(image_file)[0]}.png')
                
                generator.visualize_result(
                    original_image,
                    min_successful['example'],
                    min_successful['original_class'],
                    min_successful['adversarial_class'],
                    min_successful['epsilon'],
                    output_path
                )
                
                print(f"Minimum effective epsilon: {min_successful['epsilon']:.4f}")
                print(f"Original class: {min_successful['original_class']}")
                print(f"Adversarial class: {min_successful['adversarial_class']}")
            else:
                print("No successful adversarial examples found in epsilon range")

if __name__ == "__main__":
    main()