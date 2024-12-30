from typing import Optional, List, Dict, Union, Tuple
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from transformers import ViTForImageClassification, ViTImageProcessor

@dataclass
class AdversarialResult:
    epsilon: float
    success: bool
    original_class: str
    adversarial_class: str
    original_image: torch.Tensor
    adversarial_image: torch.Tensor
    confidence: float

class AdversarialGenerator:
    """Generate adversarial examples using FGSM (Fast Gradient Sign Method)."""
    
    def __init__(self, model_name: str = "google/vit-base-patch16-224", device: Optional[str] = None):
        """
        Initialize the generator with a specific model.
        
        Args:
            model_name: Name of the model to use (default: "google/vit-base-patch16-224")
            device: Device to run on ("cuda" or "cpu"). If None, automatically detected.
        """
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device)
        
        # Load model and processor
        print(f"Loading {model_name}...")
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        
        self.model.eval()
        self.model.to(self.device)
        
        # Get class labels
        self.id2label = self.model.config.id2label

    def _load_image(self, image_input: Union[str, Image.Image]) -> torch.Tensor:
        """Load and preprocess image from path or PIL Image."""
        if isinstance(image_input, str):
            image = Image.open(image_input)
        else:
            image = image_input
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Process image using the model's processor
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs.pixel_values.to(self.device)

    def _generate_single_adversarial(self,
                                   image: torch.Tensor,
                                   epsilon: float,
                                   target_class: Optional[int],
                                   confidence_threshold: float) -> AdversarialResult:
        """Generate a single adversarial example."""
        image.requires_grad = True
        
        # Original prediction
        with torch.no_grad():
            outputs = self.model(image)
            original_pred = outputs.logits.argmax(-1).item()
            original_probs = torch.softmax(outputs.logits, dim=-1)
            original_confidence = original_probs.max().item()
            original_class = self.id2label[original_pred]
            print(f"Original prediction: {original_class} (confidence: {original_confidence:.2f})")
        
        # Forward pass
        outputs = self.model(image)
        
        if target_class is None:
            # Untargeted attack - maximize loss for current prediction
            initial_pred = outputs.logits.argmax(-1)
            loss = -nn.CrossEntropyLoss()(outputs.logits, initial_pred)  # Negative to maximize loss
        else:
            # Targeted attack
            target = torch.tensor([target_class]).to(self.device)
            loss = nn.CrossEntropyLoss()(outputs.logits, target)
        
        # Backward pass
        loss.backward()
        
        # Generate adversarial example
        perturbed_image = image + epsilon * image.grad.data.sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        # Check result
        with torch.no_grad():
            adv_outputs = self.model(perturbed_image)
            adv_pred = adv_outputs.logits.argmax(-1).item()
            adv_probs = torch.softmax(adv_outputs.logits, dim=-1)
            confidence = adv_probs.max().item()
            print(f"Adversarial prediction: {self.id2label[adv_pred]} (confidence: {confidence:.2f})")
            
        success = (adv_pred != original_pred and confidence >= confidence_threshold)
        
        return AdversarialResult(
            epsilon=epsilon,
            success=success,
            original_class=original_class,
            adversarial_class=self.id2label[adv_pred],
            original_image=image.detach(),
            adversarial_image=perturbed_image.detach(),
            confidence=confidence
        )

    def find_minimum_noise(self, 
                          image_path: Union[str, Image.Image],
                          start_epsilon: float = 0.01,
                          max_epsilon: float = 0.3,
                          consistency_checks: int = 5,
                          epsilon_step: float = 0.01) -> Optional[AdversarialResult]:
        """
        Find the minimum epsilon that consistently generates successful adversarial examples.
        
        Args:
            image_path: Path to image or PIL Image object
            start_epsilon: Starting epsilon value
            max_epsilon: Maximum allowed epsilon
            consistency_checks: Number of times to verify consistent misclassification
            epsilon_step: How much to increase epsilon by when unsuccessful
            
        Returns:
            AdversarialResult object or None if no successful attack found
        """
        image = self._load_image(image_path)
        
        # Get original prediction
        with torch.no_grad():
            outputs = self.model(image)
            original_pred = outputs.logits.argmax(-1).item()
            original_class = self.id2label[original_pred]
        
        current_epsilon = start_epsilon
        
        while current_epsilon <= max_epsilon:
            # Try current epsilon multiple times to ensure consistency
            successes = 0
            results = []
            
            print(f"\nTesting epsilon: {current_epsilon:.4f}")
            for _ in range(consistency_checks):
                result = self._generate_single_adversarial(
                    image.clone(),
                    current_epsilon,
                    target_class=None,
                    confidence_threshold=0.7
                )
                results.append(result)
                if result.success:
                    successes += 1
                    
            success_rate = successes / consistency_checks
            print(f"Success rate: {success_rate:.2%}")
            
            # If we get consistent misclassification, return the result
            if success_rate >= 0.8:  # 80% success rate threshold
                # Return the result with median confidence among successful attempts
                successful_results = [r for r in results if r.success]
                median_result = sorted(successful_results, 
                                    key=lambda x: x.confidence)[len(successful_results)//2]
                return median_result
                
            current_epsilon += epsilon_step
        
        return None 