import os
from adversarial_noise.generator import AdversarialGenerator
from adversarial_noise.visualizer import visualize_result

def test_adversarial_generation():
    """Test the adversarial noise generator with a sample image."""
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Initialize generator
    print("Initializing generator...")
    generator = AdversarialGenerator(model_name="resnet50")
    
    # Get first image from images directory
    image_dir = "images"
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in 'images' directory. Please add some images.")
        exit(1)
        
    image_path = os.path.join(image_dir, image_files[0])
    print(f"\nProcessing image: {image_path}")
    
    result = generator.find_minimum_noise(
        image_path,
        start_epsilon=0.001,
        max_epsilon=0.1,
        consistency_checks=5,
        epsilon_step=0.001
    )
    
    if result:
        print("\nSuccess! Found adversarial example:")
        print(f"Epsilon: {result.epsilon:.4f}")
        print(f"Original class: {result.original_class}")
        print(f"Adversarial class: {result.adversarial_class}")
        print(f"Confidence: {result.confidence:.2f}")
        
        # Save visualization
        output_name = f"adversarial_{os.path.splitext(image_files[0])[0]}.png"
        output_path = os.path.join("output", output_name)
        visualize_result(
            result.original_image,
            result.adversarial_image,
            result.original_class,
            result.adversarial_class,
            result.epsilon,
            output_path=output_path,
            show=True
        )
        print(f"\nVisualization saved to: {output_path}")
    else:
        print("\nNo successful adversarial example found in epsilon range")

def test_multiple_models():
    """Test the generator with different models."""
    
    models_to_test = ["resnet18", "resnet50", "vgg16"]
    
    # Get first image from images directory
    image_dir = "images"
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in 'images' directory. Please add some images.")
        exit(1)
        
    image_path = os.path.join(image_dir, image_files[0])
    
    print("\nTesting multiple models:")
    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        generator = AdversarialGenerator(model_name=model_name)
        
        result = generator.find_minimum_noise(
            image_path,
            start_epsilon=0.001,
            max_epsilon=0.1,
            consistency_checks=5,
            epsilon_step=0.001
        )
        
        if result:
            print(f"Success with {model_name}:")
            print(f"Epsilon: {result.epsilon:.4f}")
            print(f"Original -> Adversarial: {result.original_class} -> {result.adversarial_class}")
        else:
            print(f"No success with {model_name}")

if __name__ == "__main__":
    # Check if images directory exists
    if not os.path.exists("images"):
        print("Please create an 'images' directory and add some test images before running the tests.")
        print("Supported formats: JPG, JPEG, PNG")
        exit(1)
    
    print("Starting tests...\n")
    
    # Run basic test
    test_adversarial_generation()
    
    # Uncomment to test multiple models
    # test_multiple_models()
    
    print("\nTests completed!") 