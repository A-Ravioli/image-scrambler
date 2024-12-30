import click
import os
from .generator import AdversarialGenerator
from .visualizer import visualize_result

@click.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output path for visualization')
@click.option('--epsilon-min', default=0.01, help='Minimum epsilon value')
@click.option('--epsilon-max', default=0.1, help='Maximum epsilon value')
@click.option('--steps', default=10, help='Number of epsilon steps to try')
@click.option('--model', default='resnet50', help='Model to use')
@click.option('--confidence', default=0.9, help='Minimum confidence threshold')
@click.option('--device', help='Device to use (cuda/cpu)')
def main(image_path, output, epsilon_min, epsilon_max, steps, model, confidence, device):
    """Generate adversarial example for an image with minimal perturbation."""
    generator = AdversarialGenerator(model_name=model, device=device)
    
    result = generator.find_minimum_noise(
        image_path,
        epsilon_range=(epsilon_min, epsilon_max),
        steps=steps,
        confidence_threshold=confidence
    )
    
    if result:
        click.echo(f"Success! Found adversarial example:")
        click.echo(f"Epsilon: {result.epsilon:.4f}")
        click.echo(f"Original class: {result.original_class}")
        click.echo(f"Adversarial class: {result.adversarial_class}")
        click.echo(f"Confidence: {result.confidence:.2f}")
        
        if output:
            visualize_result(
                result.original_image,
                result.adversarial_image,
                result.original_class,
                result.adversarial_class,
                result.epsilon,
                output_path=output
            )
    else:
        click.echo("No successful adversarial example found in epsilon range")

if __name__ == '__main__':
    main() 