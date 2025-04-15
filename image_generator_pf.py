import torch
import matplotlib.pyplot as plt
import torchvision
import os
import argparse
from network.gan import *  # Make sure this path is correct

def generate_and_save_images(generator, z_dim, save_path='generated_grid.png', num_images=16):
    generator.eval()
    device = next(generator.parameters()).device

    z = torch.randn(num_images, z_dim, 1, 1).to(device)
    with torch.no_grad():
        fake_images = generator(z).cpu()

    grid = torchvision.utils.make_grid(fake_images, nrow=4, normalize=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title("Generated Images from Generator")
    plt.tight_layout()
    # plt.savefig(save_path)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize Generator Output")
    parser.add_argument('--generator-path', type=str, required=True, help='Path to the saved generator weights')
    parser.add_argument('--z-dim', type=int, default=256, help='Latent vector size (default: 256)')
    parser.add_argument('--img-size', type=int, default=224, help='Image size (default: 32)')
    parser.add_argument('--num-images', type=int, default=16, help='Number of images to generate (default: 16)')
    parser.add_argument('--save-path', type=str, default='generated_grid.png', help='Path to save image grid')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = GeneratorDeconv(nz=args.z_dim, nc=3).to(device)
    generator.load_state_dict(torch.load(args.generator_path, map_location=device))

    print(f"Loaded generator from {args.generator_path}")
    generate_and_save_images(generator, args.z_dim, save_path=args.save_path, num_images=args.num_images)

if __name__ == "__main__":
    main()
