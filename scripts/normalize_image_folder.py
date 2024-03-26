import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from carlschader_ml_tools.normalize_image_folder import find_image_folder_normalization
import argparse, torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates means and standard deviations for RGB values of images in a directory')

    parser.add_argument('path', type=str, help='Path to the directory containing the images')

    parser.add_argument('--crop_size', type=int, default=224, help='Size to crop the images to before calculating the means and standard deviations')

    parser.add_argument('--gray_scale', default=False, action='store_true', help='Use this flag if the images are grayscale')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size to use for calculating the means and standard deviations')

    parser.add_argument('--total_batches', type=int, default=None, help='Total number of batches to use for calculating the means and standard deviations (default: use all batches)')

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    means, stds = find_image_folder_normalization(args.path, crop_size=args.crop_size, batch_size=args.batch_size, total_batches=args.total_batches, device=device, verbose=True)

    print(f'Means: {means}')
    print(f'Standard Deviations: {stds}')


