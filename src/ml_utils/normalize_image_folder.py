import argparse
import torchvision.transforms as transforms
from torchvision import datasets
import torch


def find_image_folder_normalization(path, crop_size=224, batch_size=64, total_batches=None, device=torch.device('cpu')):
    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    means = torch.zeros(3).to(device)
    pixel_count = 0
    count = 0

    print('Calculating means...')
    for inputs, _ in loader:
        inputs = inputs.to(device)
        means += inputs.sum(dim=(0, 2, 3))

        count += 1
        pixel_count += inputs.size(0) * inputs.size(2) * inputs.size(3)

        print(f"Processed {count}/{len(loader)}", end='\r')

        if total_batches is not None and count >= total_batches:
            break

    means /= pixel_count

    stds = torch.zeros(3).to(device)
    count = 0

    print('Calculating standard deviations...')
    for inputs, _ in loader:
        inputs = inputs.to(device)
        stds += ((inputs - means[None, :, None, None]) ** 2).sum(dim=(0, 2, 3))

        count += 1
        print(f"Processed {count}/{len(loader)}", end='\r')

        if total_batches is not None and count >= total_batches:
            break

    stds = torch.sqrt(stds / pixel_count)
    return means, stds

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


    means, stds = find_image_folder_normalization(args.path, crop_size=args.crop_size, batch_size=args.batch_size, total_batches=args.total_batches, device=device)

    print(f'Means: {means}')
    print(f'Standard Deviations: {stds}')


