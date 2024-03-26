import torchvision.transforms as transforms
from torchvision import datasets
import torch


def find_image_folder_normalization(path, crop_size=224, batch_size=64, total_batches=None, device=torch.device('cpu'), verbose=False):
    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    means = torch.zeros(3).to(device)
    pixel_count = 0
    count = 0

    print('Calculating means...') if verbose else None
    for inputs, _ in loader:
        inputs = inputs.to(device)
        means += inputs.sum(dim=(0, 2, 3))

        count += 1
        pixel_count += inputs.size(0) * inputs.size(2) * inputs.size(3)

        print(f"Processed {count}/{len(loader)}", end='\r') if verbose else None

        if total_batches is not None and count >= total_batches:
            break

    means /= pixel_count

    stds = torch.zeros(3).to(device)
    count = 0

    print('Calculating standard deviations...') if verbose else None
    for inputs, _ in loader:
        inputs = inputs.to(device)
        stds += ((inputs - means[None, :, None, None]) ** 2).sum(dim=(0, 2, 3))

        count += 1
        print(f"Processed {count}/{len(loader)}", end='\r') if verbose else None

        if total_batches is not None and count >= total_batches:
            break

    stds = torch.sqrt(stds / pixel_count)
    return means, stds




