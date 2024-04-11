import os, shutil
import torch
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision import datasets

# def embed_image(
#     encoder,
#     image_path,
#     transform=torchvision.transforms.Compose([
#         torchvision.transforms.Resize((244, 244)),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
#     ]),
#     device=torch.device('cpu'),
#     ):
#     image = torchvision.io.read_image(image_path)
#     image = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         return encoder(image)

def embed_image_folder(
    encoder, 
    data_folder,
    save_dir,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((244, 244)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]),
    batch_size=32,
    averages_only=False,
    device=torch.device('cpu'),
    verbose=False,
    ):
    dataset = ImageFolder(root=data_folder, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    shutil.rmtree(save_dir, ignore_errors=True)
    os.mkdir(save_dir)

    # get output shape of model by sampling a single image
    sample_image, _ = dataset[0]
    sample_image = sample_image.unsqueeze(0).to(device)
    output_shape = encoder(sample_image).shape[1:]

    with torch.no_grad():
        class_average = torch.zeros(output_shape)
        class_stack = []
        current_label = 0
        current_count = 0
        batches = 0
        total_batches = len(data_loader)

        for images, labels in data_loader:
            images = images.to(device)
            embeddings = encoder(images)
            embeddings = embeddings.cpu()
            for idx, emb in enumerate(embeddings):
                label = labels[idx].item()

                if averages_only:
                    if label != current_label:
                        class_name = dataset.classes[current_label]
                        class_average = class_average / current_count
                        torch.save(class_average, os.path.join(save_dir, f'{class_name}.pth'))
                        class_average = torch.zeros(output_shape)
                        current_count = 0
                        current_label = label
                    class_average += emb
                else:
                    if label != current_label:
                        class_name = dataset.classes[current_label]
                        class_stack = torch.stack(class_stack, dim=0)
                        torch.save(class_stack, os.path.join(save_dir, f'{class_name}.pth'))
                        class_stack = []
                        current_count = 0
                        current_label = label
                    class_stack.append(emb)

                current_count += 1

            batches += 1
            if verbose:
                print(f'Processed batch... {batches}/{total_batches}', end='\r')

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




