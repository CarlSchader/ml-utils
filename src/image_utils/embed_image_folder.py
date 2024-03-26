import os, shutil
import torch
import torchvision
from torchvision.datasets import ImageFolder

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
    device=torch.device('cpu'),
    verbose=False,
):
    dataset = ImageFolder(root=data_folder, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    shutil.rmtree(save_dir, ignore_errors=True)
    os.mkdir(save_dir)

    # get output shape of model by sampling a single image
    sample_image, _ = dataset[0]
    sample_image = sample_image.unsqueeze(0)
    output_shape = encoder(sample_image).shape[1:]

    with torch.no_grad():
        current_emb = torch.zeros(output_shape)
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
                class_name = dataset.classes[label]
                if label != current_label:
                    class_average = current_emb / current_count
                    torch.save(class_average, os.path.join(save_dir, f'{class_name}.pth'))
                    current_emb = torch.zeros(output_shape)
                    current_label = label
                    current_count = 0

                current_emb += emb
                current_count += 1


            batches += 1
            if verbose:
                print(f'Processed batch... {batches}/{total_batches}', end='\r')

