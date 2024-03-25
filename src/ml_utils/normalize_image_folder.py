import os, argparse, imghdr
from PIL import Image
import torch

def find_normalization(path, crop_size=224, gray_scale=False):
    r_mean = 0
    g_mean = 0
    b_mean = 0

    r_std = 0
    g_std = 0
    b_std = 0

    count = 0

    # recursively find all images in the directory
    for root, dirs, files in os.walk(path):
        for file in files:
            # if file is a valid image of any format
            if imghdr.what(os.path.join(root, file)) is not None:
                img = Image.open(os.path.join(root, file))

                if not gray_scale:
                    img = img.convert('RGB')
                else:
                    img = img.convert('L')

                img = img.resize((crop_size, crop_size))

                img = torch.tensor(img).float()

                r_mean += img[:,:,0].mean()
                g_mean += img[:,:,1].mean()
                b_mean += img[:,:,2].mean()

                r_std += img[:,:,0].std()
                g_std += img[:,:,1].std()
                b_std += img[:,:,2].std()

                count += 1


    return ((r_mean, g_mean, b_mean), (r_std, g_std, b_std))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates means and standard deviations for RGB values of images in a directory')

    parser.add_argument('path', type=str, help='Path to the directory containing the images')

    parser.add_argument('--crop_size', type=int, default=224, help='Size to crop the images to before calculating the means and standard deviations')

    parser.add_argument('--gray_scale', default=False, action='store_true', help='Use this flag if the images are grayscale')

    args = parser.parse_args()

    means, stds = find_normalization(args.path, crop_size=args.crop_size, gray_scale=args.gray_scale)

    print(f'Means: {means}')
    print(f'Standard Deviations: {stds}')


