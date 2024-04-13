import argparse
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str)
args = parser.parse_args()

data_dir = args.data_dir
dataset = ImageFolder(data_dir)

while True:
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        image, _ = dataset[random.randint(0, len(dataset) - 1)]
        ax.imshow(image)
        ax.axis('off')
    # add button to go to next page
    plt.subplots_adjust(bottom=0.2)
    ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])
    button = Button(ax_button, 'Next')
    button.on_clicked(lambda _: plt.close())
    ax_button_2 = plt.axes([0.1, 0.05, 0.1, 0.075])
    kill_button = Button(ax_button_2, 'Stop')
    kill_button.on_clicked(lambda _: exit())
    plt.show()


