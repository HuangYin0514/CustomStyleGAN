
from functools import partial
from torch.utils import data
import torchvision
from torchvision import transforms
from PIL import Image
from pathlib import Path

EXTS = ['jpg', 'png']

# dataset


def convert_rgb_to_transparent(image):
    if image.mode == 'RGB':
        return image.convert('RGBA')
    return image


def convert_transparent_to_rgb(image):
    if image.mode == 'RGBA':
        return image.convert('RGB')
    return image


def expand_greyscale(num_channels):
    def inner(tensor):
        return tensor.expand(num_channels, -1, -1)
    return inner


def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image


class Dataset(data.Dataset):
    def __init__(self, folder, image_size, transparent=False):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(
            f'{folder}').glob(f'**/*.{ext}')]

        convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
        num_channels = 3 if not transparent else 4

        self.transform = transforms.Compose([
            transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale(num_channels))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)
