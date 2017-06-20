import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(img_root, tar_root):
    images = []
    targets = []
    for root, _, fnames in sorted(os.walk(img_root)):
        for fname in fnames:
            if is_image_file(fname):
                img_path = os.path.join(img_root, fname)
                tar_path = os.path.join(tar_root, fname[:-4]+'_mask.tif')
                images.append(img_path)
                targets.append(tar_path)

    return images, targets


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
        
def p_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('P')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class NerveDataset(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/1212.tif

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, img_root, tar_root, transform=None, target_transform=None,
                 pil_loader=pil_loader, p_loader=p_loader):
        imgs, tars = make_dataset(img_root, tar_root)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + img_root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.img_root = img_root
        self.tar_root = tar_root
        self.imgs = imgs
        self.tars = tars
        self.transform = transform
        self.target_transform = target_transform
        self.loader1 = pil_loader
        self.loader2 = p_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) 
        """
        img_path = self.imgs[index]
        tar_path = self.tars[index]
        img = self.loader1(img_path)
        tar = self.loader2(tar_path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            tar = self.target_transform(tar)

        return img, tar

    def __len__(self):
        return len(self.imgs)
