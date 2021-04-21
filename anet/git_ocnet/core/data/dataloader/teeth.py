"""Pascal VOC Semantic Segmentation Dataset."""
import os
import torch
import numpy as np

from PIL import Image
from .segbase import SegmentationDataset


class TeethSegmentation(SegmentationDataset):
    """Pascal VOC Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to VOCdevkit folder. Default is './datasets/VOCdevkit'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    >>> ])
    >>> # Create Dataset
    >>> trainset = VOCSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    #BASE_DIR = 'full-VOC'
    #BASE_DIR = 'teeth0602'
    # BASE_DIR = 'teethdata0814'
    BASE_DIR = 'teethdata0922'
    # BASE_DIR = 'teethdata0819_song_au'
    NUM_CLASS = 3
    # NUM_CLASS = 10

    #def __init__(self, root='../datasets/voc', split='train', mode=None, transform=None, **kwargs):
    #def __init__(self, root='../datasets/VOCdevkit', split='train', mode=None, transform=None, **kwargs):
    def __init__(self, root='../datasets/teeth', split='train', mode=None, transform=None, **kwargs):
        super(TeethSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        _voc_root = os.path.join(root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'SegmentationClass')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
        if split == 'train':
            _split_f = os.path.join(_splits_dir, 'train.txt')
        elif split == 'val':
            _split_f = os.path.join(_splits_dir, 'val.txt')
        elif split == 'test':
            _split_f = os.path.join(_splits_dir, 'test.txt')
        elif split == 'Aval':
            _split_f = os.path.join(_splits_dir, 'Aval.txt')
        elif split == 'Bval':
            _split_f = os.path.join(_splits_dir, 'Bval.txt')
        elif split == 'Cval':
            _split_f = os.path.join(_splits_dir, 'Cval.txt')
        elif split == 'Dval':
            _split_f = os.path.join(_splits_dir, 'Dval.txt')
        elif split == 'val_train':
            _split_f = os.path.join(_splits_dir, 'trainval.txt')
        elif split == 'val_adult':
            _split_f = os.path.join(_splits_dir, 'val_adult.txt')
        elif split == 'val_0813':
            _split_f = os.path.join(_splits_dir, 'val_0813.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                if split != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".png")
                    #print('_mask:',_mask)
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if split != 'test':
            assert (len(self.images) == len(self.masks))
        print('Found {} images in the folder {}'.format(len(self.images), _voc_root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # synchronized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img, mask, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    @property
    def classes(self):
        """Category names."""
        '''
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                'tv')
        '''
        return ('background', 'teeth', 'plaque')


if __name__ == '__main__':
    dataset = TeethSegmentation()
