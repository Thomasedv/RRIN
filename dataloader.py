import os
import threading
import typing
from multiprocessing import Queue
from random import randint
from time import sleep

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Dataloader(Dataset):
    def __init__(self, path='input', train=False, cuda=False):
        self.path = path
        self.train = train
        self.cuda = cuda

        self.folder = os.listdir(self.path)

        # Needs to follow input rules for UNet.
        self.resize_dims = (640, 368)  # Training dimensions

        self.real_dims = (1280, 720)  # Real dimensions

        # Cache so we don't load the same image twice. Converting only
        self.last_img = None  # type: typing.Union[None, typing.Tuple[torch.Tensor, dict]]

        # Train data
        self.cropX0 = self.real_dims[0] - self.resize_dims[0]
        self.cropY0 = self.real_dims[1] - self.resize_dims[1]

        self.flip_map = {}

    def random_crop(self, crop_area=None):
        """ Crops image if given, else resizes to self.resize_dims, and pads image"""

        def rand_crop(img):
            return img.crop(crop_area)

        return rand_crop

    def random_flip(self, flip):
        def flip_img(img):
            if flip:
                return img.transpose([Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM][flip - 1])
            else:
                return img

        return flip_img

    def load_image_tensor(self, img_path, cuda=False, train=False, crop=None, flip: int = 0):
        img = Image.open(img_path)  # type: Image.Image

        # Keep original filetype on output
        ext = os.path.splitext(img_path)[1]

        if train:
            if crop is None:
                img.thumbnail(self.resize_dims, Image.ANTIALIAS)
                width, height = img.size

                if width % 2 ** 4 != 0:
                    right_pad = (width // 2 ** 4 + 1) * 2 ** 4 - width
                else:
                    right_pad = 0

                if height % 2 ** 4 != 0:
                    top_pad = (height // 2 ** 4 + 1) * 2 ** 4 - height
                else:
                    top_pad = 0

                # Resize to train dims, but keep aspect ratio so potential extra pixels missing. Pad missing instead.
                pre_transform = [transforms.Pad((0, top_pad, 0, right_pad), padding_mode='edge')]
                img_data = {'width': width,
                            'height': height,
                            'filetype': ext,
                            'path': img_path}
            else:
                pre_transform = [self.random_crop(crop)]
                img_data = {'width': self.resize_dims[0],
                            'height': self.resize_dims[1],
                            'filetype': ext,
                            'path': img_path}

            transform_list = pre_transform + [self.random_flip(flip),
                                              transforms.ToTensor()]

        else:
            width, height = img.size

            if width % 2 ** 4 != 0:
                right_pad = (width // 2 ** 4 + 1) * 2 ** 4 - width
            else:
                right_pad = 0

            if height % 2 ** 4 != 0:
                top_pad = (height // 2 ** 4 + 1) * 2 ** 4 - height
            else:
                top_pad = 0

            img_data = {'width': width,
                        'height': height,
                        'filetype': ext,
                        'path': img_path}

            transform_list = [transforms.Pad((0, top_pad, 0, right_pad), padding_mode='edge'), transforms.ToTensor()]

        transform = transforms.Compose(transform_list)

        # Image data used to restore dimensions of original images when converting
        # Remove alpha channel
        # return transform(img)[:3, :, :].cuda().unsqueeze(0) if cuda else transform(img)[:3, :, :].unsqueeze(0), img_data
        if cuda:
            return transform(img)[:3, :, :].unsqueeze(0).pin_memory(), img_data
        else:
            return transform(img)[:3, :, :].unsqueeze(0), img_data

    def is_randomcrop(self, index):
        return index >= len(self.folder)

    def __getitem__(self, index):
        if self.train:
            # Does dataset with resized images, and then again with a random crop
            folder_index = index - len(self.folder) * self.is_randomcrop(index)

            subfolder = self.folder[folder_index]

            sequence = []

            flip = randint(0, 2)
            self.flip_map[index] = flip

            if self.is_randomcrop(index):
                cropX = randint(0, self.cropX0)
                cropY = randint(0, self.cropY0)
                crop_area = (cropX, cropY, cropX + self.resize_dims[0], cropY + self.resize_dims[1])
            else:
                crop_area = None

            for img_path in os.listdir(os.path.join(self.path, subfolder)):
                # if self.train:
                #     idx = randint(0, 1)
                #     img.transpose([Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM][idx])
                img, _ = self.load_image_tensor(os.path.join(self.path, subfolder, img_path), self.cuda,
                                                train=self.train, flip=flip, crop=crop_area)

                sequence.append(img)
            return index, sequence

        else:
            if self.last_img is None:
                img1_path = self.folder[index]
                p1, p1_data = self.load_image_tensor(os.path.join(self.path, img1_path), self.cuda)
            else:
                last_index, p1, p1_data = self.last_img
                if last_index != index:
                    raise Exception('Error images acquired out of order')
                    # img1_path = self.folder[index]
                    # p1, p1_data = load_image_tensor(os.path.join(self.path, img1_path), self.cuda)

            img_path2 = self.folder[index + 1]
            p2, p2_data = self.load_image_tensor(os.path.join(self.path, img_path2), self.cuda)

            self.last_img = index+1, p2, p2_data
            # print('p1 data', p1_data)
            # print('p2 data', p2_data)
            return p1, p2, p1_data, p2_data

    def __len__(self):
        if self.train:
            return len(self.folder) * 2
        else:
            return len(self.folder) - 1
