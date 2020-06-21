import os
import threading
import typing
from multiprocessing import Queue
from random import randint
from time import sleep
import subprocess as sp

import av
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2


class Dataloader(Dataset):
    def __init__(self, path='input', train=False, cuda=False):
        self.path = path
        self.train = train
        self.cuda = cuda
        # TODO: Implement resume for open_CV
        #  (worst case: Load images from original target, count frames, and determine resume point for interpolation.)
        self.flip_map = {}

        # Train stuff
        if self.train:
            self.folder = os.listdir(self.path)

            # Needs to follow input rules for UNet.
            self.resize_dims = (640, 368)  # Training dimensions
            self.real_dims = (1280, 720)  # Real dimensions

            # Train data
            self.cropX0 = self.real_dims[0] - self.resize_dims[0]
            self.cropY0 = self.real_dims[1] - self.resize_dims[1]

        # Cache so we don't load the same image twice. Converting only
        self.last_img = None

        video = cv2.VideoCapture(path)
        self.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) # float
        self.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()

        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        print(self.stream)
        self.frame_iter = self.container.decode(self.stream)
        print(self.stream.frames)
        print('\n\nsdfsfsd', self.stream.frames, 'sdfsdfd\n\n')
        self.width = 1
        self.height = 1
        # self.pipe = sp.Popen(['ffmpeg', "-i", r'.\videos\testc.webm',
        #                  "-loglevel", "quiet",  # no text output
        #                  "-an",  # disable audio
        #                  "-f", "image2pipe",
        #                  "-pix_fmt", "bgr24",
        #                  "-vsync", "0",  # FPS
        #                  # "-hls_list_size", "3",
        #                  # "-hls_time", "8"
        #                  "-vcodec", "rawvideo", "-"],
        #                 stdin=sp.PIPE, stdout=sp.PIPE)

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

    def stream_image(self, cuda=False):
        raw_image = self.pipe.stdout.read(self.width * self.height * 3)  # read 432*240*3 bytes (= 1 frame)
        raw_image = np.fromstring(raw_image, dtype='uint8').reshape((self.height, self.width, 3))
        img = Image.fromarray(raw_image)
        width, height = img.size

        if width % 2 ** 4 != 0:
            right_pad = (width // 2 ** 4 + 1) * 2 ** 4 - width
        else:
            right_pad = 0

        if height % 2 ** 4 != 0:
            top_pad = (height // 2 ** 4 + 1) * 2 ** 4 - height
        else:
            top_pad = 0

        transform_list = [transforms.Pad((0, top_pad, 0, right_pad), padding_mode='edge'), transforms.ToTensor()]

        transform = transforms.Compose(transform_list)

        # Image data used to restore dimensions of original images when converting
        # Remove alpha channel
        # return transform(img)[:3, :, :].cuda().unsqueeze(0) if cuda else transform(img)[:3, :, :].unsqueeze(0), img_data

        if cuda:
            return transform(img)[:3, :, :].unsqueeze(0).pin_memory(), raw_image
        else:
            return transform(img)[:3, :, :].unsqueeze(0), raw_image

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
                # img1_path = self.folder[index]
                p1, p1_data = self.stream_image(self.cuda)

            else:
                last_index, p1, p1_data = self.last_img
                if last_index != index:
                    raise Exception('Error images acquired out of order')
                    # img1_path = self.folder[index]
                    # p1, p1_data = load_image_tensor(os.path.join(self.path, img1_path), self.cuda)
            # print(index)
            # img_path2 = self.folder[index + 1]
            p2, p2_data = self.stream_image(self.cuda)

            self.last_img = index+1, p2, p2_data
            # print('p1 data', p1_data)
            # print('p2 data', p2_data)
            return p1, p2, p1_data, p2_data

    def __len__(self):
        if self.train:
            return len(self.folder) * 2
        else:
            return self.len - 1


class ConvertLoader(Dataset):
    def __init__(self, path, cuda=False):
        self.path = path
        self.cuda = cuda
        # TODO: Implement resume for open_CV
        #  (worst case: Load images from original target, count frames, and determine resume point for interpolation.)

        # Cache so we don't load the same image twice. Converting only
        self.last_img = None

        if os.path.isdir(path):
            self.mode = 'folder'
            self.len = len(os.listdir(path))
            self.frame_iter = iter([os.path.join(path, file) for file in os.listdir(path)])
            self.width = None
            self.height = None
        else:
            self.mode = 'video'
            video = cv2.VideoCapture(path)
            self.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
            self.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video.release()

            self.container = av.open(path)
            self.stream = self.container.streams.video[0]
            self.frame_iter = self.container.decode(self.stream)

        # Get video data
        # video = cv2.VideoCapture(path)
        # self.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) # float
        # self.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # video.release()
        #
        # self.pipe = sp.Popen(['ffmpeg', "-i", r'.\videos\testc.webm',
        #                  "-loglevel", "quiet",  # no text output
        #                  "-an",  # disable audio
        #                  "-f", "image2pipe",
        #                  "-pix_fmt", "bgr24",
        #                  "-vsync", "0",  # FPS
        #                  # "-hls_list_size", "3",
        #                  # "-hls_time", "8"
        #                  "-vcodec", "rawvideo", "-"],
        #                 stdin=sp.PIPE, stdout=sp.PIPE)

    def load_image(self, cuda=False):
        # self.stream.
        img = Image.open(next(self.frame_iter))
        img = Image.merge("RGB", img.split()[::-1])
        width, height = img.size
        if self.width is None:
            self.width = width
            self.height = height
        if width % 2 ** 4 != 0:
            right_pad = (width // 2 ** 4 + 1) * 2 ** 4 - width
        else:
            right_pad = 0

        if height % 2 ** 4 != 0:
            top_pad = (height // 2 ** 4 + 1) * 2 ** 4 - height
        else:
            top_pad = 0

        transform_list = [transforms.Pad((0, top_pad, 0, right_pad), padding_mode='edge'), transforms.ToTensor()]

        transform = transforms.Compose(transform_list)

        # Image data used to restore dimensions of original images when converting
        # Remove alpha channel
        # return transform(img)[:3, :, :].cuda().unsqueeze(0) if cuda else transform(img)[:3, :, :].unsqueeze(0), img_data

        if cuda:
            return transform(img)[:3, :, :].unsqueeze(0).pin_memory(), np.array(img)
        else:
            return transform(img)[:3, :, :].unsqueeze(0), np.array(img)

    def stream_image(self, cuda=False):
        # self.stream.
        img = next(self.frame_iter).to_image()
        img = Image.merge("RGB", img.split()[::-1])
        width, height = img.size

        if width % 2 ** 4 != 0:
            right_pad = (width // 2 ** 4 + 1) * 2 ** 4 - width
        else:
            right_pad = 0

        if height % 2 ** 4 != 0:
            top_pad = (height // 2 ** 4 + 1) * 2 ** 4 - height
        else:
            top_pad = 0

        transform_list = [transforms.Pad((0, top_pad, 0, right_pad), padding_mode='edge'), transforms.ToTensor()]

        transform = transforms.Compose(transform_list)

        # Image data used to restore dimensions of original images when converting
        # Remove alpha channel
        # return transform(img)[:3, :, :].cuda().unsqueeze(0) if cuda else transform(img)[:3, :, :].unsqueeze(0), img_data

        if cuda:
            return transform(img)[:3, :, :].unsqueeze(0).pin_memory(), np.array(img)
        else:
            return transform(img)[:3, :, :].unsqueeze(0), np.array(img)

    def __getitem__(self, index):
        if self.last_img is None:
            # img1_path = self.folder[index]

            p1, p1_data = self.stream_image(self.cuda) if self.mode == 'video' else self.load_image(self.cuda)

        else:
            last_index, p1, p1_data = self.last_img
            if last_index != index:
                raise Exception('Error images acquired out of order')
                # img1_path = self.folder[index]
                # p1, p1_data = load_image_tensor(os.path.join(self.path, img1_path), self.cuda)
        # print(index)
        # img_path2 = self.folder[index + 1]
        p2, p2_data = self.stream_image(self.cuda) if self.mode == 'video' else self.load_image(self.cuda)

        self.last_img = index+1, p2, p2_data
        # print('p1 data', p1_data)
        # print('p2 data', p2_data)
        return p1, p2, p1_data, p2_data

    def __len__(self):
        return self.len - 1
