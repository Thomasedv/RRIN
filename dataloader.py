import os
from collections import deque
from random import randint
from threading import Thread
from time import sleep

import av
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TrainDataloader(Dataset):
    def __init__(self, path='input', cuda=False):
        self.path = path
        self.cuda = cuda

        self.folder = os.listdir(self.path)

        # Needs to follow input rules for UNet.
        self.resize_dims = (640, 368)  # Training dimensions
        self.real_dims = (1280, 720)  # Real dimensions

        # Train data
        self.cropX0 = self.real_dims[0] - self.resize_dims[0]
        self.cropY0 = self.real_dims[1] - self.resize_dims[1]

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

    def load_image_tensor(self, img_path, cuda=False, crop=None, flip: int = 0):
        img = Image.open(img_path)  # type: Image.Image

        # Keep original filetype on output
        ext = os.path.splitext(img_path)[1]

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
        transform = transforms.Compose(transform_list)

        if cuda:
            return transform(img)[:3, :, :].unsqueeze(0).pin_memory(), img_data
        else:
            return transform(img)[:3, :, :].unsqueeze(0), img_data

    def is_randomcrop(self, index):
        return index >= len(self.folder)

    def __getitem__(self, index):
        # Does dataset with resized images, and then again with a random crop
        folder_index = index - len(self.folder) * self.is_randomcrop(index)
        subfolder = self.folder[folder_index]

        sequence = []
        flip = randint(0, 2)

        if self.is_randomcrop(index):
            cropX = randint(0, self.cropX0)
            cropY = randint(0, self.cropY0)
            crop_area = (cropX, cropY, cropX + self.resize_dims[0], cropY + self.resize_dims[1])
        else:
            crop_area = None

        for img_path in os.listdir(os.path.join(self.path, subfolder)):
            img, _ = self.load_image_tensor(os.path.join(self.path, subfolder, img_path), self.cuda,
                                            flip=flip, crop=crop_area)
            sequence.append(img)

        return index, sequence, flip

    def __len__(self):
        return len(self.folder) * 2


class ConvertLoader(Dataset):
    exit_flag = False

    def __init__(self, path, cuda=False):
        self.path = path
        self.cuda = cuda

        # TODO: Implement resume for open_CV
        #  (worst case: Load images from original target, count frames, and determine resume point for interpolation.)

        # Transform to apply
        self.transform = None

        # Cache so we don't load the same image twice. Converting only
        self.last_img = None

        # Used for preloader to know how many images to preload
        self.current_index = 0
        self.preloaded_index = 0

        # The limit on preloaded images. Take note that gpu memory is pinned for each
        self.preload_limit = 6  # This can go pretty high, eg. 100 frames
        self.preload_queue = deque(maxlen=self.preload_limit + 1)

        if os.path.isdir(path):
            self.mode = 'folder'
            self.len = len(os.listdir(path))
            self.frame_iter = iter(Image.open(os.path.join(self.path, file)) for file in os.listdir(path))
            self.width = None
            self.height = None
        else:
            self.mode = 'video'
            video = cv2.VideoCapture(path)
            self.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
            self.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.len <= 0:
                self.len = int(1e9)
            self.input_framerate = video.get(cv2.CAP_PROP_FPS)
            video.release()

            self.container = av.open(path, buffer_size=32768 * 1000)

            self.v_stream = self.container.streams.video[0]

            # Below have been tested, minimal time gain.
            cc = self.v_stream.codec_context

            # Fast decode, non-significant gain, could possibly break stuff?
            self.v_stream.flags2 |= cc.flags2.FAST
            # if 'LOW_DELAY' in flags:
            #     self.v_stream.flags |= cc.flags.LOW_DELAY
            #

            self.v_stream.thread_type = 'AUTO'
            # print(bool(cc.flags & cc.flags.LOW_DELAY))

            # Iterator that fetches images from ffmpeg
            self.frame_iter = (i.to_image() for i in self.container.decode(self.v_stream))

        # Max one thread, streaming images from ffmpeg can't do more.
        # Potentially can be done with loading folder image, but assume it's not needed.
        self.preload_thread = Thread(target=self._preload)
        self.preload_thread.start()

    def setup_transform(self, img=None):
        if img is not None:
            width, height = img.size
            self.width = width
            self.height = height
        else:
            width, height = self.width, self.height

        if width % 2 ** 4 != 0:
            right_pad = (width // 2 ** 4 + 1) * 2 ** 4 - width
        else:
            right_pad = 0

        if height % 2 ** 4 != 0:
            top_pad = (height // 2 ** 4 + 1) * 2 ** 4 - height
        else:
            top_pad = 0

        transform_list = [transforms.Pad((0, top_pad, right_pad, 0), padding_mode='edge'), transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def process_image(self, img):
        if self.transform is None:
            self.setup_transform(img)

        # Removes Alpha channel
        return self.transform(img)[:3, :, :].unsqueeze(0)

    def stream_image(self):
        """Loads image either from ffmpeg PIPE or folder."""
        # Sometimes framerate missreported. This ensure we end properly in those cases.

        img = next(self.frame_iter)
        # img = Image.merge("RGB", img.split()[::-1])

        if self.cuda:
            return self.process_image(img).pin_memory(), np.array(img)
        else:
            return self.process_image(img), np.array(img)

    def _preload(self):
        """
        Fetches images from input, times out if it does not get new images before the timeout.
        Timeouts prevent hang in case model stops working and the program doesn't exit on it's own.
        """
        try:
            timeout_limit = 20 + (30 * (not self.cuda))  # seconds, extended if on cpu mode.(Can timeout on slow cpu?)
            timeout = 0

            while True:
                if self.exit_flag:
                    return

                if self.preloaded_index == self.len:
                    self.len += 1

                if self.current_index + self.preload_limit > self.preloaded_index:

                    # Preloading image
                    try:
                        image, data = self.stream_image()  # if self.mode == 'video' else self.load_image()
                    except StopIteration:
                        self.len = self.preloaded_index
                        return

                    self.preload_queue.append((image, data))
                    self.preloaded_index += 1
                    # print('Preload index', self.preloaded_index)
                    timeout = 0
                else:
                    # print(f'Waiting for current index to increase {self.current_index + 1}')
                    sleep(0.1)
                    timeout_limit += 0.1
                    if timeout >= timeout_limit:
                        raise Exception('Preloader timed out!')
            # print(f'Loaded all images! preload idx {self.preloaded_index}')
        except Exception as e:
            # Stop main thread by sending it exceptions from thread
            global thread_exception
            thread_exception = e

    def preload_pop(self):
        """
        Fetches images from queue, times out if queue does not get new images before the timeout.
        Timeouts prevent hang in case something stops working and the program doesn't exit on it's own.
        """
        timeout_limit = 15
        timeout = 0
        while not self.preload_queue:
            timeout += 0.2

            # Less frames than expected, but done with all frames.
            if self.current_index == self.len:
                raise StopIteration

            # Timeout check and don't timeout on start.
            if timeout > timeout_limit:
                if self.current_index != 0:
                    raise Exception('Timeout when trying to preload images. ')
                elif timeout_limit * 2 < timeout:
                    raise Exception('Timeout when trying to preload images. ')
            sleep(0.2)
        item = self.preload_queue.popleft()
        self.current_index += 1
        return item

    def __getitem__(self, index):
        if self.last_img is None:
            p1, p1_data = self.preload_pop()
        else:
            last_index, p1, p1_data = self.last_img
            if last_index != index:
                raise Exception('Error images acquired out of order')

        p2, p2_data = self.preload_pop()
        self.last_img = index + 1, p2, p2_data
        return p1, p2, p1_data, p2_data

    def __len__(self):
        return self.len - 1
