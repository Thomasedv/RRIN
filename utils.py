import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from time import sleep

import cv2
import numpy

from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms
from vidgear.gears import WriteGear


def get_sampler(start_index):
    def sampler(data_source):
        return ConvertSampler(data_source, start_index)

    return sampler


class ConvertSampler(SequentialSampler):
    def __init__(self, source, start_idx=0):
        super(ConvertSampler, self).__init__(source)
        self.start_idx = start_idx

    def __iter__(self):
        return iter(range(self.start_idx, len(self.data_source)))


class Writer(Thread):
    """
    Class that offloads image crop and save from main thread. GREATLY speeds up conversion.

    For SSDs, use a higher number of workers.

    """
    exit_flag = False

    def __init__(self, target_file):
        super(Writer, self).__init__()
        self.queue = deque()
        self.writer = None

        # TODO: Make argument
        self.ffmpeg = r'C:\Users\thoma\User PATH\ffmpeg.exe'

        output_params = {"-input_framerate": '60',
                         '-vcodec': 'libvpx-vp9',
                         '-tile-columns': '2',
                         '-tile-rows': '1',
                         '-threads': '12',
                         '-row-mt': '1',
                         '-static-thresh': '0',
                         '-frame-parallel': '0',
                         '-auto-alt-ref': '6',
                         '-lag-in-frames': '25',
                         '-g': '120',
                         '-crf': '30',
                         '-b:v': '40M'
                         # '-pix_fmt': 'yuv420p'
                         }
        self.writer = WriteGear(output_filename=target_file, compression_mode=True,
                                custom_ffmpeg=self.ffmpeg, logging=False, **output_params)

    def add_job(self, method, item, max_queue=500):
        if self.writer is None:
            raise Exception('Writer is not started yet. Call: Writer.start_writer(output)')
        # Hold thread when file IO is too far behind.
        while len(self.queue) > max_queue:
            print(' Large queue!')
            sleep(5)
        self.queue.append((method, item))

    def from_file(self, src):
        # shutil.copy(src, dst)
        frame = cv2.imread(src, cv2.IMREAD_COLOR)
        # {do something with the frame here}
        # write frame to writer
        self.writer.write(frame)

    def from_tensor(self, item):
        image, img_data, dest = item
        image = transforms.functional.to_pil_image(image.cpu().squeeze(0))

        # Restore original dimensions
        w_int, h_int = image.size
        image = image.crop((0, abs(h_int - img_data['height']), img_data['width'], h_int))

        image = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
        self.writer.write(image)
        # image.save(dest., lossless=True, quality=75, method=4)

    def run(self) -> None:
        while True:
            sleep(0.1)
            if self.queue:
                method, item = self.queue.popleft()
                if method == 'file':
                    src, dst = item
                    self.from_file(src)
                elif method == 'tensor':
                    self.from_tensor(item)

            # Exit when queue empty and no more coming.
            # TODO: Rename to no_more_jobs FLAG
            if self.exit_flag and not self.queue:
                break
        self.writer.close()
