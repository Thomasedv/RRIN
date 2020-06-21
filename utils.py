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

    def __init__(self, target_file, framerate):
        super(Writer, self).__init__()
        self.queue = deque()
        self.save_count = 0
        # TODO: Make argument
        self.ffmpeg = r'C:\Users\thoma\User PATH\ffmpeg.exe'

        output_params = {"-input_framerate": str(framerate),
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

    def add_job(self, method, item, max_queue=750):
        if self.writer is None:
            raise Exception('Writer is not started yet. Call: Writer.start_writer(output)')
        # Hold thread when file IO is too far behind.
        while len(self.queue) > max_queue:
            print(' Large queue!')
            sleep(5)
        self.queue.append((method, item))

    def from_file(self, frame):
        self.writer.write(frame)
        self.save_count += 1
        # print('  ',self.save_count)

    def from_tensor(self, item):
        image, (w, h) = item
        image = transforms.functional.to_pil_image(image.cpu().squeeze(0))

        # Restore original dimensions
        w_int, h_int = image.size
        image = image.crop((0, abs(h_int - h), w, h_int))

        # image = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_BGR2RGB)
        self.writer.write(numpy.asarray(image))
        self.save_count += 1
        # print(' \n\n ', self.save_count, ' \n\n ')
        # image.save(dest., lossless=True, quality=75, method=4)

    def run(self) -> None:
        while True:
            sleep(0.05)
            if self.queue:
                method, item = self.queue.popleft()
                if method == 'file':
                    self.from_file(item)
                elif method == 'tensor':
                    self.from_tensor(item)
                else:
                    raise Exception(f'Got unknown job: {method}')
            # Exit when queue empty and no more coming.
            # TODO: Rename to no_more_jobs FLAG
            if self.exit_flag and not self.queue:
                break
        self.writer.close()
