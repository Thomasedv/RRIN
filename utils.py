import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from time import sleep

from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms


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

    def __init__(self, workers=2):
        super(Writer, self).__init__()
        self.queue = deque()
        # Keep low to not burden main thread
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def add_job(self, method, item, max_queue=500):
        # Hold thread when file IO is too far behind.
        while len(self.queue) > max_queue:
            sleep(5)
        self.queue.append((method, item))

    def copy(self, src, dst):
        print('Copying file', dst)
        shutil.copy(src, dst)

    def write(self, item):
        image, img_data, dest = item
        image = transforms.functional.to_pil_image(image)

        # Restore original dimensions
        w_int, h_int = image.size
        image = image.crop((0, abs(h_int - img_data['height']), img_data['width'], h_int))
        image.save(dest)

    def run(self) -> None:
        while self.queue or not self.exit_flag:
            sleep(0.1)
            if self.queue:
                method, item = self.queue.popleft()
                if method == 'copy':
                    src, dst = item
                    self.pool.submit(self.copy, src, dst)
                elif method == 'write':
                    self.pool.submit(self.write, item)

        self.pool.shutdown(wait=True)
