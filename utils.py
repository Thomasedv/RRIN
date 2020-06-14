import shutil
from collections import deque
from threading import Thread
from time import sleep

from torchvision import transforms


class Writer(Thread):
    """Class that offloads image crop and save from main thread. GREATLY speeds up conversion."""
    def __init__(self):
        super(Writer, self).__init__()
        self.queue = deque()
        self.exit_flag = False

    def add_job(self, method, item):
        self.queue.append((method, item))

    def run(self) -> None:
        while self.queue or not self.exit_flag:
            sleep(0.1)
            if self.queue:
                method, item = self.queue.popleft()

                if method == 'copy':
                    origin_path, dest_path = item
                    shutil.copy(origin_path, dest_path)
                elif method == 'write':
                    image, img_data, dest = item
                    image = transforms.functional.to_pil_image(image)

                    # Restore original dimensions
                    w_int, h_int = image.size
                    image = image.crop((0, abs(h_int - img_data['height']), img_data['width'], h_int))
                    image.save(dest)
