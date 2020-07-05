from collections import deque
from threading import Thread
from time import sleep
import multiprocessing

import numpy
import tqdm
from torch.utils.data.sampler import SequentialSampler
from vidgear.gears import WriteGear

thread_exception = None


def get_thread_error():
    return thread_exception


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


class TQDM(tqdm.tqdm):
    def close(self):
        self.total = len(self.iterable)
        self.refresh()
        super(TQDM, self).close()


class Writer(Thread):
    """
    Class that offloads image crop and save from main thread. GREATLY speeds up conversion.

    For SSDs, use a higher number of workers.

    """
    exit_flag = False

    def __init__(self, target_file, framerate, source=None):
        """

        :param target_file: Target filename/path
        :type target_file: str
        :param framerate: Target framerate
        :type framerate: str
        :param source: Input video, used for linking any sound tracks to target
        :type source: str
        """
        super(Writer, self).__init__()
        self.queue = deque()
        self.save_count = 0
        # TODO: Make argument
        self.ffmpeg = r'C:\Users\thoma\User PATH\ffmpeg.exe'

        # Hardware encode (Probably even nicer if you got multiple GPUs)
        # output_params = {"-input_framerate": str(framerate),
        #                  # '-hwaccel': 'cuda',
        #                  # '-hwaccel_device': '0',
        #                  '-i': source,
        #                  '-clones': ['-map', '0:v:0', '-map', '1:a?', '-map', '1:s?'],
        #                  '-acodec': 'aac',
        #                  '-b:a': '320k',
        #                  # '-vf': 'format=nv12,hwupload',
        #                  # '-vcodec': 'vp9_vaapi',
        #                  '-vcodec': 'hevc_nvenc',
        #                  '-preset': 'slow',
        #                  '-b:v': '40M'
        #                  }

        # Used to set thread count-
        cpus = multiprocessing.cpu_count()

        output_params = {"-input_framerate": str(framerate),
                         '-i': source,
                         '-clones': ['-map', '0:v:0', '-map', '1:a?', '-map', '1:s?'],
                         '-acodec': 'libopus',
                         '-b:a': '320k',
                         '-vcodec': 'libvpx-vp9',
                         '-tile-columns': '2',
                         '-tile-rows': '1',
                         # '-threads': f'{max(cpus-2, 1)}',
                         '-threads': f'{cpus}',
                         '-cpu-used':'4',
                         '-row-mt': '1',
                         '-static-thresh': '0',
                         '-frame-parallel': '0',
                         '-lag-in-frames': '25',
                         '-g': f'{int(framerate*2)}',
                         '-crf': '25',
                         '-b:v': '40M',
                         '-r': str(framerate)
                         }

        # TODO: Remove 2-pass params
        # TODO: Test image folder for input. Conversion may fail.
        self.writer = WriteGear(output_filename=target_file, compression_mode=True,
                                custom_ffmpeg=self.ffmpeg, logging=False, **output_params)

    def add_job(self, method, item, max_queue=1000):
        # Hold thread when file IO is too far behind.
        if len(self.queue) > max_queue:
            print(' Large queue!')
            while len(self.queue) > max(max_queue - 100, 50):
                sleep(5)

        self.queue.append((method, item))

    def from_file(self, frame):
        # print(frame.shape)
        self.writer.write(frame, rgb_mode=True)
        self.save_count += 1
        # print('  ',self.save_count)

    def from_tensor(self, item):
        image, (w, h) = item

        # Manually convert to numpy array image that ffmpeg accepts. No need to go to PIL and then to numpy again
        image = image.squeeze(0)[:, -h:, :w].mul(255).byte()
        image = numpy.transpose(image.numpy(), (1, 2, 0))

        self.writer.write(image, rgb_mode=True)
        self.save_count += 1
        # print(' \n\n ', self.save_count, ' \n\n ')
        # image.save(dest., lossless=True, quality=75, method=4)

    def run(self) -> None:
        try:
            while True:
                sleep(0.1)
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
        except Exception as e:
            global thread_exception
            thread_exception = e
        finally:
            self.writer.close()
