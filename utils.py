import os
from collections import deque
from threading import Thread
from time import sleep
import multiprocessing

import numpy
from PIL import Image
import tqdm
from torch.utils.data.sampler import SequentialSampler
from vidgear.gears import WriteGear

# Holds a exception that happened in a thread.
thread_exception = None


def get_thread_error():
    """Access exceptions caused in threads, can then exit in main thread."""
    global thread_exception
    return thread_exception


def get_model(model_type, use_cuda):
    if model_type == 'RRIN' or model_type is None:
        from models.rrin import Net
        model = Net(use_cuda=use_cuda)
    elif model_type == 'CAIN':
        from models.cain import CAIN
        model = CAIN()
    elif model_type.lower() == 'softmax_rrin':
        from models.softmax_rrin import Net
        if not use_cuda:
            raise NotImplementedError('This model does not support cuda.')
        model = Net(use_cuda=use_cuda)
    else:
        raise NotImplementedError(f'Unknown model: {model_type}')
    return model


class ConvertSampler(SequentialSampler):
    """
    Creates a sampler that starts from a target index, can be used to resume progress.
    No longer useful, but can be of use later.
    """
    def __init__(self, source, start_idx=0):
        super(ConvertSampler, self).__init__(source)
        self.start_idx = start_idx

    def __iter__(self):
        return iter(range(self.start_idx, len(self.data_source)))


class TQDM(tqdm.tqdm):
    """
    tqdm subclass that changes it's total to the number new length of the iterable when it closes.
    Particularly used when the reported number of frames for a video is too high, which in this case, means it won't
    stop before 100%, which may be confusing.
    """
    def close(self):
        self.total = len(self.iterable)
        self.refresh()
        super(TQDM, self).close()


class Writer(Thread):
    """
    Class that offloads image crop and save from main thread. GREATLY speeds up conversion.

    For SSDs, using a higher number of workers might increase performance.

    """
    exit_flag = False

    def __init__(self, target_path, framerate, source=None):
        """
        :param target_path: Target filename/path
        :type target_path: str
        :param framerate: Target framerate
        :type framerate: str
        :param source: Input video, used for linking any sound tracks to target
        :type source: str
        """
        super(Writer, self).__init__()

        # Image desat
        self.target_path = target_path

        _, file = os.path.split(target_path)
        # Image saving
        self.images = not '.' in file  # If last part of output has dot, is file else folder. Edgecase when folder with dot

        # Internal queue
        self._queue = deque()
        # Images saved, aka frame counter.
        self.save_count = 0

        # TODO: Make argument or finder.
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

        # Used to set thread count
        cpus = multiprocessing.cpu_count()

        # Encoding parmeters, including an infile where audio/subs can be added from the original.
        # TODO: Handle if source has incompatible audio channels for output (Eg. using something else than webm)
        output_params = {"-input_framerate": str(framerate),
                         '-disable_force_termination': True,
                         '-i': source,
                         '-clones': ['-map', '0:v:0', '-map', '1:a?', '-map', '1:s?'],
                         '-acodec': 'libopus',
                         '-b:a': '320k',
                         '-vcodec': 'libvpx-vp9',
                         '-tile-columns': '2',
                         '-tile-rows': '1',
                         '-threads': f'{cpus}',
                         '-cpu-used': '3',  # Key required to not prevent frame drops due to realtime assumption
                         '-row-mt': '1',
                         '-lag-in-frames': '25',
                         '-g': f'{int(framerate * 2)}',  # Lookahead 2x the framerate
                         '-r': str(framerate),
                         '-crf': '25',
                         '-b:v': '40M',
                         }

        # Easier to remove than try to insert at different positions in dict if the source is not None.
        if source is None:
            for key in ['-clones', '-i', '-b:a', '-acodec']:
                del output_params[key]
        # TODO: Remove 2-pass params
        # TODO: Test image folder for input. Conversion may fail.

        if self.images:
            print(f'Writing images for folder: {self.target_path}')
            if not os.path.isdir(target_path):
                os.mkdir(target_path)
        else:
            print(f'Creating video: {self.target_path}')
            self.writer = WriteGear(output_filename=target_path, compression_mode=True,
                                    custom_ffmpeg=self.ffmpeg, logging=False, **output_params)

    def add_job(self, method, item, max_queue=1000):
        # Hold thread when file IO is too far behind.
        if len(self._queue) > max_queue:
            print(' Large queue!')
            while len(self._queue) > max(max_queue - 100, 50):
                sleep(5)

        self._queue.append((method, item))

    def from_nparray(self, frame):
        """Writes numpy frame to output"""
        if self.images:
            img = Image.fromarray(frame)
            img.save(f'{self.target_path}\\{self.save_count:08d}.png')
        else:
            self.writer.write(frame, rgb_mode=True)
        self.save_count += 1
        # print('  ',self.save_count)

    def from_tensor(self, item):
        """Converts tensor to nparray and writes it."""
        image, (w, h) = item

        # Manually convert to numpy array image that ffmpeg accepts. No need to go to PIL and then to numpy again
        image = image.squeeze(0)[:, -h:, :w].mul(255).byte()
        image = numpy.transpose(image.numpy(), (1, 2, 0))

        self.from_nparray(image)
        # print(' \n\n ', self.save_count, ' \n\n ')
        # image.save(dest., lossless=True, quality=75, method=4)

    def run(self) -> None:
        try:
            while True:
                sleep(0.1)
                if self._queue:
                    method, item = self._queue.popleft()
                    if method == 'file':
                        self.from_nparray(item)
                    elif method == 'tensor':
                        self.from_tensor(item)
                    else:
                        raise Exception(f'Got unknown job: {method}')
                # Exit when queue empty and no more coming.
                # TODO: Rename to no_more_jobs FLAG
                if self.exit_flag and not self._queue:
                    break
        except Exception as e:
            global thread_exception
            thread_exception = e
        finally:
            if not self.images:
                self.writer.close()
