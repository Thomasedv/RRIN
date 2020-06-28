import os
from collections import deque
from threading import Thread
from time import sleep

import numpy
from torch.utils.data.sampler import SequentialSampler
from torchvision.transforms.functional import to_pil_image as to_PIL
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


class FakeStr(str):
    """Since Vidgear only takes a dict for arguments, this allows for multiple map calls -map """

    def __hash__(self):
        return hash(str(self) + 'Fake')

    def __eq__(self, other):
        # if isinstance(other, self.__class__):
        #     return str(self) == str(other)
        return False


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

        # Hardware encode (Even nicer if you got multiple GPUs)
        # output_params = {"-input_framerate": str(framerate),
        #                  # '-hwaccel': 'cuda',
        #                  # '-hwaccel_device': '0',
        #                  '-i': source,
        #                  '-acodec': 'aac',
        #                  '-b:a': '320k',
        #                  # '-vf': 'format=nv12,hwupload',
        #                  # '-vcodec': 'vp9_vaapi',
        #                  '-vcodec': 'hevc_nvenc',
        #                  '-preset': 'slow',
        #                  '-b:v': '40M'
        #                  }

        output_params = {"-input_framerate": str(framerate),
                         '-i': source,
                         '-acodec': 'libvorbis',
                         '-b:a': '320k',
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
                         '-crf': '25',
                         '-b:v': '40M'
                         }
        # Hack, since ffmpeg args need to be ordered (and dicts are per python 3.7+)
        # It is easier to remove keys, than try to insert them later.
        # if os.path.isdir(source):
        #     del output_params['-i']
        #     del output_params['-map']
        #     del output_params[FakeStr('-map')]
        print(output_params)

        # TODO: Remove 2-pass params
        self.writer = WriteGear(output_filename=target_file, compression_mode=True,
                                custom_ffmpeg=self.ffmpeg, logging=False, **output_params)

    def add_job(self, method, item, max_queue=1000):
        if self.writer is None:
            raise Exception('Writer is not started yet. Call: Writer.start_writer(output)')
        # Hold thread when file IO is too far behind.
        while len(self.queue) > max_queue:
            print(' Large queue!')
            sleep(5)
        self.queue.append((method, item))

    def from_file(self, frame):
        # print(frame.shape)
        self.writer.write(frame, rgb_mode=True)
        self.save_count += 1
        # print('  ',self.save_count)

    def from_tensor(self, item):
        image, (w, h) = item
        # print('\n\n')
        # img = (image.squeeze(0)[:, :h, :w].transpose(0, 1).transpose(1, 2) * 255).int().numpy()
        # print('\n\n')
        # print('pre', image.squeeze(0).shape)
        # print('pre', image.squeeze(0)[:, :h, :w].shape)
        # print('post', img.shape)
        # print('post img2', img2.shape)
        # print(img2 - img)
        # print('\n\n')
        # print('\n\n')

        self.writer.write(numpy.asarray(to_PIL(image.squeeze(0)[:, -h:, :w].cpu())), rgb_mode=True)
        self.save_count += 1
        # print(' \n\n ', self.save_count, ' \n\n ')
        # image.save(dest., lossless=True, quality=75, method=4)

    def run(self) -> None:
        try:
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
        except Exception as e:
            global thread_exception
            thread_exception = e
        finally:
            self.writer.close()
