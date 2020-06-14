import os
import shutil
import sys
import threading
from collections import deque
from threading import Thread
from time import sleep

import torch
import tqdm
from torch.utils.data import Dataset
from torchvision import transforms

from dataloader import Dataloader
from model import Net

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True


def dummy_collate(x):
    return x


def convert(args):
    # TODO: Speedups:
    #   Save images in threads (after moved to CPU)

    temp_folder = 'temp'
    if os.path.isdir(temp_folder):
        shutil.rmtree(temp_folder, ignore_errors=True)

    os.makedirs(temp_folder)
    os.makedirs(f'{temp_folder}\\output')

    if args.input_video is not None:
        os.makedirs(f'{temp_folder}\\input')
        code = os.system(f"ffmpeg -i {args.input_video} -vsync 0 {temp_folder}\\input\\%9d.png")
        if code:
            print('Failed to convert video to images.')
            sys.exit(1)
        else:
            input_path = f'{temp_folder}\\input'

    elif args.image_folder is not None:
        input_path = args.image_folder
    else:
        raise KeyError('Either input video or images folder required!')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print('Cuda enabled!')

    dataset = Dataloader(path=input_path, cuda=use_cuda, train=False)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False,
                                             collate_fn=dummy_collate)
    model = Net()

    for i in reversed(os.listdir('models')):
        if i.lower().startswith(args.model_name.lower()):
            state = torch.load(os.path.join('models', i))
            model.load_state_dict(state['model'], strict=True)
            del state
            print(f'Using model {os.path.join("models", i)}.')
            break
    else:
        raise TypeError(f'No model found with the name: {args.model_name}')

    model = model.cuda()
    model.eval()
    intermediates = args.sf  # sf in superslomo

    dest = f'{temp_folder}\\output'
    writer = Writer()
    writer.start()

    with torch.no_grad():
        img_count = 1
        for data in tqdm.tqdm(testloader):
            img1, img2, img1_data, img2_data = data[0]
            del data
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

            if img_count == 1:
                writer.add_job('copy',
                               (img1_data["path"], os.path.join(dest, f'{img_count:09d}{img1_data["filetype"]}')))

            for i in range(1, intermediates + 1):
                # time in between frames, eg. for only a single interpolated frame, t=0.5
                time_step = i / (intermediates + 1)  # TODO: Check if correct
                output = model(img1, img2, t=time_step)

                writer.add_job('write', (
                    output.squeeze(0).cpu(),
                    img1_data,
                    os.path.join(dest, f'{img_count + i:09d}{img1_data["filetype"]}')))
                output = None

            writer.add_job('copy', (
                img2_data["path"],
                os.path.join(dest, f'{img_count + intermediates + 1:09d}{img1_data["filetype"]}')
            ))
            # print('\n\n' + os.path.join(dest, f'{img_count + intermediates + 1:09d}{img1_data["filetype"]}') + '\n')
            img_count += intermediates + 1
            # print(f'\rProcessed {img_count} images', flush=True, end='')
    writer.exit_flag = True
    while writer.is_alive():
        sleep(0.1)

    code = os.system(
        f"ffmpeg -r {args.fps} -y -i {temp_folder}\\output\\%9d.png -an -c:v libvpx-vp9 -tile-columns 2 -tile-rows 1 -threads 12 -row-mt 1 -static-thresh 0 -frame-parallel 0 -auto-alt-ref 6 -lag-in-frames 25 -g 120 -crf 30 -pix_fmt yuv420p -cpu-used 4 -b:v 20M  -f webm -passlogfile ffmpeg2pass93057 -pass 1 NUL")
    if code:
        print('Failed to convert interpolated images to video, pass 1.')
        sys.exit(code)

    code = os.system(
        f"ffmpeg -r {args.fps} -i {temp_folder}\\output\\%9d.png -c:a copy -c:v libvpx-vp9 -tile-columns 2 -tile-rows 1 -threads 12 -row-mt 1 -static-thresh 0 -frame-parallel 0 -auto-alt-ref 6 -lag-in-frames 25 -g 120 -crf 30 -pix_fmt yuv420p -cpu-used 1 -b:v 20M  -f webm -passlogfile ffmpeg2pass93057 -pass 2 {args.output_video}")

    if not code:
        pass  # shutil.rmtree({temp_folder}, ignore_errors=True)
    else:
        print('Failed to convert interpolated images to video.')
        sys.exit(1)
