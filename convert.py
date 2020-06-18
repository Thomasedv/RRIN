import os
import shutil
import sys
import time

import torch
import tqdm
from torch.utils.data import Dataset

from dataloader import Dataloader
from model import Net
from utils import Writer, ConvertSampler

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True


def dummy_collate(x):
    return x


def convert(args):
    # TODO: Create temp folders dynamically. depending on process.

    # Picking temp folder
    if args.input_video is not None:
        temp_folder = 'temp_' + os.path.basename(args.input_video)
        if args.resume:
            if not os.path.exists(temp_folder):
                raise Exception('Did not find temp folder to resume!')

    elif args.image_folder is not None:

        temp_folder = 'temp'
    else:
        raise Exception('Missing arguments! Video or folder needs to be specified')

    # TODO: Add temp folder out folder
    temp_out = temp_folder
    t = time.time()
    _extract_and_interpolate(args, temp_folder, temp_out)
    print('end=', time.time() - t)
    # Separate into functions, to clear memory once it's not needed.
    # _create_video(args, temp_out)


def _extract_and_interpolate(args, temp_folder, temp_out=None):
    if temp_out is None:
        temp_out = temp_folder

    inp = f'{temp_folder}\\input'
    dest = f'{temp_out}\\output'
    resume_index = 1

    if args.resume:
        print('Resuming conversion...')
        if os.path.exists(dest):
            if len(os.listdir(dest)) > 5:
                resume_index = (len(os.listdir(dest)) - 1) // (args.sf + 1)

        print('Resuming from index:', resume_index)

    else:
        if os.path.isdir(temp_folder):
            if os.path.exists(dest) and len(os.listdir(dest)):
                raise Exception(
                    'Folder is already in use! Did you intend to resume the progress? Use the --resume flag')

            elif args.image_folder is None:  # Remove if we are about to process a new video
                shutil.rmtree(temp_folder, ignore_errors=True)
                os.makedirs(temp_folder, exist_ok=True)
                os.makedirs(inp, exist_ok=True)

        else:
            os.makedirs(temp_folder)
            os.makedirs(inp)

        os.makedirs(dest, exist_ok=True)

    # Setup conditions depending on video or image input
    if args.input_video is not None:
        if not args.resume:
            # code = os.system(f'ffmpeg -i "{args.input_video}" -an -c:v libwebp -vsync 0 -lossless 1 -compression_level 4 -qscale 75 {inp}\\%9d.webp')
            code = os.system(f'ffmpeg -hide_banner -i "{args.input_video}" -an -lossless 1 {inp}\\%9d.png')

            if code:
                print('Failed to convert video to images.')
                sys.exit(1)

        input_path = inp
    elif args.image_folder is not None:
        input_path = args.image_folder
    else:
        raise KeyError('Either input video or images folder required!')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print('Cuda enabled!')

    dataset = Dataloader(path=input_path, cuda=use_cuda, train=False)
    testloader = torch.utils.data.DataLoader(dataset, sampler=ConvertSampler(dataset, resume_index - 1),
                                             batch_size=1, shuffle=False, pin_memory=False,
                                             collate_fn=dummy_collate, num_workers=0)
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

    writer = Writer(args.output_video)
    writer.start()

    with torch.no_grad():
        img_count = resume_index + resume_index * args.sf - args.sf
        # TQDM starting index possibly off by one.
        for (img1, img2, img1_data, img2_data), *_ in tqdm.tqdm(testloader, initial=resume_index - 1):
            # print(img1.device)
            # print(img2.device)
            if img_count == 1:
                writer.add_job('copy',
                               (img1_data["path"], os.path.join(dest, f'{img_count:09d}{img1_data["filetype"]}')))

            for i in range(1, intermediates + 1):
                # time in between frames, eg. for only a single interpolated frame, t=0.5
                time_step = i / (intermediates + 1)  # TODO: Check if correct
                output = model(img1.cuda(), img2.cuda(), t=time_step)
                # print(output.device)
                writer.add_job('tensor', (
                    output.cpu(),
                    img1_data,
                    os.path.join(dest, f'{img_count + i:09d}{img1_data["filetype"]}')))
                output = None
            # print(img1.device)
            # print(img2.device)
            writer.add_job('file', (
                img2_data["path"],
                os.path.join(dest, f'{img_count + intermediates + 1:09d}{img2_data["filetype"]}')
            ))
            # print('\n\n' + os.path.join(dest, f'{img_count + intermediates + 1:09d}{img1_data["filetype"]}') + '\n')
            img_count += intermediates + 1
            # print(f'\rProcessed {img_count} images', flush=True, end='')
    writer.exit_flag = True
    print('Waiting for file io to end...')
    writer.join()
    print('File io done!')


def _create_video(args, temp_folder):
    code = os.system(
        f'ffmpeg -r {args.fps} -y -i "{temp_folder}\\output\\%9d.webp" -an -c:v libvpx-vp9 -tile-columns 2 -tile-rows 1 -threads 12 -row-mt 1 -static-thresh 0 -frame-parallel 0 -auto-alt-ref 6 -lag-in-frames 25 -g 120 -crf 25 -pix_fmt yuv420p -cpu-used 4 -b:v 25M  -f webm -passlogfile ffmpeg2pass93057 -pass 1 NUL')
    if code:
        print('Failed to convert interpolated images to video, pass 1.')
        sys.exit(code)

    code = os.system(
        f'ffmpeg -r {args.fps} -y -i "{temp_folder}\\output\\%9d.webp" -c:a copy -c:v libvpx-vp9 -tile-columns 2 -tile-rows 1 -threads 12 -row-mt 1 -static-thresh 0 -frame-parallel 0 -auto-alt-ref 6 -lag-in-frames 25 -g 120 -crf 25 -pix_fmt yuv420p -cpu-used 1 -b:v 25M  -f webm -passlogfile ffmpeg2pass93057 -pass 2 {args.output_video}')

    if code:
        print('Failed to convert interpolated images to video.')
        sys.exit(1)

    if args.rm:
        shutil.rmtree(temp_folder, ignore_errors=True)

    print('Finished conversion')
