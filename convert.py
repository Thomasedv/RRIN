import os
import shutil
import sys
import time

import torch
import tqdm
from torch.utils.data import Dataset

from dataloader import Dataloader, ConvertLoader
from model import Net
from utils import Writer, ConvertSampler

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True


def dummy_collate(x):
    return x


def convert(args):
    start_time  = time.time()
    _extract_and_interpolate(args)
    print(f'Conversion finished in {time.time() - start_time} seconds!')

def _extract_and_interpolate(args):
    # For future reference, this keeps image count on resume
    resume_index = 1

    # if args.resume:
    #     print('Resuming conversion...')
    #     if os.path.exists(dest):
    #         if len(os.listdir(dest)) > 5:
    #             resume_index = (len(os.listdir(dest)) - 1) // (args.sf + 1)
    #     print('Resuming from index:', resume_index)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print('Cuda enabled!')

    dataset = ConvertLoader(path=args.input_video, cuda=use_cuda)
    convert_loader = torch.utils.data.DataLoader(dataset, sampler=ConvertSampler(dataset, resume_index - 1),
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

    writer = Writer(args.output_video, args.fps)
    writer.start()

    with torch.no_grad():
        img_count = resume_index + resume_index * args.sf - args.sf
        # TQDM starting index possibly off by one.
        for (img1, img2, img1_data, img2_data), *_ in tqdm.tqdm(convert_loader):

            # NB: Make sure jobs are added sequentially. Orders matters.
            if img_count == 1:
                writer.add_job('file', img1_data)

            for i in range(1, intermediates + 1):
                # time in between frames, eg. for only a single interpolated frame, t=0.5
                time_step = i / (intermediates + 1)
                output = model(img1.cuda(), img2.cuda(), t=time_step)

                writer.add_job('tensor', (output.cpu(), (dataset.width, dataset.height)))
                output = None

            writer.add_job('file', img2_data)

            img_count += intermediates + 1

    writer.exit_flag = True
    print('Waiting for output processing to end...')
    writer.join()
    print('File io done!')

