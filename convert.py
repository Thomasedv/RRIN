import os
import time

import torch
from torch.utils.data import Dataset

from dataloader import ConvertLoader
from utils import Writer, ConvertSampler, get_thread_error, TQDM, get_model

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True


def dummy_collate(x):
    return x


def convert(args):
    start_time = time.time()
    _extract_and_interpolate(args)
    print(f'Conversion finished in {time.time() - start_time:.2f} seconds!')


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

    model = get_model(args.model_type, use_cuda)

    for i in reversed(os.listdir('checkpoints')):
        if i.lower().startswith(args.model_name.lower()):
            state = torch.load(os.path.join('checkpoints', i))
            model.load_state_dict(state['model'], strict=True)
            del state
            print(f'Using model {os.path.join("checkpoints", i)}.')
            break
    else:
        raise TypeError(f'No model found with the name: {args.model_name}')

    if use_cuda:
        model = model.cuda()
    model.eval()

    if args.chop_forward:
        if not hasattr(model, 'forward_chop'):
            print('chop_forward is not supported by this model, disabling it.')
            args.chop_forward = False
        print('Using chop_forward')

    if args.sf is None:
        if 'x' in args.fps:
            intermediates = max(int(args.fps.replace('x', '')) - 1, 1)
        else:
            raise Exception('--sf has to be given, unless --fps is given as a multiple of the input, eg. "--fps 2x"')
    else:
        intermediates = args.sf

    output_fps = dataset.input_framerate * int(args.fps.replace('x', '')) if 'x' in args.fps else int(args.fps)

    # If the duration is changed, audio/subs are not included.
    if intermediates + 1 != output_fps / dataset.input_framerate:
        args.input_video = None

    print(f'Input Framerate: {dataset.input_framerate:.4f}\nOutput Framerate: {output_fps:.4f}')
    writer = Writer(args.output_video, output_fps, source=args.input_video)
    writer.start()

    with torch.no_grad():
        # Deprecated, but may prove useful in the future.
        # img_count = resume_index + resume_index * args.sf - args.sf
        img_count = 1  # resume_index

        # Dataset defaults to 1e9 frames when total length is unknown.
        # Change these to use infinte tqdm mode.
        inf_mode = len(dataset) > 1e8
        if inf_mode:
            print('Unknown frame count! Progress bar set to inf mode.')
            total = float('inf')
            unit = 'frames'
        else:
            total = len(dataset)
            unit = 'frame(s)'

        frame_iterator = TQDM(convert_loader, desc='Converting', unit=unit, total=total)
        # TQDM starting index possibly off by one.
        for (img1, img2, img1_data, img2_data), *_ in frame_iterator:

            # Raise treaded errors to main thread.
            if get_thread_error() is not None:
                raise get_thread_error()

            # NB: Make sure jobs are added sequentially. Orders matters.
            if img_count == 1:
                writer.add_job('file', img1_data)

            for i in range(1, intermediates + 1):
                # time in between frames, eg. for only a single interpolated frame, t=0.5
                time_step = i / (intermediates + 1)

                output = model(img1,
                               img2.cuda(non_blocking=True),
                               t=time_step,
                               chop=args.chop_forward,
                               reuse_flow=(i < intermediates),
                               min_size=args.patch_size,
                               padding=args.padding
                               )

                writer.add_job('tensor', (output.detach().cpu(), (dataset.width, dataset.height)))
                output = None

            writer.add_job('file', img2_data)
            img_count += intermediates + 1

    writer.exit_flag = True
    print('Waiting for output processing to end...')
    writer.join()
    print('File IO done!')
