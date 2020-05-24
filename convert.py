import os
import shutil
import sys

import torch
import tqdm
from torch.utils.data import Dataset
from torchvision import transforms

from model import Net


def convert(args):
    if os.path.isdir('temp'):
        shutil.rmtree('temp', ignore_errors=True)

    os.makedirs('temp')
    os.makedirs('temp\\output')

    if args.input_video is not None:
        code = os.system(f"ffmpeg -i {args.input_video} -vsync 0 temp\\input\\%9d.png")
        if code:
            print('Failed to convert video to images.')
            sys.exit(1)
        else:
            input_path = 'temp\\input'
            os.makedirs(input_path)
    elif args.image_folder is not None:
        input_path = args.image_folder
    else:
        raise KeyError('Either input video or images folder required!')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print('Cuda enabled!')

    dataset = Dataloader(path=input_path, cuda=use_cuda, train=False)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False,
                                             collate_fn=lambda x: x)
    model = Net()

    for i in reversed(os.listdir('models')):
        if i.lower().startswith(args.model_name.lower()):
            state = torch.load(os.path.join('models', i))
            model.load_state_dict(state['model'], strict=True)
            print(f'Using model {os.path.join("models", i)}.')
            break
    else:
        raise TypeError(f'No model found with the name: {args.model_name}')

    model = model.cuda()
    model.eval()
    intermediates = args.sf  # sf in superslomo

    dest = 'temp\\output'
    with torch.no_grad():
        img_count = 1
        for data in tqdm.tqdm(testloader):
            img1, img2, img1_data, img2_data = data[0]
            del data
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

            if img_count == 1:
                shutil.copy(img1_data["path"], os.path.join(dest, f'{img_count:09d}{img1_data["filetype"]}'))
                # print('\n\n'+os.path.join(dest, f'{img_count:09d}{img1_data["filetype"]}')+'\n')

            for i in range(1, intermediates + 1):
                # time in between frames, eg. for only a single interpolated frame, t=0.5
                time_step = i / (intermediates + 1)  # TODO: Check if correct

                output = model(img1, img2, t=time_step)

                output = output.squeeze(0).cpu()
                output = transforms.functional.to_pil_image(output)

                # Restore original dimensions
                w_int, h_int = output.size

                output = output.crop((0, abs(h_int - img1_data['height']), img1_data['width'], h_int))
                output.save(os.path.join(dest, f'{img_count + i:09d}{img1_data["filetype"]}'))
                # print('\n\n' + os.path.join(dest, f'{img_count + i:09d}{img1_data["filetype"]}') + '\n')
                output = None

            shutil.copy(img2_data["path"],
                        os.path.join(dest, f'{img_count + intermediates + 1:09d}{img1_data["filetype"]}'))
            # print('\n\n' + os.path.join(dest, f'{img_count + intermediates + 1:09d}{img1_data["filetype"]}') + '\n')
            img_count += intermediates + 1
            # print(f'\rProcessed {img_count} images', flush=True, end='')

    code = os.system(f"ffmpeg -r {args.fps} -i temp\\output\\%9d.png -b:v 10M -crf 30 {args.output_video}")

    if not code:
        shutil.rmtree('temp', ignore_errors=True)
    else:
        print('Failed to convert interpolated images to video.')
        sys.exit(1)
