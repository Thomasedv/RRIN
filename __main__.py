import argparse

import torch

from convert import convert
from train import train

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True


def main(args):
    try:
        if args.mode == 'train':
            train(args)
        elif args.mode == 'convert':
            # if args.output_video is None:
            #     args.output_video = os.path.splitext(args.input_video)

            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                convert(args)
    except:
        if args.mode == 'convert':
            from utils import Writer
            Writer.exit_flag = True
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Video Frame Interpolation via Residue Refinement')
    parser.add_argument('--model_name', type=str, default='Model',
                        required=True, help='Name of model')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')

    parser.add_argument('--rm', action='store_true', default=False, help='Removed temp folder on proper finish.')

    sub = parser.add_subparsers(help='Performs training of model', dest='mode')
    sub.required = True

    train_args = sub.add_parser('train', help='Train the model', )
    train_args.add_argument('--train_folder', type=str, required=True,
                            help='Path to train sequences (training)')
    train_args.add_argument('--resume', action='store_true', default=False,
                            help='Resume model progress (training)')
    train_args.add_argument('--batch_size', type=int, default=2, required=False,
                            help='How many frames per batch (training)')

    sub_convert = sub.add_parser('convert', help='Performs interpolation of a video.')
    sub_convert.add_argument('--input_video', type=str,
                             required=False, help='Path to video to be interpolated.')
    sub_convert.add_argument('--output_video', type=str,
                             required=False, help='Path to new videofile.')
    sub_convert.add_argument('--sf', type=int,
                             required=True, help='How many intermediate frames to make.')
    sub_convert.add_argument('--fps', type=str,
                             required=True, help='FPS of output')
    sub_convert.add_argument('--image_folder', type=str,
                             required=False, help='Instead of taking frames from video, convert frames from a folder')
    sub_convert.add_argument('--resume', action='store_true', default=False,
                            help='Resume converting')

    # TODO: Add parameter to redo only part of conversion, eg, do not reinterpolate frames. Only convert frames to video

    # parser.add_argument('--samples', action='store_true', default=False, help='Enables samples during testing')
    # parser.add_argument('--test_folder', type=str, required=False, help='path to folder for saving checkpoints')

    args = parser.parse_args()
    main(args)
