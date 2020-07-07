import argparse

import torch

from convert import convert
from train import train

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True


def main():
    parser = argparse.ArgumentParser(description='PyTorch Video Frame Interpolation via Residue Refinement')
    parser.add_argument('--model_name', type=str, default='Model',
                        required=True, help='Name of model, provide BEFORE selecting train or convert mode!')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA')

    sub = parser.add_subparsers(help='Select mode, training or convert. Use ', dest='mode')
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
                             required=False, help='Path to video to be interpolated, or a folder of images!')
    sub_convert.add_argument('--output_video', type=str,
                             required=False, help='Path to new videofile. Must end with .webm. (Encoded with VP9)')
    sub_convert.add_argument('--sf', type=int,
                             required=False, help='How many intermediate frames to make. --sf 1 doubles frames')
    sub_convert.add_argument('--fps', type=str,
                             required=True, help='Frames per second of output. '
                                                 'Eg. from 30fps to 60fps, use --sf 1 --fps 60. '
                                                 'Can also be "2x" to double framerate')
    sub_convert.add_argument('--image_folder', type=str,
                             required=False, help='Instead of taking frames from video, convert frames from a folder')

    # Deprecated due to video frame piping.
    # sub_convert.add_argument('--resume', action='store_true', default=False,
    #                         help='Resume converting')

    # parser.add_argument('--samples', action='store_true', default=False, help='Enables samples during testing')
    # parser.add_argument('--test_folder', type=str, required=False, help='path to folder for saving checkpoints')

    args = parser.parse_args()

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
            from dataloader import ConvertLoader
            # Stop threads
            Writer.exit_flag = True
            ConvertLoader.exit_flag = True
        raise


if __name__ == '__main__':
    main()