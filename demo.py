import argparse
import sys
import typing
from random import randint

import math
import torch
import torchvision
from numpy.random.mtrand import random_integers
from torch import mean
from torchvision import transforms
from torch.optim.adamw import AdamW
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from model import Net
import os
import shutil
import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True


def load_image_tensor(img_path, cuda=False, resize=False, train=False):
    img = Image.open(img_path)

    if resize:
        size = (img.size[0] / 2, img.size[1] / 2)
        img.thumbnail(size)

    width, height = img.size

    # Keep original filetype on output
    ext = os.path.splitext(img_path)[1]

    if width % 2 ** 4 != 0:
        right_pad = (width // 2 ** 4 + 1) * 2 ** 4 - width
    else:
        right_pad = 0

    if height % 2 ** 4 != 0:
        top_pad = (height // 2 ** 4 + 1) * 2 ** 4 - height
    else:
        top_pad = 0

    transform = transforms.Compose([transforms.Pad((0, top_pad, 0, right_pad), padding_mode='edge'),
                                    transforms.ToTensor()])
    img_data = {'width': width,
                'height': height,
                'filetype': ext,
                'path': img_path}
    # Image data used to restore dimentions of original images when converting
    return transform(img)[:3, :, :].cuda() if cuda else transform(img)[:3, :, :], img_data


class Dataloader(Dataset):
    def __init__(self, path='input', train=False, cuda=False):
        self.path = path
        self.train = train
        self.cuda = cuda

        self.folder = os.listdir(self.path)

        # Cache so we don't load the same image twice. Converting only
        self.last_img = None  # type: typing.Union[None, typing.Tuple[torch.Tensor, dict]]

    def __getitem__(self, index):
        if self.train:
            subfolder = self.folder[index]
            sequence = []
            for img_path in os.listdir(os.path.join(self.path, subfolder)):
                # TODO: Implement flipping (and not flipping), and unflip for debug images.
                # if self.train:
                #     idx = randint(0, 1)
                #     img.transpose([Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM][idx])
                img, _ = load_image_tensor(os.path.join(self.path, subfolder, img_path), self.cuda,
                                           resize=self.train,
                                           train=self.train)
                sequence.append(img)
            return index, sequence

        else:
            if self.last_img is None:
                assert index == 0
                img1_path = self.folder[index]
                p1, p1_data = load_image_tensor(os.path.join(self.path, img1_path), self.cuda)
            else:
                last_index, p1, p1_data = self.last_img
                if last_index != index - 1:
                    print('Error images required out of order')
                    img1_path = self.folder[index - 1]
                    p1, p1_data = load_image_tensor(os.path.join(self.path, img1_path), self.cuda)

            img_path2 = self.folder[index]
            p2, p2_data = load_image_tensor(os.path.join(self.path, img_path2), self.cuda)

            self.last_img = index, p2, p2_data
            return p1, p2, p1_data, p2_data

    def __len__(self):
        if self.train:
            return len(self.folder)
        else:
            return len(self.folder) - 1


def convert(args):
    code = os.system(f"ffmpeg -i {args.input_video} -vsync 0 {args.convert_folde}\\%9d.png")
    if code:
        print('Failed to convert video to images.')
        sys.exit(1)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print('Cuda enabled!')

    dataset = Dataloader(path=args.convert_folder, cuda=use_cuda, train=args.train)
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

    dest = args.convert_folder_out
    with torch.no_grad():
        img_count = 1
        for data in tqdm.tqdm(testloader):
            img1, img2, img1_data, img2_data = data[0]

            output = model(img1.unsqueeze(0), img2.unsqueeze(0))
            output = output.squeeze(0).cpu()
            output = transforms.functional.to_pil_image(output)

            w_int, h_int = output.size
            # print(w_int, h_int)
            # output.save(f'output/intermediate uncropped{img_count + 1:09d}.jpg')
            # print((0, abs(h_int - orig_height), orig_width, h_int))

            # Restore original dimensions

            output = output.crop((0, abs(h_int - img1_data['height']), img1_data['width'], h_int))

            if img_count == 1:
                shutil.copy(img1_data["path"], os.path.join(dest, f'{img_count:09d}{img1_data["filetype"]}'))

            output.save(f'output/{img_count + 1:09d}{img1_data["filetype"]}')
            shutil.copy(img2_data["path"], os.path.join(dest, f'{img_count + 2:09d}{img1_data["filetype"]}'))

            img_count += 2
            # print(f'\rProcessed {img_count} images', flush=True, end='')

    code = os.system(f"ffmpeg -i {args.convert_folder_out}\\%9d.png -r 30 -b:v 10M -crf 15 {args.output_video}")
    if code:
        print('Failed to convert interpolated images to video.')
        sys.exit(1)


class VggLoss(nn.Module):
    def __init__(self):
        super(VggLoss, self).__init__()

        model = torchvision.models.vgg19(pretrained=True).cuda()

        self.features = nn.Sequential(
            # stop at relu4_4 (-10)
            *list(model.features.children())[:-10]
        )

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        outputFeatures = self.features(output)
        targetFeatures = self.features(target)

        loss = torch.norm(outputFeatures - targetFeatures, 2)

        return loss


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.vgg = VggLoss()
        self.l1 = nn.L1Loss()

    def forward(self, output, target) -> torch.Tensor:
        return self.vgg(output, target) + self.l1(output, target)


def charbonnierLoss(output, target):
    epsilon = 1e-6
    N = target.shape[0]
    return torch.sum(torch.sqrt((output - target).pow(2) + epsilon)) / N

os.getcwd()

def train(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print('Cuda enabled!')

    train_dataset = Dataloader(path=args.train_folder, cuda=use_cuda, train=args.train)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, pin_memory=False)

    model = Net()

    if args.resume:
        for i in reversed(os.listdir('models')):
            if i.lower().startswith(args.model_name.lower()):
                state = torch.load(os.path.join("models", i))
                print(f'Using model: {os.path.join("models", i)}')
                model.load_state_dict(state['model'], strict=True)
                break
        else:
            print('No checkpoint found with that modelname! Starting fresh!', )
            state = {}
    else:
        print('Starting new model')
        state = {}

    model = model.cuda()
    model.eval()

    start_epoch = state.get('epoch', 1)

    optim = AdamW(model.parameters(), lr=1e-4)
    for param_group in optim.param_groups:
        param_group['initial_lr'] = 1e-4

    sched = torch.optim.lr_scheduler.MultiStepLR(optim, [10, 25, 35], gamma=0.1, last_epoch=start_epoch)

    if 'optim' in state:
        optim.load_state_dict(state.get('optim'))

    if 'sched' in state:
        sched.load_state_dict(state.get('sched'))

    vgg16 = torchvision.models.vgg16(pretrained=True)
    vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
    vgg16_conv_4_3.cuda()

    for param in vgg16_conv_4_3.parameters():
        param.requires_grad = False
    #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    L1_lossFn = nn.L1Loss().to(device)
    MSE_LossFn = nn.MSELoss().to(device)
    ComboLossFn = CombinedLoss().to(device)

    epochs = 2000

    for epoch in range(start_epoch, epochs):
        print(f'--------- New Epoch {epoch} Current lr: {optim.param_groups[0]["lr"]} ---------')
        step = 0

        for indexes, (I0, It, I1) in trainloader:
            loss = 0
            itrs = 0
            for idx, f0, f_gt, f1 in zip(indexes, I0, It, I1):
                itrs += 1

                f0 = f0.unsqueeze(0)
                f1 = f1.unsqueeze(0)
                f_gt = f_gt.unsqueeze(0)

                f_int = model(f0, f1)
                #
                # recnLoss = L1_lossFn(f_int, f_gt)
                # prcpLoss = L1_lossFn(vgg16_conv_4_3(f_int), vgg16_conv_4_3(f_gt))
                # print('ch loss', charbonnierLoss(f_int, f_gt))
                charLoss = charbonnierLoss(f_int, f_gt)
                comboLoss = ComboLossFn(f_int, f_gt)
                loss += comboLoss / 10 + charLoss / 1e3
                # Samples at 10, 30
                if idx % 1000 == 0:
                    if not os.path.exists(f'debug/{idx}'):
                        os.makedirs(f'debug/{idx}')
                    print(f'Losses idx {idx}: charb {charLoss.item() / 1e3:6.4f}, combo {comboLoss / 10:6.4f}')

                    transforms.functional.to_pil_image(f0.squeeze(0).cpu()).save(f'debug/{idx}/Epoch{epoch:04d}_1Pre.png')
                    transforms.functional.to_pil_image(f1.squeeze(0).cpu()).save(f'debug/{idx}/Epoch{epoch:04d}_3Post.png')
                    transforms.functional.to_pil_image(f_int.squeeze(0).cpu()).save(f'debug/{idx}/Epoch{epoch:04d}_2int.png')

            step += 1
            optim.zero_grad()
            loss = loss / itrs
            loss.backward()
            optim.step()

            if step % 50 == 0:
                MSE_val = MSE_LossFn(f_int, f_gt)
                psnr = (10 * math.log10(1 / MSE_val.item()))
                print(f'step: {step}, psnr {psnr:2.4f}, loss {loss:4.4f}')

        sched.step()

        if epoch % 1 == 0:
            state = {'model': model.state_dict(),
                     'optim': optim.state_dict(),
                     'sched': sched.state_dict(),
                     'epoch': epoch}

            torch.save(state, 'models' + f"/{args.model_name}" + str(epoch) + ".pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Video Frame Interpolation via Residue Refinement')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--convert', action='store_true', default=False, help='Performs interpolation of images')
    group.add_argument('--train', action='store_true', default=False, help='Performs training of model')

    parser.add_argument('--train_folder', type=str, required=False, help='Path to train sequences')

    parser.add_argument('--input_video', type=str,
                        required=False, help='Path to video to be interpolated.')

    parser.add_argument('--output_video', type=str,
                        required=False, help='Path to new videofile.')

    parser.add_argument('--convert_folder', type=str, default='temp/input',
                        required=False, help='Path to folder to place images to be interpolated.')
    parser.add_argument('--convert_folder_out', type=str, default='temp/output',
                        required=False, help='Path to folder for converted images')

    parser.add_argument('--model_name', type=str, default='Model',
                        required=False, help='Name of model')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume model progress')

    # parser.add_argument('--samples', action='store_true', default=False, help='Enables samples during testing')
    # parser.add_argument('--test_folder', type=str, required=False, help='path to folder for saving checkpoints')

    args = parser.parse_args()

    if args.train and (args.train_folder is None):
        parser.error("Train folder needs to be specified")

    if args.train:
        train(args)
    else:
        if None in [args.output_video, args.input_video]:
            parser.error("Both a input video, and output video needs to be specified!")
        convert(args)
