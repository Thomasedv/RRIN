import os

import math
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.optim.adamw import AdamW
from torch.utils.data import Dataset
from torchvision import transforms

from dataloader import Dataloader
from losses import CombinedLoss, charbonnierLoss
from model import Net

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True


def train(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print('Cuda enabled!')

    train_dataset = Dataloader(path=args.train_folder, cuda=use_cuda, train=True)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False)

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


    start_epoch = state.get('epoch', 1)

    optim = AdamW(model.parameters(), lr=1e-4)
    for param_group in optim.param_groups:
        param_group['initial_lr'] = 1e-4

    sched = torch.optim.lr_scheduler.MultiStepLR(optim, [75, 125, 135], gamma=0.1, last_epoch=start_epoch)

    if 'optim' in state:
        optim.load_state_dict(state.get('optim'))

    # if 'sched' in state:
    #     sched.load_state_dict(state.get('sched'))
    #     sched.milestones = Counter([10, 25, 35])

    vgg16 = torchvision.models.vgg16(pretrained=True)
    vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
    vgg16_conv_4_3.cuda()

    for param in vgg16_conv_4_3.parameters():
        param.requires_grad = False
    #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # L1_lossFn = nn.L1Loss().to(device)
    MSE_LossFn = nn.MSELoss().to(device)
    ComboLossFn = CombinedLoss().to(device)

    epochs = 150

    # Use below to increase learning rate if the stepsize was reduced too early
    # for param_group in optim.param_groups:
    #     param_group['lr'] = 1e-5

    for epoch in range(start_epoch, epochs):
        print(f'--------- New Epoch {epoch} Current lr: {optim.param_groups[0]["lr"]:.2e} ---------')
        step = 0

        # TODO: Modify train code to be able to train on more than a single intermediate frame
        # Model supports finding more than a single time step between two input frames.
        # Per Super-SloMo paper, training on up to 7 intermediate frames, may increase model accuracy
        # at least in their case.
        for indexes, (I0, It, I1) in trainloader:
            loss = 0
            itrs = 0
            for idx, f0, f_gt, f1 in zip(indexes, I0, It, I1):
                itrs += 1

                f0 = f0
                f1 = f1
                f_gt = f_gt

                f_int = model(f0.cuda(), f1.cuda())
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

                    with torch.no_grad():
                        MSE_val = MSE_LossFn(f_int, f_gt)
                        psnr = (10 * math.log10(1 / MSE_val.item()))
                        print(f'INDEX {idx}: psnr {psnr:2.4f}, '
                              f'charb {charLoss.item() / 1e3:6.4f}, '
                              f'combo {comboLoss / 10:6.4f}')

                        flipped = train_dataset.flip_map[idx.item()]

                        if flipped:

                            if train_dataset.is_randomcrop(idx.item()):
                                transforms.functional.to_pil_image(f0.squeeze(0)).transpose(
                                    [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM][flipped - 1]).save(
                                    f'debug/{idx}/Epoch{epoch:04d}_1Pre.png')
                                transforms.functional.to_pil_image(f1.squeeze(0)).transpose(
                                    [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM][flipped - 1]).save(
                                    f'debug/{idx}/Epoch{epoch:04d}_3Post.png')
                            transforms.functional.to_pil_image(f_int.squeeze(0)).transpose(
                                [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM][flipped - 1]).save(
                                f'debug/{idx}/Epoch{epoch:04d}_2int.png')
                        else:
                            if train_dataset.is_randomcrop(idx.item()):
                                transforms.functional.to_pil_image(f0.squeeze(0)).save(
                                    f'debug/{idx}/Epoch{epoch:04d}_1Pre.png')
                                transforms.functional.to_pil_image(f1.squeeze(0)).save(
                                    f'debug/{idx}/Epoch{epoch:04d}_3Post.png')
                            transforms.functional.to_pil_image(f_int.squeeze(0)).save(
                                f'debug/{idx}/Epoch{epoch:04d}_2int.png')

            step += 1
            optim.zero_grad()
            loss = loss / itrs
            loss.backward()
            optim.step()

            if step % 50 == 0:
                MSE_val = MSE_LossFn(f_int, f_gt)
                psnr = (10 * math.log10(1 / MSE_val.item()))
                print(f'step: {step:5d}, psnr {psnr:7.4f}, (last img: '
                      f'loss {comboLoss / 10 + charLoss / 1e3:8.4f} '
                      f'charb {charLoss.item() / 1e3:8.4f}, '
                      f'combo {comboLoss / 10:8.4f}) {"CROPPED" * train_dataset.is_randomcrop(idx.item())}')

        sched.step()

        # Save progress
        state = {'model': model.state_dict(),
                 'optim': optim.state_dict(),
                 'epoch': epoch + 1}
        torch.save(state, 'models' + f"/{args.model_name}{epoch:04d}.pth")
