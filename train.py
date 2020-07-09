import os

import math
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.optim.adamw import AdamW
from torch.utils.data import Dataset
from torchvision import transforms

from dataloader import TrainDataloader
from losses import CombinedLoss, charbonnierLoss
from model import Net


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True


def train(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print('Cuda enabled!')

    train_dataset = TrainDataloader(path=args.train_folder, cuda=use_cuda)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False,
                                              num_workers=3)

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
        print(f'Starting new model: "{args.model_name}"')
        state = {}

    model = model.cuda()
    model.train()

    start_epoch = state.get('epoch', 1)

    optim = AdamW(model.parameters(), lr=1e-4)
    for param_group in optim.param_groups:
        param_group['initial_lr'] = 1e-4

    sched = torch.optim.lr_scheduler.MultiStepLR(optim, [15, 25, 35], gamma=0.1, last_epoch=start_epoch)

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

    L1_lossFn = nn.L1Loss().to(device)
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
        for indexes, (f0, f_gt, f1), flipped in trainloader:

            f0 = f0.cuda(non_blocking=True)
            f1 = f1.cuda(non_blocking=True)
            f_gt = f_gt.cuda(non_blocking=True)

            f_int = model(f0, f1)

            prcpLoss = L1_lossFn(vgg16_conv_4_3(f_int), vgg16_conv_4_3(f_gt)) * 20
            charLoss = charbonnierLoss(f_int, f_gt) / 1e3
            comboLoss = ComboLossFn(f_int, f_gt) / 10

            loss = comboLoss + charLoss + prcpLoss

            for pos, (idx, flip) in enumerate(zip(indexes, flipped)):
                if idx % 750 == 0:
                    print(f'IDX: {idx} | Debug image saved!')
                    if not os.path.exists(f'debug/{idx}'):
                        os.makedirs(f'debug/{idx}')

                    with torch.no_grad():
                        if flip:
                            if train_dataset.is_randomcrop(idx.item()):
                                transforms.functional.to_pil_image(f0[pos].detach().cpu()).transpose(
                                    [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM][flip - 1]).save(
                                    f'debug/{idx}/Epoch{epoch:04d}_1Pre.png')
                                transforms.functional.to_pil_image(f1[pos].detach().cpu()).transpose(
                                    [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM][flip - 1]).save(
                                    f'debug/{idx}/Epoch{epoch:04d}_3Post.png')
                            transforms.functional.to_pil_image(f_int[pos].detach().cpu()).transpose(
                                [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM][flip - 1]).save(
                                f'debug/{idx}/Epoch{epoch:04d}_2int.png')
                        else:
                            if train_dataset.is_randomcrop(idx.item()):
                                transforms.functional.to_pil_image(f0[pos].detach().cpu()).save(
                                    f'debug/{idx}/Epoch{epoch:04d}_1Pre.png')
                                transforms.functional.to_pil_image(f1[pos].detach().cpu()).save(
                                    f'debug/{idx}/Epoch{epoch:04d}_3Post.png')
                            transforms.functional.to_pil_image(f_int[pos].detach().cpu()).save(
                                f'debug/{idx}/Epoch{epoch:04d}_2int.png')

            step += 1
            optim.zero_grad()
            # loss = loss / itrs
            loss.backward()
            optim.step()

            if step % 50 == 0:
                with torch.no_grad():
                    MSE_val = MSE_LossFn(f_int, f_gt)
                    psnr = (10 * math.log10(1 / MSE_val.item()))

                    print(f'step: {step:5d}, psnr {psnr:7.4f}, '
                          f'(Last Image | '
                          f'Total Loss: {comboLoss + charLoss + prcpLoss.item():8.4f} | '
                          f'charb: {charLoss.item() :8.4f}, '
                          f'combo: {comboLoss:8.4f}, '
                          f'prcp: {prcpLoss.item():8.4f}), '
                          f'{"CROPPED" * train_dataset.is_randomcrop(idx.item())}')

        sched.step()

        # Save progress
        state = {'model': model.state_dict(),
                 'optim': optim.state_dict(),
                 'epoch': epoch + 1}
        torch.save(state, 'models' + f"/{args.model_name}{epoch:04d}.pth")

    print('Training finished!')