import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.unet import UNet

# Store grids in variables, saves time recreating them every call to warp
grids = {}


def warp(img, flow, cuda):
    global grids
    _, _, H, W = img.size()
    dims = H, W

    if dims in grids:
        gridX, gridY = grids[dims]
    else:

        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))

        if cuda:
            gridX = torch.tensor(gridX, requires_grad=False).cuda()
            gridY = torch.tensor(gridY, requires_grad=False).cuda()
        else:
            gridX = torch.tensor(gridX, requires_grad=False)
            gridY = torch.tensor(gridY, requires_grad=False)
        grids[dims] = gridX, gridY

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]
    x = gridX.unsqueeze(0).expand_as(u).float() + u
    y = gridY.unsqueeze(0).expand_as(v).float() + v
    normx = 2 * (x / W - 0.5)
    normy = 2 * (y / H - 0.5)
    grid = torch.stack((normx, normy), dim=3)
    warped = F.grid_sample(img, grid)
    return warped


class Net(nn.Module):
    def __init__(self, use_cuda=True):
        super(Net, self).__init__()
        self.Mask = UNet(16, 2, 4)
        self.Flow = UNet(6, 4, 5)
        self.refine_flow = UNet(10, 4, 4)
        self.final = UNet(9, 3, 4)
        self.use_cuda = use_cuda

    def process(self, x0, x1, t):
        x = torch.cat((x0, x1), 1)

        Flow = self.Flow(x)

        Flow_0_1, Flow_1_0 = Flow[:, :2, :, :], Flow[:, 2:4, :, :]

        Flow_t_0 = -(1 - t) * t * Flow_0_1 + t * t * Flow_1_0
        Flow_t_1 = (1 - t) * (1 - t) * Flow_0_1 - t * (1 - t) * Flow_1_0

        Flow_t = torch.cat((Flow_t_0, Flow_t_1, x), 1)
        Flow_t = self.refine_flow(Flow_t)

        Flow_t_0 = Flow_t_0 + Flow_t[:, :2, :, :]
        Flow_t_1 = Flow_t_1 + Flow_t[:, 2:4, :, :]

        xt1 = warp(x0, Flow_t_0, self.use_cuda)
        xt2 = warp(x1, Flow_t_1, self.use_cuda)

        temp = torch.cat((Flow_t_0, Flow_t_1, x, xt1, xt2), 1)

        Mask = torch.sigmoid(self.Mask(temp))

        w1, w2 = (1 - t) * Mask[:, 0:1, :, :], t * Mask[:, 1:2, :, :]
        output = (w1 * xt1 + w2 * xt2) / (w1 + w2 + 1e-8)

        return output

    def forward(self, input0, input1, t=0.5):
        output = self.process(input0, input1, t)
        compose = torch.cat((input0, input1, output), 1)
        final = self.final(compose) + output
        final = final.clamp(0, 1)

        return final

    def forward_chop(self, x0, x1, t=0.5, padding=200, min_size=320_000):
        """
        Performs a memory efficient forward where frames are split into smaller patches.
        Degrades performance and the result of the model, due to information potentially being lost between a patch.
        High padding values may reduce the output degradation, but adds more work.

        min_size is maximum allowed total pixels of a patch. If less is memory available, reduce this value.
        min_size has to be at least 160000 with a padding of 200. (Gives roughly 400 x 400 pixel patches)

        For reference, resolutions and their pixel counts.
        1080p = 2 073 600 pixels
        720p  =   921 600 pixels
        720p  = 1 361 600 pixels (200 pixel padding padding)
        480p  =   307 200 pixels
        480p  =   571 200 pixles (200 pixel padding padding)

        RRIN can use just under 11 GB with 1080p videos, if you go such a graphics card, then 2M pixels is worth a try.
        If you are unable to do 1080p videos, reduce min_size til you get good results.
        """

        b, c, h, w = x0.size()

        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + padding, w_half + padding

        # Adds padding to fit dimensions required by model (UNet specifically)
        if w_size % 2 ** 4 != 0:
            w_size = (w_size // 2 ** 4 + 1) * 2 ** 4

        if h_size % 2 ** 4 != 0:
            h_size = (h_size // 2 ** 4 + 1) * 2 ** 4

        # Prevent padding from going beyond image dims (in this case, chop may not be needed.)
        # The source input is already padded by dataloader to fid the model
        h_size, w_size = min(h_size, h), min(w_size, h)

        x0_list = [
            x0[:, :, 0:h_size, 0:w_size],
            x0[:, :, 0:h_size, (w - w_size):w],
            x0[:, :, (h - h_size):h, 0:w_size],
            x0[:, :, (h - h_size):h, (w - w_size):w]]

        x1_list = [
            x1[:, :, 0:h_size, 0:w_size],
            x1[:, :, 0:h_size, (w - w_size):w],
            x1[:, :, (h - h_size):h, 0:w_size],
            x1[:, :, (h - h_size):h, (w - w_size):w]]

        # print(w_size, h_size)
        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4):
                sr_batch = self.forward(x0_list[i], x1_list[i], t=t)
                sr_list.extend(sr_batch.chunk(1, dim=0))
        else:
            sr_list = [
                self.forward_chop(*patches, t=t, padding=padding, min_size=min_size)
                for patches in zip(x0_list, x1_list)
            ]

        output = x1.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output
