import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.parameter import Parameter

from models.softsplat import ModuleSoftsplat
from models.unet import UNet

# Store grids in variables, saves time recreating them every call to warp
grids = {}

backwarp_tenGrid = {}


def backwarp(tenInput, tenFlow):
    if str(tenFlow.size()) not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(
            tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(
            tenFlow.shape[0], -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.size())] = torch.cat([tenHorizontal, tenVertical], 1).cuda()
    # end

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tenInput,
                                           grid=(backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1),
                                           mode='bilinear', padding_mode='zeros', align_corners=True)


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
        self.beta = Parameter(torch.ones(1), requires_grad=True)
        self.softmax = ModuleSoftsplat('softmax')
        self.final = UNet(9, 3, 4)
        self.use_cuda = use_cuda

    def process(self, x0, x1, t):
        x = torch.cat((x0, x1), 1)

        Flow = self.Flow(x)

        Flow_0_1, Flow_1_0 = Flow[:, :2, :, :], Flow[:, 2:4, :, :]
        # cv2.imshow(winname='softmax', mat=Flow_0_1[0, :, :, :].cpu().detach().numpy().transpose(1, 2, 0))

        Flow_t_0 = -(1 - t) *  t      * Flow_0_1 + t * t       * Flow_1_0
        Flow_t_1 =  (1 - t) * (1 - t) * Flow_0_1 - t * (1 - t) * Flow_1_0
        # cv2.imshow(winname='softmaFt0', mat=Flow_t_0[0, :, :, :].cpu().detach().numpy().transpose(1, 2, 0))

        Flow_t = torch.cat((Flow_t_0, Flow_t_1, x), 1)
        Flow_t = self.refine_flow(Flow_t)

        Flow_t_0 = Flow_t_0 + Flow_t[:, :2, :, :]
        Flow_t_1 = Flow_t_1 + Flow_t[:, 2:4, :, :]

        # xt1 = warp(x0, Flow_t_0, self.use_cuda)
        xt2 = warp(x1, Flow_t_1, self.use_cuda)
        # tenMetric = torch.nn.functional.l1_loss(input=tenFirst, target=backwarp(tenInput=tenSecond, tenFlow=tenFlow),
        #                                         reduction='none').mean(1, True)

        # metric = torch.nn.functional.l1_loss(input=x0, target=xt1, reduction='none').mean(1, True)
        # print(metric)
        metric = torch.nn.functional.l1_loss(input=x0, target=xt2, reduction='none').mean(1, True)
        # print(metric)

        xt1 = self.softmax(x0, Flow_t_0, metric * self.beta)
        # xt2 = self.softmax(x1, Flow_t_1, metric * self.beta)
        # cv2.imshow(winname='softmaFt0', mat=xt1[0, :, :, :].cpu().detach().numpy().transpose(1, 2, 0))
        # cv2.imshow(winname='softmaFt1', mat=xt2[0, :, :, :].cpu().detach().numpy().transpose(1, 2, 0))
        # cv2.waitKey(delay=0)

        # xt1 = warp(x0, Flow_t_0, self.use_cuda)
        # xt2 = warp(x1, Flow_t_1, self.use_cuda)

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
