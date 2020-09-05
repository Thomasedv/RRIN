import torch
import PIL.Image as Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

from math import ceil
from .utils import *
from .flow_utils import *
from .model import DynFilter, DFNet, BMNet


# Lazy workaround
class temp:
    dict = None


args = temp
args.dict = dict()

torch.backends.cudnn.benchmark = True

args.dict['context_layer'] = nn.Conv2d(3, 64, (7, 7), stride=(1, 1), padding=(3, 3), bias=False)
args.dict['BMNet'] = BMNet()
args.dict['DF_Net'] = DFNet(32, 4, 16, 6)
args.dict['filtering'] = DynFilter()

args.dict['context_layer'].load_state_dict(torch.load('models/BMBC/Weights/context_layer.pth'))
args.dict['BMNet'].load_state_dict(torch.load('models/BMBC/Weights/BMNet_weights.pth'))
args.dict['DF_Net'].load_state_dict(torch.load('models/BMBC/Weights/DFNet_weights.pth'))
ReLU = torch.nn.ReLU()

for param in args.dict['context_layer'].parameters():
    param.requires_grad = False
for param in args.dict['BMNet'].parameters():
    param.requires_grad = False
for param in args.dict['DF_Net'].parameters():
    param.requires_grad = False

if torch.cuda.is_available():
    args.dict['BMNet'].cuda()
    args.dict['DF_Net'].cuda()
    args.dict['context_layer'].cuda()
    args.dict['filtering'].cuda()
    ReLU.cuda()


class Net(nn.Module):
    def forward(self, I0, I1, t, **kwargs):
        args.time_step = t

        x = torch.cat((I0, I1), dim=1)
        # F_0_1 = args.dict['BMNet'](F.interpolate(torch.cat((I0, I1), dim=1), (H_, W_), mode='bilinear'), time=0) * 2.0
        F_0_1 = args.dict['BMNet'](x, time=0) * 2.0
        F_1_0 = args.dict['BMNet'](x, time=1) * (-2.0)
        BM = args.dict['BMNet'](x, time=args.time_step)  # V_t_1

        C1 = warp(torch.cat((I0, ReLU(args.dict['context_layer'](I0))), dim=1), (-args.time_step) * F_0_1)  # F_t_0
        C2 = warp(torch.cat((I1, ReLU(args.dict['context_layer'](I1))), dim=1), (1 - args.time_step) * F_0_1)  # F_t_1
        C3 = warp(torch.cat((I0, ReLU(args.dict['context_layer'](I0))), dim=1), (args.time_step) * F_1_0)  # F_t_0
        C4 = warp(torch.cat((I1, ReLU(args.dict['context_layer'](I1))), dim=1), (args.time_step - 1) * F_1_0)  # F_t_1
        C5 = warp(torch.cat((I0, ReLU(args.dict['context_layer'](I0))), dim=1), BM * (-2 * args.time_step))
        C6 = warp(torch.cat((I1, ReLU(args.dict['context_layer'](I1))), dim=1), BM * 2 * (1 - args.time_step))

        input = torch.cat((I0, C1, C2, C3, C4, C5, C6, I1), dim=1)
        DF = F.softmax(args.dict['DF_Net'](input), dim=1)

        candidates = input[:, 3:-3, :, :]

        R = args.dict['filtering'](candidates[:, 0::67, :, :], DF)
        G = args.dict['filtering'](candidates[:, 1::67, :, :], DF)
        B = args.dict['filtering'](candidates[:, 2::67, :, :], DF)

        I2 = torch.cat((R, G, B), dim=1)

        return I2
