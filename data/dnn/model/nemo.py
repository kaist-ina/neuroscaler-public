import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, num_channels, kernel_size):
        super(ResBlock, self).__init__()

        layers = []
        layers.append(nn.Conv2d(num_channels, num_channels, kernel_size, padding=(kernel_size//2)))
        layers.append(nn.ReLU(True)) #TODO: does this harm backpropagation?
        layers.append(nn.Conv2d(num_channels, num_channels, kernel_size, padding=(kernel_size//2)))

        self.body = nn.Sequential(*layers)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class UpsampleBlock(nn.Module):
    def __init__(self, num_channels, kernel_size, scale):
        super(UpsampleBlock, self).__init__()

        layers = []
        layers.append(nn.Conv2d(num_channels, (scale**2) * 3, kernel_size, padding=(kernel_size//2)))
        layers.append(nn.PixelShuffle(scale))

        self.body = nn.Sequential(*layers)

    def forward(self, x):
        x = self.body(x)
        return x

class NEMO(nn.Module):
    def __init__(self, args):
        super(NEMO, self).__init__()

        num_blocks = args.num_blocks
        num_channels = args.num_channels
        scale = args.scale
        kernel_size = 3

        layers1 = [nn.Conv2d(3, num_channels, kernel_size, padding=(kernel_size//2))]
        layers2 = [ResBlock(num_channels, kernel_size) for _ in range(num_blocks)] \
                + [nn.Conv2d(num_channels, num_channels, kernel_size, padding=(kernel_size//2))]
        layers3 = [UpsampleBlock(num_channels, kernel_size, scale)]

        self.head = nn.Sequential(*layers1)
        self.body = nn.Sequential(*layers2)
        self.tail = nn.Sequential(*layers3)
        self.name = f'NEMO_B{num_blocks}_F{num_channels}_S{scale}'

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

        return x

if __name__ == '__main__':
    device = torch.device('cuda:0')
    dtype = torch.float
    input_tensor = torch.rand((1, 3, 100, 100), device=device, dtype=dtype)
    class arguments: pass
    args = arguments()
    args.num_channels = 48
    args.num_blocks = 8
    args.scale = 3
    model = NEMO(args).to(device)
    output_tensor = model(input_tensor)
    print(output_tensor.size())
