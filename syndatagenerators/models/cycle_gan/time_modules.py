import torch as th
from torch import nn
from torch.nn.utils import parametrizations as param
import torch.nn.functional as F

import numpy as np



class TimeCycleDiscriminator(nn.Module):
    '''Discriminator for CycleGAN; similar to "normal DCGANDiscriminator but larger"'''
    def __init__(self, channels, depth=4, h_channels=64, in_size=1024) -> None:
        super(TimeCycleDiscriminator, self).__init__()

        kernel_w = 3
        
        last_size = int(np.ceil(in_size / (2**depth)))

        convs = [self._block(channels, h_channels, kernel_size=kernel_w, stride=1)]
        for i in range(depth):
            factor = 2**i
            convs.append(self._block(h_channels * factor, h_channels * factor * 2, kernel_size=kernel_w))

        convs.append(nn.Conv1d(h_channels * 2 ** depth, 1, kernel_size=3, padding=1, stride=1))
        
        self.convs = nn.Sequential(*convs) 

        self.last = nn.Linear(last_size, 1)

    def forward(self, x):
        # dont forget to reduce the dims
        bs, channels, ts = x.shape
        out = self.convs(x).view(bs,-1)
        out = self.last(out)
        return out.view(-1,1)


    def _block(self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        spectral_norm=True,
        batch_norm=True,
        bias=False,
        device='cpu') -> nn.Module:
        '''Returns a CONV block for the Discriminator'''

        conv = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=bias,
            device=device
        )
        if spectral_norm:
            conv = [param.spectral_norm(conv)]

        if batch_norm:
            conv = [
                *conv,
                nn.BatchNorm1d(num_features=out_channels)
            ]

        return nn.Sequential(*conv, nn.LeakyReLU(negative_slope=0.2))


class TimeCycleGenerator(nn.Module):
    ''' 
    wrapper for ResNet and U-Net Generators
    '''

    def __init__(self, generator_type: str, *args, **kwargs) -> None:
        super(TimeCycleGenerator, self).__init__()

        if generator_type == 'resnet':
            self.model = TimeResNetGenerator(*args, **kwargs)
            self.ex_weights = self.model.convs[0][0].weight
        elif generator_type == 'unet':
            self.model = TimeUnetGenerator(*args, **kwargs)
            self.ex_weights = self.model.unet.ublock[0].weight
        else:
            raise ValueError(f'No Generator for parameter {generator_type} found.')

    def forward(self, input):
        return self.model(input)

class TimeResNetGenerator(nn.Module):
    ''' Generator of the CycleGAN. consists of: Encoding, Residuals, Decoding. 
        layers almost identical to paper InstanceNorm? and ReflectPadding missing
    '''
    def __init__(self, src_channels, dst_channels, hidden_channels=64, depth=3, residuals=3, dilations=False, heads=4, norm=nn.BatchNorm1d, **kwargs) -> None:
        super(TimeResNetGenerator, self).__init__()
        # first and last layer are these big convolutions
        self.dilations = dilations
        # (b, c, ts)
        self.first = nn.Sequential(
            self._conv_block(src_channels, hidden_channels, kernel_size=7, padding=3, stride=1, norm=None),
            nn.ReLU()
        )
        # (b, hidden, ts)
        convs = []
        dils = []
        prev = hidden_channels
        downsampling = depth - (0 if dilations else 1)
        for c in range(downsampling):
            factor = 2 ** c
            channels = min(hidden_channels * factor * 2, hidden_channels * 4)
            convs += [
                self._conv_block(prev, channels, norm=norm),
                nn.ReLU()
            ]
            if dilations:
                dils += [
                    self._conv_block(prev, channels, padding=factor * 2, dilation=factor * 2, norm=norm),
                    nn.ReLU()
                ]
            prev = channels
                
        self.convs = nn.Sequential(*convs)
        # (b, hidden * 2**downsampling, ts / 2**downsampling) (* 2 when dilation net is used) 

        factor = 2 ** (downsampling)

        # attention implementation. instead of looping over multiple heads one can increace the embedding dim
        if dilations:
            self.dils = nn.Sequential(*dils)
            self.attention = TimeAttentionBlock(prev * 2, int(prev), heads, prev, norm=norm)

        res = []
        for r in range(residuals):
            res += [
                TimeResidualBlock(conv_channels=prev, norm=norm)
            ]
        self.res = nn.Sequential(*res)
        # (b, hidden * 2**downsampling, ts / 2**downsampling)
        deconv = []
        for d in range(0, downsampling):
            factor = 2 ** (downsampling - d)
            channels = max(hidden_channels * factor, hidden_channels * 2)
            deconv += [
                self._deconv_block(prev, int(channels / 2), norm=norm),
                nn.ReLU()
            ]
            prev = int(channels / 2)
        
        self.deconv = nn.Sequential(*deconv)
        # (b, hidden, ts)
        last = [
            self._conv_block(hidden_channels, dst_channels, kernel_size=7, padding=3, stride=1, norm=None),
        ]
        self.last = nn.Sequential(*last)
        # (b, c, ts)
        

    def forward(self, x) -> th.Tensor:
        transform = self.first(x)
        out = self.convs(transform)
        if self.dilations:
            dilated = self.dils(transform)
            attntn = self.attention(th.concat((out,dilated), dim=1))
            out = attntn
        out = self.res(out)
        out = self.deconv(out)
        out = self.last(out)
        return out

    def _conv_block(self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=1,
        norm=nn.BatchNorm1d,
        device='cpu') -> nn.Module:
        '''simple CONV block of the encoder'''
        bias = norm != nn.BatchNorm1d

        layers = []

        layers.append(
            nn.Conv1d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding,
                dilation=dilation,
                bias=bias,
                device=device
            )
        )
        if norm != None:
            layers.append(norm(out_channels))

        return nn.Sequential(*layers)

    def _deconv_block(self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        out_padding=1,
        norm=nn.BatchNorm1d,
        device='cpu'
    ) -> nn.Module:
        '''Simple DECONV block of the decoder'''
        bias = norm != nn.BatchNorm1d

        deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=out_padding,
            output_padding=padding,
            bias=bias,
            device=device
        )
        layers = [deconv]
        if norm != None:
            bn = norm(num_features=out_channels)
            layers.append(bn)

        return nn.Sequential(*layers)
    


class TimeResidualBlock(nn.Module):
    '''
    defines a residual block consisting of two conv layers, first activated by relu 
    '''

    def __init__(self,
        conv_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        norm=nn.BatchNorm1d,
        device='cpu'
    ) -> None:
        super(TimeResidualBlock, self).__init__()
        bias = norm != nn.BatchNorm1d
        self.conv1 = nn.Conv1d(
            in_channels=conv_channels, 
            out_channels=conv_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=bias
        )
        self.norm1 = norm(conv_channels)
        self.conv2 = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.norm2 = norm(conv_channels)
        

    def forward(self, x) -> th.Tensor:
        res = self.conv1(x)
        res = F.relu(self.norm1(res))
        res = self.conv2(res)
        res = self.norm2(res)
        return F.relu(res + x)      # skip connection
    
class TimeAttentionBlock(nn.Module):
    '''A attention block implementing CHANNEL-wise multihead self-attention'''
    def __init__(self, in_channels, h_channels, heads, out_channels, norm=nn.BatchNorm1d) -> None:
        super(TimeAttentionBlock, self).__init__()
        bias = not norm == nn.BatchNorm1d
        self.h_channels = h_channels
        self.Wk = nn.Conv1d(in_channels, h_channels * heads,1,1,0, bias=bias)
        self.Wq = nn.Conv1d(in_channels, h_channels * heads,1,1,0, bias=bias)
        self.Wv = nn.Conv1d(in_channels, h_channels * heads,1,1,0, bias=bias)

        self.combine = nn.Conv1d(h_channels * heads, in_channels,1,1,0, bias=bias)
        self.norm1 = norm(in_channels)
        #reduce channels as last step 
        self.last = nn.Conv1d(in_channels, out_channels,1,1,0, bias=bias)
        self.norm2 = norm(out_channels)

    def forward(self, x):
        k = self.Wk(x)
        q = self.Wq(x)
        v = self.Wv(x)

        k_v_pair = F.softmax(q @ k.mT, dim=1)
        attention_vec = (k_v_pair @ v) / (self.h_channels ** 0.5)
        out = self.combine(attention_vec)
        out = self.norm1(out) + x
        return F.relu(self.norm2(self.last(out)))

class TimeUnetGenerator(nn.Module):
    '''U-Net architecture of the Generator'''
    def __init__(
        self,
        src_channels,
        dst_channels,
        time_steps,
        hidden_channels=64,
        depth=3,
        residuals=3,
        norm=nn.BatchNorm1d,
        **kwargs
    ) -> None:
        super(TimeUnetGenerator, self).__init__()

        # inner
        block = UnetBlock(hidden_channels * (2 ** depth), hidden_channels * (2 ** depth), int(time_steps / (2 ** (depth+1))), input_nc=None, submodule=None, norm_layer=norm, innermost=True)
        
        for i in range(residuals):
            block = UnetBlock(hidden_channels * (2 ** depth), hidden_channels * (2 ** depth), int(time_steps / (2 ** (depth+1))), input_nc=None, reduction=False, submodule=block, norm_layer=norm)

        for i in np.arange(1, depth+1, 1, dtype=int)[::-1]: # itterate from back to front
            block = UnetBlock(hidden_channels * (2 ** (i-1)), hidden_channels * (2 ** i), int(time_steps / (2 ** (i))), input_nc=None, submodule=block, norm_layer=norm)


        self.unet = UnetBlock(dst_channels, hidden_channels, time_steps, input_nc=src_channels, submodule=block, outermost=True, norm_layer=norm)

    def forward(self, input):
        return self.unet(input)


class UnetBlock(nn.Module):  # stolen from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/e2c7618a2f2bf4ee012f43f96d1f62fd3c3bec89/models/networks.py#L469

    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, ts_length, input_nc=None, reduction=True,
                submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm1d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """

        super(UnetBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        if reduction == True:
            stride = 2
            kernel_size = 4
            out_padding = ts_length % 2
        else:
            stride = 1
            kernel_size = 3
            out_padding = 0
        downrelu = nn.LeakyReLU(0.2, True)
        downconv = nn.Conv1d(input_nc, inner_nc, kernel_size=kernel_size,
                            stride=stride, padding=1, bias=use_bias)
        downnorm = norm_layer(inner_nc)

        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:   # upscale 2x and embedd submodule
            upconv = nn.ConvTranspose1d(inner_nc * 2, outer_nc,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=1, output_padding=out_padding)

            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up

        elif innermost: # no embed submodule
            upconv = nn.ConvTranspose1d(inner_nc, outer_nc,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=1, output_padding=out_padding, bias=use_bias)

            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up

        else: # same as outermost + additional ReLU activation 
            upconv = nn.ConvTranspose1d(inner_nc * 2, outer_nc,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=1, output_padding=out_padding, bias=use_bias)

            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.ublock = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.ublock(x)

        else:   # add skip connections
            return th.cat([x, self.ublock(x)], 1)
