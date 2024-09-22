import random
import math
from layers import *


# ------------------- #
#   ResNet Example    #
# ------------------- #
def sew_function(x: torch.Tensor, y: torch.Tensor):
    return x + y

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
# for basic
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            # norm_layer = tdBatchNorm
            norm_layer = tdBatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.conv2_s = tdLayer(self.conv2, self.bn2)
        self.downsample = downsample

        # for LIF
        self.spike1 = LIFSpike()
        self.spike2 = LIFSpike()

    def forward(self, x):
        
        identity = x
        

        out = self.conv1_s(x)
        out = self.spike1(out)
     
        
        out = self.conv2_s(out)

        if self.downsample is not None:
            identity = self.downsample(x)




        out = out + identity
        
        out = self.spike2(out)
        
         
        return out

# for sew-network
# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(BasicBlock, self).__init__()
#         if norm_layer is None:
#             # norm_layer = tdBatchNorm
#             norm_layer = tdBatchNorm
#         if groups != 1 or base_width != 64:
#             raise ValueError('BasicBlock only supports groups=1 and base_width=64')
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes, )
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes, )
#         self.downsample = downsample
#         self.stride = stride

#         self.conv1_s = tdLayer(self.conv1, self.bn1)
#         self.conv2_s = tdLayer(self.conv2, self.bn2)    
#         self.spike1 = LIFSpike()
#         self.spike2 = LIFSpike()




        
        

#     def forward(self, x):
        
#         identity = x
        

#         out = self.conv1_s(x)
#         out = self.spike1(out)
        
#         out = self.conv2_s(out)
#         out = self.spike2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)



#         out = sew_function(identity, out)
        

#         return out

### start net1:resnet19
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = tdBatchNorm
        self._norm_layer = norm_layer
        
        self.inplanes = 64
        
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.conv1_s = tdLayer(self.conv1, self.bn1)
      
  
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        

        
        self.avgpool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))
        self.pool = SeqToANNContainer(nn.AvgPool2d(2))
        
        self.fc1 = nn.Linear(512 * block.expansion, 256)
        self.fc1_s = tdLayer(self.fc1)
        self.fc2 = nn.Linear(256, num_classes)
        self.fc2_s = tdLayer(self.fc2)

        
        self.spike1 = LIFSpike()
        self.spike2 = LIFSpike()              
        self.T = 1


        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            print('YES')
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tdLayer(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        x = self.conv1_s(x)
        
        x = self.spike1(x)

        # 
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        
        x = self.avgpool(x)
        x = torch.flatten(x, 2) 
       
              
        x = self.fc1_s(x)
        x = self.spike2(x) 
        
      
        x = self.fc2_s(x)

            
        return x

    def forward(self, x):
        y = add_dimention(x, self.T)
        return self._forward_impl(y)


def _resnet( block, layers,  **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def resnet19(**kwargs):
    return _resnet(BasicBlock, [3, 3, 2],**kwargs)

# end net1:resnet19

## Resnet-20
class ResNet_Cifar_Modified(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_Cifar_Modified, self).__init__()
        if norm_layer is None:
            norm_layer = tdBatchNorm
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        reduction=16
        self.att = MultiSpectralAttentionLayer(self.inplanes, c2wh[self.inplanes], c2wh[self.inplanes],  
    reduction=reduction, freq_sel_method = 'top16')
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # 1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.conv1_s = tdLayer(self.conv1, self.bn1)
        # 2.
        self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = norm_layer(self.inplanes)
        self.conv2_s = tdLayer(self.conv2, self.bn2)
        # 3.
        self.conv3 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn3 = norm_layer(self.inplanes)
        self.conv3_s = tdLayer(self.conv3, self.bn3)
        
  
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        
        self.avgpool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))
        self.pool = SeqToANNContainer(nn.AvgPool2d(2))
        
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc1_s = tdLayer(self.fc1)

        self.spike1 = LIFSpike()
        self.spike2 = LIFSpike()
        self.spike3 = LIFSpike()
        # for burst spike
        # self.spike1 = BurstSpike()
        # self.spike2 = BurstSpike()
   
        self.T = 1



    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tdLayer(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        
        
        x = self.conv1_s(x)

        x= self.spike1(x)

        
        x = self.conv2_s(x)
        x = self.spike2(x)
        
        
        x = self.conv3_s(x)
        x = self.spike3(x)
        
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        
        x = torch.flatten(x, 2)       
        x = self.fc1_s(x)
        # x = self.spike2(x) 
        # x = self.fc2_s(x)

            
        return x

    def forward(self, x):
        y = add_dimention(x, self.T)
        # print(y.shape)
        return self._forward_impl(y)


def resnet20(**kwargs):
    model = ResNet_Cifar_Modified(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


# ms-resnet
class BasicBlock_MS(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock_MS, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = SeqToANNContainer(conv3x3(inplanes, planes, stride))
        self.bn1 = SeqToANNContainer(norm_layer(planes))
        self.conv2 = SeqToANNContainer(conv3x3(planes, planes))
        self.bn2 = SeqToANNContainer(norm_layer(planes))
        self.downsample = downsample
        self.stride = stride
        self.spike1 = LIFSpike()
        self.spike2 = LIFSpike()

    def forward(self, x):
            identity = x
            out = self.spike1(x)
            out = self.bn1(self.conv1(out))
            out = self.spike2(out)
            out = self.bn2(self.conv2(out))

            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity

            return out

class MS_ResNet(nn.Module):
    def __init__(self, block, layers, T=6,num_classes=100,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(MS_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = SeqToANNContainer(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False))

        self.bn1 = SeqToANNContainer(norm_layer(self.inplanes))

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = SeqToANNContainer((nn.AdaptiveAvgPool2d((1, 1))))

        self.fc1 = SeqToANNContainer(nn.Linear(512*block.expansion, num_classes))

        self.spike = LIFSpike()
        self.T = T
        

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = SeqToANNContainer(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.spike(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.spike(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc1(x)
        return x

    def forward(self, x):
        y = add_dimention(x, self.T)
        return self._forward_impl(y)


def ms_resnet(arch, block, layers, **kwargs):
    model = MS_ResNet(block, layers, **kwargs)
    return model


def msresnet18(pretrained=False, progress=True, **kwargs):
    return ms_resnet('resnet18', BasicBlock_MS, [3, 3, 2],
                   **kwargs)


if __name__ == '__main__':
    model = resnet19(num_classes=100)
    model.T = 2
    
    #print(model)
    x = torch.rand(10,3,32,32)
    y = model(x)
    print(y.shape)
