import torch
import torch.nn as nn

""" Resnet architecture used by FPN for image segmentation """


class BottleNeck(nn.Module):
    
    # expansion factor of the number of output channels
    expansion = 4

    def __init__(self, in_ch, out_ch,
                 stride=1, is_first_block=False):
        """
        Implements a bottleneck residual block with skip connection for larger resnet arch.

        Args: 
            in_ch: number of input channels
            out_ch: number of output channels
            stride: stride using in (a) the first 3x3 convolution and 
                    (b) 1x1 convolution used for downsampling for skip connection
            is_first_block: whether it is the first residual block of the layer
        """

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch,
                         out_channels=out_ch,
                         kernel_size=1,
                         stride=1,
                         padding=0)
        self.bn1 = nn.BatchNorm2d(out_ch) 
     
        self.conv2 =  nn.Conv2d(in_channels=out_ch,
                         out_channels=out_ch,
                         kernel_size=3,
                         stride=stride,
                         padding=1)          
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.conv3 = nn.Conv2d(in_channels=out_ch,
                         out_channels=out_ch*self.expansion,
                         kernel_size=1,
                         stride=1,
                         padding=0)
        self.bn3 = nn.BatchNorm2d(out_ch*self.expansion) 

        self.relu = nn.ReLU()

        # Skip connection goes through 1x1 convolution with stride=2 for 
        # the first blocks of conv3, conv4, and conv5 layers in the bottom-up network. 
        # This ensures that the spatial dimension of feature maps and number of channels 
        # are the same for performing the element-wise add operations in the top-down network.
        self.downsample = None
        if is_first_block:
           self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels=in_ch,
                         out_channels=out_ch*self.expansion,
                         kernel_size=1,
                         stride=stride,
                         padding=0),
                    nn.BatchNorm2d(out_ch*self.expansion)) 


    def forward(self, x):
        """
        Args: 
          x: input tensor
        Returns:
          residual block output  
        """

        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.downsample:
            identity = self.downsample(identity)
  
        #skip connection
        x += identity
        x = self.relu(x)

        return x  


class BasicBlock(nn.Module):
    # expansion factor of the number of output channels
    expansion = 1

    def __init__(self, in_ch, out_ch,
                 stride=1, is_first_block=False):
        """
        Implements a basic residual block with skip connection for resnet18 and resnet34.

        Args: 
            in_ch: number of input channels
            out_ch: number of output channels
            stride: stride using in (a) the first 3x3 convolution and 
                    (b) 1x1 convolution used for downsampling for skip connection
            is_first_block: whether it is the first residual block of the layer
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch,
                         out_channels=out_ch,
                         kernel_size=3,
                         stride=stride,
                         padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch) 
        self.conv2 =  nn.Conv2d(in_channels=out_ch,
                         out_channels=out_ch,
                         kernel_size=3,
                         stride=1,
                         padding=1)          
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

        # Skip connection goes through 1x1 convolution with stride=2 for 
        # the first blocks of conv3, conv4, and conv5 layers in the bottom-up network. 
        # This ensures that the spatial dimension of feature maps and number of channels 
        # are the same for performing the element-wise add operations in the top-down network.
        self.downsample = None
        if is_first_block and stride != 1:
           self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels=in_ch,
                         out_channels=out_ch,
                         kernel_size=1,
                         stride=stride,
                         padding=0),
                    nn.BatchNorm2d(out_ch)) 


    def forward(self, x):
        """
        Args: 
          x: input tensor
        Returns:
          residual block output  
        """

        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample:
           identity = self.downsample(identity)
  
        #skip connection
        x += identity
        x = self.relu(x)

        return x  

         
class ResNet(nn.Module):

    def __init__(self, ResBlock, inp_ch=3, out_ch=[64, 128, 256, 512], nblocks_list=[3, 4, 6, 3]):
        """ 
        Create a conv block consisting of multiple conv layers.
        Args: 
            ResBlock: residual block type, BasicBlock for ResNet-18, 34 or 
                      BottleNeck for ResNet-50, 101, 152
            inp_ch: number of input channels
            out_ch: number of output channels for each block
            stride: stride used in the first 3x3 convolution of the first resdiual block
                    of the layer and 1x1 convolution for skip connection in that block
            nblocks_list: number of residual blocks for each conv block
        Returns: 
            convolutional block
        """

        super().__init__()

        #first conv layer
        self.conv1 = nn.Sequential(
             nn.Conv2d(in_channels=inp_ch, out_channels=out_ch[0], kernel_size=7, stride=2, padding=3),
             nn.BatchNorm2d(out_ch[0]),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        )

        #create 4 conv blocks
        in_channels = out_ch[0]
        # For the first block of conv2, do not downsample and use stride=1.
        self.conv2 = self.createConvBlock(ResBlock, nblocks_list[0], in_channels, out_ch[0], stride=1)
        # For the first blocks of conv3 - conv5, perform downsampling using stride=2.
        # By default, ResBlock.expansion = 4 for ResNet-50, 101, 152; ResBlock.expansion = 1 for ResNet-18, 34.
        self.conv3 = self.createConvBlock(ResBlock, nblocks_list[1], out_ch[0]*ResBlock.expansion, out_ch[1], stride=2)
        self.conv4 = self.createConvBlock(ResBlock, nblocks_list[2], out_ch[1]*ResBlock.expansion, out_ch[2], stride=2)
        self.conv5 = self.createConvBlock(ResBlock, nblocks_list[3], out_ch[2]*ResBlock.expansion, out_ch[3], stride=2)


    def forward (self, x):
        """ 
        Bottom-up feature maps using Resnet backbone 
        x: input image

        Returns: the convolutional feature maps for FPN

        """
        #print("Resnet input", x.shape)
        x = self.conv1(x)
        #print("conv1", x.shape)
        #feature maps for FPN
        C2 = self.conv2(x)
        #print("conv2", C2.shape)
        C3 = self.conv3(C2)
        #print("conv3", C3.shape)
        C4 = self.conv4(C3)
        #print("conv4", C4.shape)
        C5 = self.conv5(C4)
        #print("conv5", C5.shape)
        out = [C2, C3, C4, C5]

        return out

    def createConvBlock(self, ResBlock, num_blocks, inp_ch, out_ch, stride):
        """ 
        Create a conv block consisting of multiple conv layers.
        Args: 
            ResBlock: residual block type, BasicBlock for ResNet-18, 34 or 
                      BottleNeck for ResNet-50, 101, 152
            num_blocks: number of residual blocks
            inp_ch: number of input channels
            out_ch: number of output channels
            stride: stride used in the first 3x3 convolution of the first resdiual block
                    of the layer and 1x1 convolution for skip connection in that block
        Returns: 
            convolutional block
        """

        blocks = []

        for i in range(num_blocks):
            #apply a stride for downsampling the feature map for the first block.
            if i == 0:
               cb = ResBlock(inp_ch, out_ch, stride, is_first_block=True)
            else:
               # Keep the feature map size same for the remaining blocks
               # by setting stride=1 and is_first_block=False.
               # By default, ResBlock.expansion = 4 for ResNet-50, 101, 152, 
               # ResBlock.expansion = 1 for ResNet-18, 34.
               cb = ResBlock(out_ch*ResBlock.expansion, out_ch)  
            blocks.append(cb)

        return nn.Sequential(*blocks)     
