import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resnet

""" Feature Pyramid Network for Image Segmentation """

class FPNBlock(nn.Module):

    def __init__(self, inp_ch, out_ch=256, highest_block=False):
        """
        Implements the top-down network of FPN. 
        The feature map from the lateral layer in the bottom-up network is fused 
        with the feature-map from the higher layer in the top-down network. This allows the 
        semantic information from the higher layers to be propagated to the lower layer feature maps.  
        Args:
            inp_ch: number of channels in the input feature map
            out_ch: number of channels in the output feature map
            highest_block: indicates if the block is the topmost level of the pyramid
        Returns:
            output of the top-down network
        """

        super().__init__()

        #1x1 conv results in same number of channels (=256 by default) for the multi-scale feature maps 
        self.conv1 = nn.Conv2d(inp_ch, out_ch, kernel_size=1, stride=1, padding=0)
        ##3x3 conv for FPN output
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.highest_block = highest_block    

    def forward(self, cur, higher):
        """
        cur: input feature map of current layer
        higher: input feature map from previous higher layer 
        """

        #print("FPNBlock cur.shape", cur.shape)
        #first pass the input through 1x1 conv 
        x = self.conv1(cur) 
        #print("FPNBlock", x.shape)

        #upscale the feature from the previous higher layer and element-wise add it 
        #to the feature map from current layer. The result is that all the feature maps
        #will have similar semantic information.
        if not self.highest_block: 
           #print("FPNBlock", x.shape, higher.shape) 
           x += F.interpolate(higher, scale_factor=2, mode="bilinear", align_corners=True)
        #pass the feature map through the 3x3 conv
        out = self.conv2(x)

        #return the output before and after passing through the 3x3 conv
        return x, out


class FPN(nn.Module):

    def __init__(self, expansion=4, inp_ch=[64, 128, 256, 512],  out_ch=256):
        """
        Args:
           expansion: expansion of ResBlock (1 for BasicBlock, 4 for BottleNeck)
           inp_ch: list of input channels for the conv blocks
           out_ch: out channel for each block

        Implements the fully conv FeaturePyramid Network architecture that extracts multi-scale feature maps from an
        input image. The FPN architecture consists of a bottom-up and top-down network. 
        
        The bottom-up fully conv network consists of a conv network (e.g. resnet) stacked with fully conv blocks  each of which 
        outputs increasing number of feature maps (number of channels) from bottom to top with decreasing spatial resolution.
        The top-level feature maps have lower spatial resolution and capture more global, high-level semantics from 
        an input image, which are useful for classification tasks. In contrast, the lower-level feature maps have higher spatial 
        resolution and capture more local, pixel-lvel features. 

        The top-down network fuses the feature maps output by each block with those output by the block below it. 
        All the fused maps have the same number of channels. This fusion ensures that the semantic information 
        from the top-level feature maps is embedded into all the lower-level feature maps along with their local
        spatial information, so that  these fused features can be used for downstream applications, 
        like object detection and segmentation. For e.g. The fusion of the current and higher-level features in the 
        top-down path helps in image segmentation tasks, which require features that provide pixel-level location and 
        image-level object classification.  
        """

        super().__init__()

        self.P2 = FPNBlock(inp_ch[0]*expansion, out_ch) 
        self.P3 = FPNBlock(inp_ch[1]*expansion, out_ch) 
        self.P4 = FPNBlock(inp_ch[2]*expansion, out_ch) 
        self.P5 = FPNBlock(inp_ch[3]*expansion, out_ch, highest_block=True) 

        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

    def forward(self, C2, C3, C4, C5):
        """
        input: C2, C3, C4, C5: feature maps input through skip connections from the lateral conv blocks 
        in the bottom-up network.
        output: feature maps output by the FPN blocks P2-P6. 
        """
        
        #print("FPN", C5.shape)
        #Top-down path. Each block takes as input a combination of the feature maps output by the higher-level block 
        #and the lateral block from the bottom-up network and uses the fused input to output the feature maps.   
        x, outP5 = self.P5(C5, None)   
        #print("P5", x.shape, outP5.shape)
        x, outP4 = self.P4(C4, x)
        #print("P4", x.shape, outP4.shape)
        x, outP3 = self.P3(C3, x)
        #print("P3", x.shape, outP3.shape)
        x, outP2 = self.P2(C2, x)
        #print("P2", x.shape, outP2.shape)
        outP6 = self.P6(outP5)
        #print("P6", outP6.shape)
        return [outP2, outP3, outP4, outP5, outP6]


class ResnetFPN(nn.Module):
    """ FPN with Resnet backbone """

    def __init__(self, arch=50):
        """
        FPN architecure based on resnet for the bottom-up backbone network.

        arch: defines the number of blocks in the Resnet arch.
        """

        super().__init__()
        assert (arch in [18, 34, 50, 101, 152]), "Unsupported resnet architecture"
        ResBlock = resnet.BasicBlock if arch in [18, 34] else resnet.BottleNeck

        if arch == 18:
           nblocks_list = [2, 2, 2, 2]
        elif arch in [34, 50]:
           nblocks_list = [3, 4, 6, 3]
        elif arch == 101:
           nblocks_list = [3, 4, 23, 3]
        else:
           nblocks_list = [3, 8, 36, 3]
         
        #Resnet backbone for the bottom-up network
        self.bottom_up = resnet.ResNet(ResBlock, nblocks_list=nblocks_list)
        #feature pyramid top-down net
        self.top_down = FPN(expansion=ResBlock.expansion) 

    def forward(self, x):
        """
        x: input img tensor
        Pass the input tensor through the bottom-up (resnet) backbone.
        Then pass the feature maps through the top-down network in the FPN module.
        """
        #print("ResnetFPN", x.shape)
        #extract features from each level of the bottom-up backbone net
        C2, C3, C4, C5 = self.bottom_up(x)
        #pass each feature pap through the top-down network to output the mult-scale features
        out_fmaps = self.top_down(C2, C3, C4, C5)

        return out_fmaps



class SegmentFPN(nn.Module):
    """ Semantic segmentation head for processing the FPN features """

    def __init__(self, cfg):
        """

        Args:
           backbone: FPN architecure for the bottom-up backbone network (based on resnet).
           arch: defines the number of blocks in the Resnet arch.
        """

        super().__init__()
 
        num_classes = cfg.num_classes
        backbone= cfg.backbone
        arch= cfg.arch

        assert(backbone == "resnet"), "unsupported architecture"

        self.fpn = ResnetFPN(arch)
       
        #segmentation head for processing the FPN features
        fpn_ch = 256 
        self.seg1 = nn.Conv2d(fpn_ch, fpn_ch, kernel_size=3, stride=1, padding=1)
        self.seg2 = nn.Conv2d(fpn_ch, 128, kernel_size=3, stride=1, padding=1)
        self.seg3 = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0)
        #group norm:, num_groups, num_channels
        self.gn1 = nn.GroupNorm(256, 256)
        self.gn2 = nn.GroupNorm(128, 128)

   

    def forward(self, x):

        #print("SegmenFPN", x.shape)
        fpn_out = self.fpn(x)    
        P2, P3, P4, P5, P6 = fpn_out

        _, _, h, w = P2.shape
        #256-->256
        s5 = upSample(F.relu(self.gn1(self.seg1(P5))), h, w)
        s5 = upSample(F.relu(self.gn1(self.seg1(s5))), h, w)
        #256-->128
        s5 = upSample(F.relu(self.gn2(self.seg2(s5))), h, w)

        #256-->256
        s4 = upSample(F.relu(self.gn1(self.seg1(P4))), h, w)
        #256-->128
        s4 = upSample(F.relu(self.gn2(self.seg2(s4))), h, w)

        #256-->128
        s3 = upSample(F.relu(self.gn2(self.seg2(P3))), h, w)

        #256->128, no upscaling
        s2 = F.relu(self.gn2(self.seg2(P2)))

        #add the feature maps and upscale the segmentation map to the image size
        map = self.seg3(s5 + s4 + s3 +s2)
        logits = upSample(map, h*4, w*4)

        return logits



def upSample(x,  h, w):
    """ 
      Upsample x to the specified dim (h, w).
    """    

    out = F.interpolate(x, size=(h, w), mode= "bilinear", align_corners=True)
        
    return out


def test():
    # Resnet arch types
    arch = [18, 34, 50, 101, 152]
    resnet_arch = 50

    x = torch.randn((1, 3, 512, 800), dtype=torch.float32)
    
    #arch check for resnet-based FPN
    test_fpn = False
    if test_fpn:
        net = ResnetFPN(arch)
        fpn_out = net(x)
        for i, f in enumerate(fpn_out):
            print(f"{i}:{f.shape}")

    net = SegmentFPN(32, arch=resnet_arch)
    logits = net(x)
    print(logits, logits.shape)

#test()
