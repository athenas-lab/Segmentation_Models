import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as T

""" U-Net architecture with batch normalization that can be enabled"""

class Block(nn.Module):

    def __init__(self, in_ch, out_ch, norm=""):

        super().__init__()
        self.in_ch = in_ch
        self.norm = norm

        #unpadded convolution
        self.conv1 = nn.Conv2d(in_ch, out_ch, 2)
        if norm == "bn": 
          self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()
      
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
        if norm == "bn": 
           self.bn2 = nn.BatchNorm2d(out_ch)
        

    def forward(self, x):
        
        #2 unpadded convolutions interleaved with ReLU
        out = self.conv1(x)
        if self.norm == "bn":
           out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        if self.norm == "bn":
           out = self.bn2(out)
        out = self.act(out)

        return out

class Encoder(nn.Module):
    """ 
    Takes an input image, generates more feature maps (more output channels) 
    with reduced spatial dimensions at each stage.
    """

    def __init__(self, channels = [3, 16, 32, 64], norm=""):
    
        super().__init__()

        #Starting with 3 channels for RGB input, number of output 
        #feature maps/channels is doubled in each encoder block.
        self.encBlocks = nn.ModuleList(
            [ Block(channels[i], channels[i+1], norm) for i in range (len(channels)-1)])
        #2x2 pooling layer with stride 2 for downsampling halves the spatial dimension (H, W).
        self.pool = nn.MaxPool2d(2) 


    def forward(self, x):
      
        block_out = []
        for b in self.encBlocks:
            x = b(x)
            #store the unpooled output for passing to the corresponding decoder layer
            block_out.append(x)
            #downsample
            x = self.pool(x)

        return block_out

class Decoder(nn.Module):
    """ 
    The decoder phase generates fewer feature maps (halves the number of channels) 
    with higher spatial dimension at each stage. Final output is a segmentation map
    that has the same dimensions as the input image.
    """

    def __init__(self, channels=(64, 32, 16), norm=""):

        super().__init__()
        self.channels = channels
        #upsampling layer consists of the 2x2 deconv operation
        self.upsample = nn.ModuleList(
            [nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=2, stride=2) 
              for i in range (len(channels)-1)])
        #each decoder block consists of 2 3x3 conv layers interleaved with ReLU      
        self.dec_blocks = nn.ModuleList(
            [Block(channels[i], channels[i+1], norm) for i in range (len(channels)-1)])
        
    def forward(self, x, enc_feats):
        
        # Each step consists of an upsampling operation which doubles the spatial dimension (H, W)
        # and halves the number of channels,
        # followed by concatenation of the upsampled feature with the cropped encoded feature
        # from the corresponding  downsampling block
        # and finally, decoding the concatenated features using a pair of 3x3 conv. 
        for i in range (len(self.channels)-1):
            #upsample the input to the decoder block using a 2x2 deconv operation
            #which increases the spatial dimension and halves the number of channels
            x = self.upsample[i](x)
            #crop the encoded features from the corresponding encoder block 
            #so that its size is the same as the upsampled input. The cropping
            #is necessary due to loss of border pixels during unpadded conv 
            #in each  conv layer in the encoder block.
            crop = self.crop(enc_feats[i], x)
            #skip-connection: channel-wise concatenation of the cropped encoded features 
            #with the upsampled input
            x = torch.concat([x, crop], dim=1)
            #decode the concatenated input through 2 3x3 conv layers
            x = self.dec_blocks[i](x)
            
        #return the final decoder output
        return x    

    def crop(self, feat, x):

        #crop the encoded features to match the input dimensions
        _, _, H, W = x.shape
        feat = T.CenterCrop([H, W])(feat)

        #return the cropped features
        return feat


class UNet(nn.Module):

    def __init__(self, 
           cfg,
           retain_dim = True #output dim = inp dim
        ):

        super(UNet, self).__init__()
            
        enc_ch = cfg.enc_ch if cfg.enc_ch is not None else [3, 16, 32, 64]
        dec_ch = cfg.dec_ch if cfg.dec_ch is not None else [64, 32, 16]
        num_class = cfg.num_classes  #categories to classify

        self.enc = Encoder(enc_ch, cfg.norm)
        self.dec = Decoder(dec_ch, cfg.norm)

        self.retain_dim = retain_dim
        if retain_dim:
            self.out_size = (cfg.img_height, cfg.img_width)
        # final 1x1 conv layer outputs the segmentation  map using regression.
        # input is the output of the final decoder layer
        # output channels == number of classes     
        self.head = nn.Conv2d(dec_ch[-1], num_class, kernel_size=1)    

    
    def forward(self, x):

        #get the encoded features from each encoder block
        enc_feats = self.enc(x)
        
        #reverse the order of the encoded features since the 
        #decoding starts with the final encoder output as the input.
        enc_feats = enc_feats[::-1]
        dec_feats = self.dec(enc_feats[0], enc_feats[1:])

        #final 1x1 conv layer outputs the segmentation map using regression.
        #thresholding is using for classification.
        seg_map = self.head(dec_feats)

        #resize the dimensions of the segmentation map to match the dim of the input image,
        #if needed.
        if self.retain_dim:
            seg_map = F.interpolate(seg_map, self.out_size)

        #return the segmentation map
        return seg_map
