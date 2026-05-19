
from torch import nn
from torch.nn import functional as F
import torch
import torch.utils.checkpoint as cp
from maskadapter.convnext import ConvNextBlock
from einops import rearrange,repeat

class MASKAdapterHead(nn.Module):
    def __init__(
        self,
        clip_model_name='_large',
        mask_in_chans=16,
        num_channels=768,
        num_output_maps=16,
        pretrained=True
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        
        if '_base' in clip_model_name:
            clip_dim = 640
        elif '_large' in clip_model_name:
            clip_dim = 768
        
        self.fuse = nn.Conv2d(clip_dim, num_channels, 1)
                
        self.cnext1 = ConvNextBlock(num_channels)
        
        self.cnext2 = ConvNextBlock(num_channels)
        
        self.cnext3 = ConvNextBlock(num_channels)
        
        self.norm = nn.LayerNorm(num_channels)
        self.final = nn.Conv2d(num_channels, num_output_maps, 1)
        
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, clip_dim, kernel_size=1),
        )
        
        if pretrained:
            ckpt = torch.load("src/third_party/maskadapter/maskadpater_ckpt.pth", map_location=torch.device('cpu'), weights_only=False)
            self.load_state_dict(ckpt, strict=True)
        
    def forward(self, clip_feature, masks):

        
        N = masks.size(1)
        masks = rearrange(masks, 'B N H W -> (B N) H W').unsqueeze(dim=1)
        
        clip_feature = repeat(clip_feature, "B C H W -> (B N) C H W", N=N)
        
        H,W = clip_feature.shape[-2:]
        masks = F.interpolate(masks, size=(H*4,W*4),
                                                mode='bilinear', align_corners=False)
        masks = self.mask_downscaling(masks)
        
        outputs = clip_feature + masks
        
        def _inner_forward(outputs):
            outputs = self.fuse(outputs)
        
            outputs = self.cnext1(outputs)
            
            outputs = self.cnext2(outputs)
            
            outputs = self.cnext3(outputs)
            
            outputs = outputs.permute(0, 2, 3, 1) 
            outputs = self.norm(outputs.contiguous())
            outputs = outputs.permute(0, 3, 1, 2) 
            
            outputs = self.final(outputs.contiguous()) 
            
            outputs = rearrange(outputs, '(B N) C H W -> B (N C) H W',N=N)
    
            return outputs

        outputs = _inner_forward(outputs)
        return outputs

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x