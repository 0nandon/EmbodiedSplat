import antialiased_cnns
from torchvision import models
import numpy as np
import timm
import torch
from torch import nn
from torchvision.ops import FeaturePyramidNetwork
import torch.nn.functional as F

from modules.layers import BasicBlock
from sr_utils.generic_utils import upsample
from einops import *
from src.model.encoder.costvolume.ldm_unet.unet import UNetModel


def double_basic_block(num_ch_in, num_ch_out, num_repeats=2):
    layers = nn.Sequential(BasicBlock(num_ch_in, num_ch_out))
    for i in range(num_repeats - 1):
        layers.add_module(f"conv_{i}", BasicBlock(num_ch_out, num_ch_out))
    return layers


class DepthDecoderPP(nn.Module):
    def __init__(
                self, 
                num_ch_enc, 
                scales=range(4), 
                num_output_channels=1,  
                use_skips=True,
                near=0.5,
                far=15.0,
                num_samples=64,
                n_levels=-1,
                use_planes=True,
                log_plane=False,
                wo_msd=False,
                refine=False,
                num_context_views=2,
                low_res=False,
            ):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.num_samples = num_samples
        self.n_levels = n_levels

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([64, 64, 128, 256])
        self.use_planes = use_planes
        self.log_plane = log_plane
        self.refine = refine
        self.near = near
        self.far = far
        self.low_res = low_res

        # print('depth pred range:', near, far)

        # decoder
        self.convs = nn.ModuleDict()
        # i is encoder depth (top to bottom)
        # j is decoder depth (left to right)
        max_depth = 2 + 2 * (not wo_msd)
        self.max_depth = max_depth
        for j in range(1, max_depth+1):
            max_i = max_depth - j
            for i in range(max_i, -1, -1):

                num_ch_out = self.num_ch_dec[i]
                total_num_ch_in = 0

                num_ch_in = self.num_ch_enc[i + 1] if j == 1 else self.num_ch_dec[i + 1]
                self.convs[f"diag_conv_{i + 1}{j - 1}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                total_num_ch_in += num_ch_out

                num_ch_in = self.num_ch_enc[i] if j == 1 else self.num_ch_dec[i]
                self.convs[f"right_conv_{i}{j - 1}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                total_num_ch_in += num_ch_out

                if i + j != max_depth:
                    num_ch_in = self.num_ch_dec[i + 1]
                    self.convs[f"up_conv_{i + 1}{j}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                    total_num_ch_in += num_ch_out

                self.convs[f"in_conv_{i}{j}"] = double_basic_block(
                                                                total_num_ch_in, 
                                                                num_ch_out,
                                                            )

                    # print('+++++++++++++++++num_output_channels:', num_output_channels)
                    # if i <= max(n_levels,0):
                    #     self.convs[f"output_{i}"] = nn.Sequential(
                    #                 BasicBlock(num_ch_out, num_ch_out*2),
                    #                 nn.Conv2d(num_ch_out*2, self.num_output_channels, 1),
                    #                 )
                    #     self.convs[f"depth_{i}"] = nn.Sequential(
                    #                 BasicBlock(num_ch_out, num_ch_out*2),
                    #                 nn.Conv2d(num_ch_out*2, self.num_samples, 1),
                    #                 )
                    # if i==0:
                self.convs[f"output_{i}"] = nn.Sequential(
                        BasicBlock(num_ch_out, num_ch_out) if i != 0 else nn.Identity(),
                        nn.Conv2d(num_ch_out, self.num_output_channels if use_planes else 1*(i>0)+(i==0)*self.num_output_channels, 1),
                        )
        # self.last_depth = nn.Sequential(
        #                 BasicBlock(num_ch_out, 128),
        #                 nn.Conv2d(128, num_samples, 1),
        #                 )
        if use_planes:
            if log_plane:
                # print('near:', near)
                # print('far:', far)
                depth_candi_curr = (
                    torch.log(torch.tensor(near))
                    + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0)
                    * torch.log(torch.tensor(far / near))
                )
            else:
                min_depth = 1.0 / far
                max_depth = 1.0 / near
                depth_candi_curr = (
                    max_depth
                    + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0)
                    * (min_depth - max_depth)
                )
            self.depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d () ()")  # [vxb, d, 1, 1]

            # print('depth_candi_curr:', torch.exp(self.depth_candi_curr) if log_plane else 1 / self.depth_candi_curr)

            self.conv_depth = nn.ModuleDict()
        
            for i in range(4):
                self.conv_depth[f'{i}'] = nn.Sequential(
                                    BasicBlock(self.num_output_channels, num_samples),
                                    nn.Conv2d(num_samples, num_samples, 1),
                                    )

        
        
        if self.refine:
            channels = 16
            self.refine_unet = nn.Sequential(
                nn.Conv2d(1+65+3, channels, 3, 1, 1),
                nn.GroupNorm(4, channels),
                nn.GELU(),
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=1, 
                    attention_resolutions=[],
                    channel_mult=[1,1,1,1,1],
                    num_head_channels=channels,
                    dims=2,
                    postnorm=True,
                    num_frames=num_context_views,
                    use_cross_view_self_attn=False,
                ),
                nn.Conv2d(channels, channels * 2, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(channels * 2, 1+self.num_output_channels+1, 3, 1, 1),
            )

        else:
            self.conv_last = nn.Sequential(
                        BasicBlock(self.num_output_channels, 128),
                        nn.Conv2d(128, self.num_output_channels+(not self.use_planes), 1),
                        )

        # self.last_out = nn.Sequential(
        #                     BasicBlock(num_ch_out, 128),
        #                     nn.Conv2d(128, self.num_output_channels, 1),
        #                     )

    def forward(self, input_features, imgs=None):
        prev_outputs = input_features
        outputs = []
        depth_outputs = {}
        ms_outputs = {}
        for j in range(1, self.max_depth+1):
            max_i = self.max_depth - j
            for i in range(max_i, -1, -1):

                inputs = [self.convs[f"right_conv_{i}{j - 1}"](prev_outputs[i])]
                inputs += [upsample(self.convs[f"diag_conv_{i + 1}{j - 1}"](prev_outputs[i + 1]))]

                if i + j != self.max_depth:
                    inputs += [upsample(self.convs[f"up_conv_{i + 1}{j}"](outputs[-1]))]
                
                output = self.convs[f"in_conv_{i}{j}"](torch.cat(inputs, dim=1))

                outputs += [output]

                if i==0:
                    outputs_s0 = self.convs[f"output_{i}"](output)
                # ms_outputs[f'{i}'] = self.convs[f"output_{i}"](output)
                
                if not self.use_planes:
                    depth_outputs[f"output_pred_s{i}_b1hw"] = self.convs[f"output_{i}"](output)

                # if i + j == 4 and i <= max(self.n_levels, 0):
                #     depth_outputs[f"output_pred_s{i}_b1hw"] = self.convs[f"output_{i}"](output)
                #     depth_planes = F.softmax(self.convs[f"depth_{i}"](output), dim=1)
                #     coarse_disps = (self.depth_candi_curr.to(depth_planes.device) * depth_planes).sum(dim=1, keepdim=True)
                #     depth_outputs[f'depth_pred_s{i}_b1hw'] = 1.0 / coarse_disps
                #     if j == 4:
                #         depth_outputs[f"output_pred_s-1_b1hw"] = self.last_out(upsample(output))
                #         depth_planes = F.softmax(self.last_depth(upsample(output)), dim=1)
                #         fine_disps = (self.depth_candi_curr.to(depth_planes.device) * depth_planes).sum(dim=1, keepdim=True)
                #         depth_outputs[f'depth_pred_s-1_b1hw'] = 1.0 / fine_disps


            prev_outputs = outputs[::-1]

        # print('output_pred_s0_b1hw.shape:', depth_outputs['output_pred_s0_b1hw'].shape)
        # print('self.output_channels:', self.num_output_channels)

        # for i in range(self.max_depth-1,-1,-1):
        i = 0
        if self.use_planes:
            # depth_planes = F.softmax(self.conv_depth[f'{i}'](depth_outputs[f"output_pred_s{i}_b1hw"]), dim=1)
            depth_planes = F.softmax(self.conv_depth[f'{i}'](outputs_s0), dim=1)
            coarse_disps = (self.depth_candi_curr.to(depth_planes.device) * depth_planes).sum(dim=1, keepdim=True)
            # depth_outputs[f'depth_pred_s{i}_b1hw'] = torch.exp(coarse_disps) if self.log_plane else 1.0 / coarse_disps
            # depth_outputs[f'log_depth_pred_s{i}_b1hw'] = coarse_disps if self.log_plane else torch.log(1.0 / coarse_disps+1e-8)

            coarse_map = torch.exp(coarse_disps) if self.log_plane else 1.0 / coarse_disps
            if not self.low_res:
                fine_disps = F.interpolate(
                    coarse_disps,
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=True,
                )
                fine_map = torch.exp(fine_disps) if self.log_plane else 1.0 / fine_disps

                depth_outputs['depth_pred_s-1_b1hw'] = fine_map
            # fine_map = F.interpolate(
            #         fine_map,
            #         scale_factor=2,
            #         mode="bilinear",
            #         align_corners=True,
            #     )
            if self.refine:
                # print(f'fine_disps.shape: {fine_disps.shape}, depth_outputs.shape: {depth_outputs[f"output_pred_s0_b1hw"].shape}')
                fine_feat = F.interpolate(
                                outputs_s0,
                                scale_factor=2,
                                mode="bilinear",
                                align_corners=True,
                            )
                # print('fine_disps.shape:', fine_disps.shape)
                # print('fine_feat.shape:', fine_feat.shape)
                # print('imgs.shape:', imgs.shape)
                fine_outputs = self.refine_unet(torch.cat([fine_disps, fine_feat, imgs], dim=1))
                disps_delta = fine_outputs[:, :1]
                fine_disps = (fine_disps + disps_delta).clamp(
                            np.log(self.near) if self.log_plane else 1.0 / self.far,
                            np.log(self.far) if self.log_plane else 1.0 / self.near,
                        )
                fine_map = torch.exp(fine_disps) if self.log_plane else 1.0 / fine_disps
                
            depth_outputs['depth_pred_s0_b1hw'] = coarse_map
        else:
            for i in range(self.max_depth-1,-1,-1):
                log_depth = depth_outputs[f"output_pred_s{i}_b1hw"][:, :1]
                # depth_outputs[f'depth_pred_s{i}_b1hw'] = torch.exp(log_depth)
                # depth_outputs[f'log_depth_pred_s{i}_b1hw'] = log_depth
                depth_outputs[f'depth_pred_s{i}_b1hw'] = torch.exp(log_depth)
            fine_log_depth = F.interpolate(
                log_depth,
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
            fine_map = torch.exp(fine_log_depth)
            depth_outputs['depth_pred_s-1_b1hw'] = fine_map
        # normed_depth = fine_map / fine_map.max()
        # weights = torch.exp(-1.0 * (normed_depth**2) / 0.72).detach()
        # depth_outputs['depth_weights'] = weights

        # print('depth_planes:', depth_planes.min(), depth_planes.max())
        # print('coarse_disps:', coarse_disps.min(), coarse_disps.max())
        # print('fine_disps:', fine_disps.min(), fine_disps.max())
        # print('depth_pred_s0_b1hw:', depth_outputs['depth_pred_s0_b1hw'].min(), depth_outputs['depth_pred_s0_b1hw'].max())
        # print('depth_pred_s-1_b1hw:', depth_outputs['depth_pred_s-1_b1hw'].min(), depth_outputs['depth_pred_s-1_b1hw'].max())

        # depth_outputs[f"output_pred_s-1_b1hw"] = self.conv_last(upsample(depth_outputs[f"output_pred_s0_b1hw"]))
        

        if self.use_planes:
            if self.refine:
                depth_outputs[f"output_pred_s-1_b1hw"] = fine_outputs[:, 1:1+self.num_output_channels]
                depth_outputs['depth_weights'] = nn.Sigmoid()(fine_outputs[:, -1:])
            else:
                if not self.low_res:
                    depth_outputs[f"output_pred_s-1_b1hw"] = self.conv_last(upsample(outputs_s0))
                    depth_outputs['depth_weights'] = F.interpolate(depth_planes,
                                                                    scale_factor=2,
                                                                    mode="bilinear",
                                                                    align_corners=True,
                                                                ).max(dim=1, keepdim=True)[0]
                else:
                    depth_outputs[f"output_pred_s0_b1hw"] = self.conv_last(outputs_s0)
                    depth_outputs['depth_weights'] = nn.Sigmoid()(depth_planes.max(dim=1, keepdim=True)[0])


        else:
            output_last = self.conv_last(upsample(outputs_s0))
            depth_outputs[f"output_pred_s-1_b1hw"] = output_last[:, :-1]
            depth_outputs['depth_weights'] = nn.Sigmoid()(output_last[:, -1:])
            # print('weight.shape:', depth_outputs['depth_weights'].shape)

        # print('1 / self.depth_candi_curr:', 1 / self.depth_candi_curr)
        
        
        return depth_outputs


class CVEncoder(nn.Module):
    def __init__(self, num_ch_cv, num_ch_enc, num_ch_outs):
        super().__init__()

        self.convs = nn.ModuleDict()
        self.num_ch_enc = []

        self.num_blocks = len(num_ch_outs)

        for i in range(self.num_blocks):
            num_ch_in = num_ch_cv if i == 0 else num_ch_outs[i - 1]
            num_ch_out = num_ch_outs[i]
            self.convs[f"ds_conv_{i}"] = BasicBlock(num_ch_in, num_ch_out, 
                                                    stride=1 if i == 0 else 2)

            self.convs[f"conv_{i}"] = nn.Sequential(
                BasicBlock(num_ch_enc[i] + num_ch_out, num_ch_out, stride=1),
                BasicBlock(num_ch_out, num_ch_out, stride=1),
            )
            self.num_ch_enc.append(num_ch_out)

    def forward(self, x, img_feats):
        outputs = []
        for i in range(self.num_blocks):
            x = self.convs[f"ds_conv_{i}"](x)
            x = torch.cat([x, img_feats[i]], dim=1)
            x = self.convs[f"conv_{i}"](x)
            outputs.append(x)
        return outputs

class MLP(nn.Module):
    def __init__(self, channel_list, disable_final_activation = False):
        super(MLP, self).__init__()

        layer_list = []
        for layer_index in list(range(len(channel_list)))[:-1]:
            layer_list.append(
                            nn.Linear(channel_list[layer_index], 
                                channel_list[layer_index+1])
                            )
            layer_list.append(nn.LeakyReLU(inplace=True))

        if disable_final_activation:
            layer_list = layer_list[:-1]

        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        try:
            return self.net(x)
        except:
            print('x.shape:', x.shape)
            print('x:', x)
            print('self.net:', self.net)
            raise ValueError

class ResnetMatchingEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(
                self, 
                num_layers, 
                num_ch_out, 
                pretrained=True,
                antialiased=True,
            ):
        super().__init__()

        self.num_ch_enc = np.array([64, 64])

        model_source = antialiased_cnns if antialiased else models
        resnets = {18: model_source.resnet18,
                   34: model_source.resnet34,
                   50: model_source.resnet50,
                   101: model_source.resnet101,
                   152: model_source.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers"
                                                            .format(num_layers))

        encoder = resnets[num_layers](pretrained)

        resnet_backbone = [
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,
            encoder.layer1,
        ]


        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.num_ch_out = num_ch_out

        self.net = nn.Sequential(
            *resnet_backbone,
            nn.Conv2d(self.num_ch_enc[-1], 128, (1, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                    128, 
                    self.num_ch_out, 
                    (3, 3), 
                    padding=1, 
                    padding_mode="replicate"
                ),
            nn.InstanceNorm2d(self.num_ch_out)
        )

    def forward(self, input_image):
        return self.net(input_image)

class UNetMatchingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
                                        "mnasnet_100", 
                                        pretrained=True, 
                                        features_only=True,
                                    )

        self.decoder = FeaturePyramidNetwork(
                                        self.encoder.feature_info.channels(), 
                                        out_channels=32,
                                    )
        self.outconv = nn.Sequential(
                                    nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(32, 16, 1),
                                    nn.InstanceNorm2d(16),
                                )

    def forward(self, x):
        encoder_feats = {f"feat_{i}": f for i, f in enumerate(self.encoder(x))}
        return self.outconv(self.decoder(encoder_feats)["feat_1"])
