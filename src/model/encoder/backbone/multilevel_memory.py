# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/detectors/single_stage_sparse.py # noqa
try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    pass
from MinkowskiEngine.MinkowskiPooling import MinkowskiAvgPooling

import torch
import torch.nn as nn


class MultilevelMemory(nn.Module):
    def __init__(
            self, 
            in_channels=[64, 128, 256, 512], 
            scale=2.5, 
            queue=-1, 
            vmp_layer=(0,1,2,3),
            norm="BN"
        ):
        super().__init__()
        self.scale = scale
        self.queue = queue
        self.vmp_layer = list(vmp_layer)
        self.conv_d1 = nn.ModuleList()
        self.conv_d3 = nn.ModuleList()
        self.conv_convert = nn.ModuleList()
        self.norm = norm

        for i, C in enumerate(in_channels):
            if i in self.vmp_layer:
                self.conv_d1.append(nn.Sequential(
                    ME.MinkowskiConvolution(
                        in_channels=C,
                        out_channels=C,
                        kernel_size=3,
                        stride=1,
                        dilation=1,
                        bias=False if self.norm=="BN" else True,
                        dimension=3),
                    self.get_norm(C, self.norm),
                    ME.MinkowskiReLU()))
                self.conv_d3.append(nn.Sequential(
                    ME.MinkowskiConvolution(
                        in_channels=C,
                        out_channels=C,
                        kernel_size=3,
                        stride=1,
                        dilation=3,
                        bias=False if self.norm=="BN" else True,
                        dimension=3),
                    self.get_norm(C, self.norm),
                    ME.MinkowskiReLU()))
                self.conv_convert.append(nn.Sequential(
                    ME.MinkowskiConvolutionTranspose(
                        in_channels=3*C,
                        out_channels=C,
                        kernel_size=1,
                        stride=1,
                        dilation=1,
                        bias=False if self.norm=="BN" else True,
                        dimension=3),
                    self.get_norm(C, self.norm)))
            else:
                self.conv_d1.append(nn.Identity())
                self.conv_d3.append(nn.Identity())
                self.conv_convert.append(nn.Identity())

        self.relu = ME.MinkowskiReLU()

        self.proj = nn.ModuleList()
        for i, C in enumerate(in_channels):
            if i in self.vmp_layer:
                self.proj.append(nn.Sequential(
                    ME.MinkowskiConvolution(
                        in_channels=64 if i == 0 else in_channels[i-1],
                        out_channels=C,
                        kernel_size=2,
                        stride=2,
                        bias=False if self.norm=="BN" else True,
                        dimension=3),
                    self.get_norm(C, self.norm),
                    ME.MinkowskiReLU())
                )

        self.init_weights()

    def init_weights(self):
        for n, m in self.named_modules():
            if 'proj' in n and isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, ME.MinkowskiConvolution) or isinstance(m, ME.MinkowskiConvolutionTranspose):
                nn.init.constant_(m.kernel, 0)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def get_norm(self, C, norm):
        if norm == "BN":
            return ME.MinkowskiBatchNorm(C)
        elif norm == "IN":
            return ME.MinkowskiInstanceNorm(C)
    
    def global_avg_pool_and_cat(self, feat1, feat2, feat3):
        coords1 = feat1.decomposed_coordinates
        feats1 = feat1.decomposed_features
        coords2 = feat2.decomposed_coordinates
        feats2 = feat2.decomposed_features
        coords3 = feat3.decomposed_coordinates
        feats3 = feat3.decomposed_features

        for i in range(len(coords3)):
            # shape 1 N
            global_avg_feats3 = torch.mean(feats3[i], dim=0).unsqueeze(0).repeat(coords3[i].shape[0],1)
            feats1[i] = torch.cat([feats1[i], feats2[i]], dim=1)
            feats1[i] = torch.cat([feats1[i], global_avg_feats3], dim=1)

        coords_sp, feats_sp = ME.utils.sparse_collate(coords1, feats1)
        feat_new = ME.SparseTensor(
            coordinates=coords_sp,
            features=feats_sp,
            tensor_stride=feat1.tensor_stride,
            coordinate_manager=feat1.coordinate_manager
        )
        return feat_new
    
    def accumulate(self, accumulated_feat, current_feat, index):
        """Accumulate features for a single stage.

        Args:
            accumulated_feat (ME.SparseTensor)
            current_feat (ME.SparseTensor)

        Returns:
            ME.SparseTensor: refined accumulated features
            ME.SparseTensor: current features after accumulation
        """
        
        if index in self.vmp_layer:
            # VMP
            tensor_stride = current_feat.tensor_stride
            accumulated_feat = ME.TensorField(
                features=torch.cat([current_feat.features, accumulated_feat.features], dim=0),
                coordinates=torch.cat([current_feat.coordinates, accumulated_feat.coordinates], dim=0),
                quantization_mode=ME.SparseTensorQuantizationMode.MAX_POOL
            ).sparse()
            accumulated_feat = ME.SparseTensor(
                coordinates=accumulated_feat.coordinates,
                features=accumulated_feat.features,
                tensor_stride=tensor_stride,
                coordinate_manager=accumulated_feat.coordinate_manager
            )

            # Select neighbor region for current frame
            accumulated_coords = accumulated_feat.decomposed_coordinates
            current_coords = current_feat.decomposed_coordinates
            accumulated_coords_select_list=[]
            zero_batch_feature_list=[]
            for i in range(len(current_coords)):
                accumulated_coords_batch = accumulated_coords[i]
                current_coords_batch = current_coords[i]
                current_coords_batch_max, _ = torch.max(current_coords_batch,dim=0)
                current_coords_batch_min, _ = torch.min(current_coords_batch,dim=0)
                current_box_size = current_coords_batch_max - current_coords_batch_min
                current_box_add = ((self.scale-1)/2) * current_box_size
                margin_positive = accumulated_coords_batch-current_coords_batch_max-current_box_add
                margin_negative = accumulated_coords_batch-current_coords_batch_min+current_box_add
                in_criterion = torch.mul(margin_positive,margin_negative)
                zero = torch.zeros_like(in_criterion)
                one = torch.ones_like(in_criterion)
                in_criterion = torch.where(in_criterion<=0,one,zero)
                mask = in_criterion[:,0]*in_criterion[:,1]*in_criterion[:,2]
                mask = mask.type(torch.bool)
                mask = mask.reshape(mask.shape[0],1)
                accumulated_coords_batch_select = torch.masked_select(accumulated_coords_batch,mask)
                accumulated_coords_batch_select = accumulated_coords_batch_select.reshape(-1,3)
                zero_batch_feature = torch.zeros_like(accumulated_coords_batch_select)
                accumulated_coords_select_list.append(accumulated_coords_batch_select)
                zero_batch_feature_list.append(zero_batch_feature)
            accumulated_coords_select_coords, _ = ME.utils.sparse_collate(accumulated_coords_select_list, zero_batch_feature_list)
            current_feat_new = ME.SparseTensor(
                coordinates=accumulated_coords_select_coords,
                features=accumulated_feat.features_at_coordinates(accumulated_coords_select_coords.float()),
                tensor_stride=tensor_stride,
                coordinate_manager=current_feat.coordinate_manager # new shorcut
            )
            
            branch1 = self.conv_d1[index](current_feat_new)
            branch3 = self.conv_d3[index](current_feat_new)
            branch  = self.global_avg_pool_and_cat(branch1, branch3, current_feat_new)
            branch = self.conv_convert[index](branch)
            current_feat_new = branch + current_feat # new shorcut
            current_feat_new = self.relu(current_feat_new)
            
            current_feat = ME.SparseTensor(
                coordinates=current_feat.coordinates,
                features=current_feat_new.features_at_coordinates(current_feat.coordinates.float()),
                tensor_stride=tensor_stride,
                coordinate_manager=current_feat.coordinate_manager
            )

        return current_feat
    
    def forward(self, x, accumulated_feats=None):
        if accumulated_feats is None:
            for i in range(len(x)):
                if i in self.vmp_layer:
                    branch1 = self.conv_d1[i](x[i])
                    branch3 = self.conv_d3[i](x[i])
                    branch  = self.global_avg_pool_and_cat(branch1, branch3, x[i])
                    branch = self.conv_convert[i](branch)
                    x[i] = branch + x[i]
                    x[i] = self.relu(x[i])
            return x
        else:
            tuple_feats = []
            for i in range(len(x)):
                accumulated_feats = self.proj[i](accumulated_feats)
                tuple_feats.append(
                    self.accumulate(
                        accumulated_feats, 
                        x[i], 
                        i
                    )
                )
            return tuple_feats
