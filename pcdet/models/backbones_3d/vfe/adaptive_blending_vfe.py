import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated
        

class AdaptiveBlendingVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        self.original_num_point_features = num_point_features

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        
        num_point_features_for_sharp = 0
        if self.use_absolute_xyz:
            num_point_features_for_sharp += self.original_num_point_features
        else:
            num_point_features_for_sharp += self.original_num_point_features - 3
        
        num_point_features_for_sharp += 6
        
        if self.with_distance:
            num_point_features_for_sharp += 1
        
        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features_for_sharp] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.gate_pfn = PFNLayer(
            in_channels=self.original_num_point_features,
            out_channels=16,
            use_norm=self.use_norm,
            last_layer=True
        )
        self.gate_final_layer = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.smooth_projector = nn.Linear(
            self.original_num_point_features, 
            self.num_filters[-1]
        )

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        
        voxel_count = voxel_features.shape[1] 
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask_for_sharp = torch.unsqueeze(mask, -1).type_as(voxel_features)

        points_mean_for_smooth = voxel_features.sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        f_smooth = points_mean_for_smooth / normalizer
        f_smooth_projected = self.smooth_projector(f_smooth)
        
        gated_features = self.gate_pfn(voxel_features * mask_for_sharp)
        alpha = self.gate_final_layer(gated_features.squeeze(1))

        points_mean_xyz = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean_xyz
        
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features_for_sharp = [voxel_features, f_cluster, f_center]
        else:
            features_for_sharp = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features_for_sharp.append(points_dist)
            
        features_for_sharp = torch.cat(features_for_sharp, dim=-1)
        
        features_for_sharp *= mask_for_sharp
        
        for pfn in self.pfn_layers:
            features_for_sharp = pfn(features_for_sharp)
            
        f_sharp = features_for_sharp.squeeze(1)

        f_final = alpha * f_sharp + (1 - alpha) * f_smooth_projected
        
        batch_dict['voxel_features'] = f_final.contiguous()
        return batch_dict