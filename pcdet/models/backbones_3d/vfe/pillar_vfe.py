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


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)
        self.original_num_point_features = num_point_features
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.gate_input_dim = self.original_num_point_features
        
        self.gate_network = nn.Sequential(
            nn.Linear(self.gate_input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 出力を0〜1の混合比率αにする
        )
        # 2.f_smoothをf_sharpと同じ次元数に変換するプロジェクター層
        self.smooth_input_dim = self.original_num_point_features
        self.smooth_output_dim = self.num_filters[-1] # f_sharpの出力次元数
        self.smooth_projector = nn.Linear(self.smooth_input_dim, self.smooth_output_dim)

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
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            rot_num = trans_param.shape[1]
        else:
            rot_num = 1

        for i in range(rot_num):
            if i==0:
                frame_id = ''
            else:
                frame_id = str(i)
        
            voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']

            # --- ここからが「適応的ブレンディング」のメイン処理 ---

            # 1.【F_smooth】滑らかな特徴量を計算 (MeanVFEのロジック)
            #    入力された点群特徴量(8次元など)を単純に平均する
            points_mean_for_smooth = voxel_features.sum(dim=1, keepdim=False)
            normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
            f_smooth = points_mean_for_smooth / normalizer

            # 2.【F_sharp】シャープな特徴量を計算 (PillarVFEのロジック)
            #    既存のPillarVFEの処理をそのまま実行
            points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
            f_cluster = voxel_features[:, :, :3] - points_mean
            
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

            voxel_count = features_for_sharp.shape[1]
            mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
            mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
            
            features_for_sharp *= mask
            
            for pfn in self.pfn_layers:
                features_for_sharp = pfn(features_for_sharp)
                
            f_sharp = features_for_sharp.squeeze()
            
            # 3.【αの計算】混合比率αをゲート機構で計算
            #    ゲート機構の入力として、F_smoothを使うのが合理的
            #    (フラグや信頼度スコアの平均情報が含まれているため)
            alpha = self.gate_network(f_smooth)
            
            f_final = alpha * f_sharp + (1 - alpha) * self.smooth_projector(f_smooth)
            
            batch_dict['voxel_features' + frame_id] = f_final
        
        return batch_dict
