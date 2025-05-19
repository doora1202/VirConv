import torch
import torch.nn as nn
from ..ops.iou3d_nms import iou3d_nms_utils
from ..utils import box_utils             

class GIoU3DLoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-7):
        super(GIoU3DLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps 

    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes (torch.Tensor): 予測されたバウンディングボックス (N, 7)
                                      [x, y, z, dx, dy, dz, heading]
            target_boxes (torch.Tensor): 真のバウンディングボックス (N, 7)
                                        [x, y, z, dx, dy, dz, heading]
        Returns:
            torch.Tensor: 計算された3D GIoU損失
        """
        if pred_boxes.shape[0] == 0 or target_boxes.shape[0] == 0:
            return torch.tensor(0.0, device=pred_boxes.device, dtype=pred_boxes.dtype)


        vol_pred = (pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5])
        vol_target = (target_boxes[:, 3] * target_boxes[:, 4] * target_boxes[:, 5])
        iou_matrix = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes, target_boxes)
        if iou_matrix.numel() > 0:
            iou3d_calculated_by_func = torch.diag(iou_matrix)
        else:
            iou3d_calculated_by_func = torch.tensor([], device=pred_boxes.device, dtype=pred_boxes.dtype)

        bev_overlap_matrix = pred_boxes.new_zeros(pred_boxes.shape[0], pred_boxes.shape[0])
        iou3d_nms_utils.iou3d_nms_cuda.boxes_overlap_bev_gpu(pred_boxes.contiguous(), target_boxes.contiguous(), bev_overlap_matrix)
        overlaps_bev_diag = torch.diag(bev_overlap_matrix) # (N,)

        pred_height_max = (pred_boxes[:, 2] + pred_boxes[:, 5] / 2)
        pred_height_min = (pred_boxes[:, 2] - pred_boxes[:, 5] / 2)
        target_height_max = (target_boxes[:, 2] + target_boxes[:, 5] / 2)
        target_height_min = (target_boxes[:, 2] - target_boxes[:, 5] / 2)
        max_of_min_h_diag = torch.max(pred_height_min, target_height_min)
        min_of_max_h_diag = torch.min(pred_height_max, target_height_max)
        overlaps_h_diag = torch.clamp(min_of_max_h_diag - min_of_max_h_diag, min=0) 

        overlaps_h_diag = torch.clamp(torch.min(pred_height_max, target_height_max) - torch.max(pred_height_min, target_height_min), min=0)


        intersection_volume = overlaps_bev_diag * overlaps_h_diag

        union_volume = vol_pred + vol_target - intersection_volume + self.eps

        iou3d = intersection_volume / torch.clamp(union_volume, min=self.eps)

        corners_pred = box_utils.boxes_to_corners_3d(pred_boxes)
        corners_target = box_utils.boxes_to_corners_3d(target_boxes)

        all_corners = torch.cat((corners_pred, corners_target), dim=1)

        min_coords = torch.min(all_corners, dim=1)[0]
        max_coords = torch.max(all_corners, dim=1)[0]

        enclosing_volume = (max_coords[:, 0] - min_coords[:, 0]) * \
                           (max_coords[:, 1] - min_coords[:, 1]) * \
                           (max_coords[:, 2] - min_coords[:, 2])
    
        enclosing_volume = torch.clamp(enclosing_volume, min=self.eps)


        giou3d = iou3d - (enclosing_volume - union_volume) / enclosing_volume

        loss = 1 - giou3d

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none'
            return loss