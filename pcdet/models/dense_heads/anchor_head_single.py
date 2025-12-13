import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate
import torch
import cv2
import numpy as np

def get_layer(dim,out_dim,init = None):
    init_func = nn.init.kaiming_normal_
    layers = []
    conv = nn.Conv2d(dim, dim,
                      kernel_size=3, padding=1, bias=True)
    nn.init.normal_(conv.weight, mean=0, std=0.001)
    layers.append(conv)
    layers.append(nn.BatchNorm2d(dim))
    layers.append(nn.ReLU())
    conv2 = nn.Conv2d(dim, out_dim,
                     kernel_size=1, bias=True)

    if init is None:
        nn.init.normal_(conv2.weight, mean=0, std=0.001)
        layers.append(conv2)

    else:
        conv2.bias.data.fill_(init)
        layers.append(conv2)

    return nn.Sequential(*layers)

class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        self.grid_size = grid_size  # [1408 1600   40]
        self.range = point_cloud_range

        self.voxel_size = (point_cloud_range[3] - point_cloud_range[0]) / grid_size[0]


        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )
        self.use_iou_score = self.model_cfg.get("USE_IOU_SCORE", False)
        if self.use_iou_score:
            self.conv_iou = nn.Conv2d(
                input_channels, self.num_anchors_per_location,
                kernel_size=1
            )
            self.alpha = self.model_cfg.get("ALPHA", 0.5)


        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

        #for child in self.children():
        #    for param in child.parameters():
        #        param.requires_grad = False

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
        if self.use_iou_score:
            nn.init.normal_(self.conv_iou.weight, mean=0, std=0.001)
            nn.init.constant_(self.conv_iou.bias, 0)

    def get_anchor_mask(self,data_dict,shape):

        stride = np.round(self.voxel_size*8.*10.)

        minx=self.range[0]
        miny=self.range[1]

        points = data_dict["points"]

        mask = torch.zeros(shape[-2],shape[-1])

        mask_large = torch.zeros(shape[-2]//10,shape[-1]//10)

        in_x = (points[:, 1] - minx) / stride
        in_y = (points[:, 2] - miny) / stride

        in_x = in_x.long().clamp(max=shape[-1]//10-1)
        in_y = in_y.long().clamp(max=shape[-2]//10-1)


        mask_large[in_y,in_x] = 1

        mask_large = mask_large.clone().int().detach().cpu().numpy()

        mask_large_index = np.argwhere( mask_large>0 )

        mask_large_index = mask_large_index*10

        index_list=[]

        for i in np.arange(-10, 10, 1):
            for j in np.arange(-10, 10, 1):
                index_list.append(mask_large_index+[i,j])

        index_list = np.concatenate(index_list,0)

        inds = torch.from_numpy(index_list).cuda().long()

        mask[inds[:,0],inds[:,1]]=1

        return mask.bool()


    def forward(self, data_dict):

        anchor_mask = self.get_anchor_mask(data_dict,data_dict['st_features_2d'].shape)

        new_anchors = []
        for anchors in self.anchors_root:
            new_anchors.append(anchors[:, anchor_mask, ...])

        self.anchors = new_anchors

        st_features_2d = data_dict['st_features_2d']

        cls_preds = self.conv_cls(st_features_2d)
        box_preds = self.conv_box(st_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]  # [N, H, W, C]

        if self.use_iou_score:
            iou_preds = self.conv_iou(st_features_2d)
            iou_preds = iou_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]
            self.forward_ret_dict['iou_preds'] = iou_preds

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(st_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None



        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)
            data_dict['gt_ious'] = targets_dict['gt_ious']

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            
            if self.use_iou_score:
                batch_iou_preds = iou_preds.view(batch_box_preds.shape[0], -1, 1)
                batch_cls_preds = (batch_cls_preds.sigmoid()**(1-self.alpha) * batch_iou_preds.sigmoid()**self.alpha).pow(0.5)
                batch_cls_preds = torch.clamp(batch_cls_preds, min=0.001, max=1.0)
                data_dict['cls_preds_normalized'] = True
            else:
                data_dict['cls_preds_normalized'] = False


            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds

        if self.model_cfg.get('NMS_CONFIG', None) is not None:
            self.proposal_layer(
                data_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
            )

        return data_dict

    def get_iou_loss(self):
        iou_preds = self.forward_ret_dict['iou_preds']
        iou_targets = self.forward_ret_dict['gt_ious']
        pos_mask = self.forward_ret_dict['box_cls_labels'] > 0
        
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        iou_loss = nn.functional.binary_cross_entropy_with_logits(iou_preds[pos_mask], iou_targets[pos_mask], reduction='none')
        iou_loss = (iou_loss * loss_cfgs.LOSS_WEIGHTS['iou_pred_weight']).sum() / torch.clamp(pos_mask.sum(), min=1.0)
        
        tb_dict = {
            'rpn_loss_iou': iou_loss.item()
        }
        return iou_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss
        
        if self.use_iou_score:
            iou_loss, tb_dict_iou = self.get_iou_loss()
            tb_dict.update(tb_dict_iou)
            rpn_loss += iou_loss

        if self.model_cfg.get('OD_LOSS',False):
            od_loss = self.get_od_loss()
            rpn_loss += od_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

