from torch import nn
import numpy as np
from layers.backbones.lss_fpn import LSSFPN
from layers.heads.bev_height_head import BEVHeightHead
from mmengine.structures import InstanceData
from mmdet3d.structures.bbox_3d.lidar_box3d import LiDARInstance3DBoxes

__all__ = ['BEVHeight']


class BEVHeight(nn.Module):
    """
    Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_height (bool): Whether to return height.
            Default: False.
    """

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(self, backbone_conf, head_conf, is_train_height=False):
        super(BEVHeight, self).__init__()
        self.backbone = LSSFPN(**backbone_conf)
        self.head = BEVHeightHead(**head_conf)
        self.is_train_height = is_train_height

    def forward(
        self,
        x,
        mats_dict,
        timestamps=None,
    ):
        """Forward function for BEVHeight

        Args:
            x (Tensor): Input ferature map.
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps (long): Timestamp.
                Default: None.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        if self.is_train_height and self.training:
            x, height_pred = self.backbone(x,
                                          mats_dict,
                                          timestamps,
                                          is_return_height=True)
            preds = self.head(x)
            return preds, height_pred
        else:
            x = self.backbone(x, mats_dict, timestamps)
            preds = self.head(x)
            return preds

    def get_targets(self, gt_boxes, gt_labels):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        batch_gt_instance_3d = []
        for gt_box,label in zip(gt_boxes,gt_labels) :
            gt_instances_3d = InstanceData()
            gt_instances_3d["bboxes_3d"]=LiDARInstance3DBoxes(
            np.array(
                [[8.7314, -1.8559, -1.5997, 1.2000, 0.4800, 1.8900,
                  -1.5808]]))
            gt_instances_3d["labels_3d"]=np.array([1])
            batch_gt_instance_3d.append(gt_instances_3d)
        return self.head.get_targets(batch_gt_instance_3d)

    def loss(self, targets, preds_dicts):
        """Loss function for BEVHeight.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.loss(targets, preds_dicts)

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)
