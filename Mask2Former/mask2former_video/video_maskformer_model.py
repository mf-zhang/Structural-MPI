# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from typing import Tuple

import torch, os, time
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.layers import Conv2d
import fvcore.nn.weight_init as weight_init

from .modeling.criterion import VideoSetCriterion
from .modeling.matcher import VideoHungarianMatcher
from .utils.memory import retry_if_cuda_oom

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class VideoMaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        num_frames,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames

        if os.environ['TRAIN_PHASE'] in ['1','2']:
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = Conv2d(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
            weight_init.c2_xavier_fill(self.conv1)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        plane_para_L1_weight = cfg.MODEL.MASK_FORMER.PLANE_PARA_L1_WEIGHT
        plane_para_cos_weight = cfg.MODEL.MASK_FORMER.PLANE_PARA_COS_WEIGHT
        plane_para_depth_weight = cfg.MODEL.MASK_FORMER.PLANE_PARA_DEPTH_WEIGHT
        # nonplane_depth_weight = cfg.MODEL.MASK_FORMER.NONPLANE_DEPTH_WEIGHT
        tgt_view_rgb_weight = cfg.MODEL.MASK_FORMER.TGT_VIEW_RGB_WEIGHT
        tgt_view_ssim_weight = cfg.MODEL.MASK_FORMER.TGT_VIEW_SSIM_WEIGHT
        tgt_view_depth_weight = cfg.MODEL.MASK_FORMER.TGT_VIEW_DEPTH_WEIGHT


        # building criterion
        matcher = VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        if os.environ['TRAIN_PHASE'] in ['0','1']:
            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight, "loss_plane_para_L1": plane_para_L1_weight, "loss_plane_para_cos": plane_para_cos_weight, "loss_plane_para_depth": plane_para_depth_weight} #, "loss_nonplane_depth": nonplane_depth_weight, "loss_tgt_view_rgb": tgt_view_rgb_weight, "loss_tgt_view_ssim": tgt_view_ssim_weight} 
        elif os.environ['TRAIN_PHASE'] in ['2']:
            weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight, "loss_plane_para_L1": plane_para_L1_weight, "loss_plane_para_cos": plane_para_cos_weight, "loss_plane_para_depth": plane_para_depth_weight, "loss_tgt_view_rgb": tgt_view_rgb_weight, "loss_tgt_view_ssim": tgt_view_ssim_weight, "loss_tgt_view_depth": tgt_view_depth_weight}
            if os.environ['SAMPLING_FRAME_NUM'] == '2':
                weight_dict['loss_tgt_view_rgb_merge'] = tgt_view_rgb_weight
                weight_dict['loss_tgt_view_ssim_merge'] = tgt_view_ssim_weight

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        if os.environ['ONLY_SEG'] == 'True':
            losses = ['labels', 'masks']#, 'plane_para'] #, 'nonplane_depth', 'tgt_view']
        elif os.environ['TRAIN_PHASE'] in ['0','1']:
            losses = ['labels', 'masks', 'plane_para'] #, 'nonplane_depth', 'tgt_view']
        elif os.environ['TRAIN_PHASE'] in ['2']:
            losses = ['labels', 'masks', 'plane_para', 'tgt_view']

        criterion = VideoSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            HW=cfg.INPUT.IMAGE_SIZE
        )

        return {
            'backbone': backbone,
            'sem_seg_head': sem_seg_head,
            'criterion': criterion,
            'num_queries': cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            'object_mask_threshold': cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            'overlap_threshold': cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            'metadata': MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            'size_divisibility': cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            'sem_seg_postprocess_before_inference': True,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            # video
            'num_frames': cfg.INPUT.SAMPLING_FRAME_NUM,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        # os.environ['TIME06'] = '%.2f'%(time.process_time()); print('5-6:',float(os.environ['TIME06'])-float(os.environ['TIME05']))
        images = []
        depths = []
        if os.environ['SEG_NONPLANE'] == 'True':
            depths2 = []
        if os.environ['TRAIN_PHASE'] == '2':
            tgt_images = []
            tgt_depths = []
        for video in batched_inputs:
            for frame in video['image']:
                images.append(frame.to(self.device))
            for depth_frame in video['depth_from_plane']:
                depths.append(depth_frame.to(self.device))
            if os.environ['SEG_NONPLANE'] == 'True':
                for depth_frame in video['depth_from_plane_and_nonplane']:
                    depths2.append(depth_frame.to(self.device))
            if os.environ['TRAIN_PHASE'] == '2':
                tgt_images.append(video['tgt_image'])
                tgt_depths.append(video['tgt_depth_map'])
        
        images_orig = ImageList.from_tensors(images, self.size_divisibility)
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        depths = ImageList.from_tensors(depths, self.size_divisibility)
        if os.environ['SEG_NONPLANE'] == 'True':
            depths2 = ImageList.from_tensors(depths2, self.size_divisibility)
        else:
            depths2 = None
        if os.environ['TRAIN_PHASE'] == '2':
            tgt_images = ImageList.from_tensors(tgt_images, self.size_divisibility)
            tgt_depths = ImageList.from_tensors(tgt_depths, self.size_divisibility)
            # print(images.tensor.shape, depths.tensor.shape, tgt_images.tensor.shape, tgt_depths.tensor.shape)
            # torch.Size([4, 3, 256, 384]) torch.Size([4, 256, 384]) torch.Size([2, 3, 256, 384]) torch.Size([2, 256, 384])
        else:
            tgt_images, tgt_depths = None, None
        # os.environ['TIME02'] = '%.2f'%(time.process_time()); print('1-2:',float(os.environ['TIME02'])-float(os.environ['TIME06']))
        
        features = self.backbone(images.tensor)

        if os.environ['TRAIN_PHASE'] in ['1','2']:
            conv1_out = self.conv1(images.tensor)
            conv1_out = self.bn1(conv1_out)
            conv1_out = self.relu(conv1_out)
            features['conv1_out'] = conv1_out

        outputs = self.sem_seg_head(features)
        # os.environ['TIME03'] = '%.2f'%(time.process_time()); print('2-3:',float(os.environ['TIME03'])-float(os.environ['TIME02']))

        if os.environ['TEST_NVS_ONLY'] != 'True':
            targets = self.prepare_targets(batched_inputs, images, depths, depths2, tgt_images, tgt_depths, images_orig)
        else:
            targets = []
            for bid,targets_per_video in enumerate(batched_inputs):
                targets.append({"gt_cam_pose": torch.stack(targets_per_video['cam_pose']).to(self.device)})
        
        if self.training:
            # mask classification target
            
            # os.environ['TIME04'] = '%.2f'%(time.process_time()); print('3-4:',float(os.environ['TIME04'])-float(os.environ['TIME03']))
            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)
            # os.environ['TIME05'] = '%.2f'%(time.process_time()); print('4-5:',float(os.environ['TIME05'])-float(os.environ['TIME04']))

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs['pred_logits']
            mask_pred_results = outputs['pred_masks']
            pred_plane_paras = outputs['pred_plane_para']
            if os.environ['TRAIN_PHASE'] in ['1','2']:
                pred_alphas = outputs['pred_alpha']
            else:
                pred_alphas = outputs['pred_masks']
            if os.environ['TRAIN_PHASE'] in ['2']:
                pred_rgbs = outputs['pred_rgb']
            else:
                pred_rgbs = outputs['pred_masks']

            # print(mask_cls_results.shape, mask_pred_results.shape, pred_alphas.shape,pred_rgbs.shape ) 
            # torch.Size([1, 100, 134]) torch.Size([1, 100, 2, 64, 96]) torch.Size([1, 100, 2, 256, 384]) torch.Size([1, 100, 2, 3, 256, 384])

            del outputs

            processed_results = []
            if not os.environ['ONLY_SEG'] == 'True':
                for mask_cls_result, mask_pred_result, input_per_image, image_size, pred_plane_para, pred_alpha, pred_rgb, target in zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes, pred_plane_paras, pred_alphas, pred_rgbs, targets
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    mask_pred_result = F.interpolate(
                        mask_pred_result,
                        size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                        mode="bilinear",
                        align_corners=False,
                    )

                    if os.environ['RUN_ON_SCANNET'] == 'False':
                        return retry_if_cuda_oom(self.inference_video)(mask_cls_result, mask_pred_result, image_size, height, width)
                    else:
                        panoptic_r =  retry_if_cuda_oom(self.inference_scannet_video)(mask_cls_result, mask_pred_result, pred_plane_para, pred_alpha, pred_rgb, target)
                        processed_results[-1]['panoptic_seg'] = panoptic_r
                        # processed_results[-1]['pred_nonplane_rgba'] = pred_nonplane_rgba
            else:
                for mask_cls_result, mask_pred_result, input_per_image, image_size, target in zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes, targets
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})

                    mask_pred_result = F.interpolate(
                        mask_pred_result,
                        size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                        mode="bilinear",
                        align_corners=False,
                    )

                    if os.environ['RUN_ON_SCANNET'] == 'False':
                        return retry_if_cuda_oom(self.inference_video)(mask_cls_result, mask_pred_result, image_size, height, width)
                    else:
                        panoptic_r =  retry_if_cuda_oom(self.inference_scannet_video)(mask_cls_result, mask_pred_result, None, target)#, pred_plane_rgba)
                        processed_results[-1]['panoptic_seg'] = panoptic_r
                        # processed_results[-1]['pred_nonplane_rgba'] = pred_nonplane_rgba
            
            return processed_results

    def prepare_targets(self, targets, images, depths, depths2, tgt_images, tgt_depths, src_image_orig):
        BT, _, h_pad, w_pad = images.tensor.shape
        gt_instances = []
        B = len(targets)
        T = int(BT/B)
        for bid,targets_per_video in enumerate(targets):
            if os.environ['RUN_ON_SCANNET'] == 'False':
                _num_instance = len(targets_per_video['instances'][0])
                mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
                gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

                gt_ids_per_video = []
                for f_i, targets_per_frame in enumerate(targets_per_video['instances']):
                    targets_per_frame = targets_per_frame.to(self.device)
                    h, w = targets_per_frame.image_size

                    gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor

                gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
                valid_idx = (gt_ids_per_video != -1).any(dim=-1)

                gt_classes_per_video = targets_per_frame.gt_classes[valid_idx]          # N,
                gt_ids_per_video = gt_ids_per_video[valid_idx]                          # N, num_frames

                gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
                gt_masks_per_video = gt_masks_per_video[valid_idx].float()          # N, num_frames, H, W
                gt_instances[-1].update({"masks": gt_masks_per_video})
            else:
                # GET P_min in _num_instance
                # os.environ['TIME031'] = '%.2f'%(time.process_time())
                if True:
                    _num_instance = []
                    for fi in range(len(targets_per_video['instances'])):
                        _num_instance_this = len(targets_per_video['instances'][fi])
                        _num_instance.append(_num_instance_this)
                    # print(_num_instance)
                    _num_instance = min(_num_instance)

                # os.environ['TIME032'] = '%.2f'%(time.process_time()); print('3.1-3.2:',float(os.environ['TIME032'])-float(os.environ['TIME031']))

                # PREPARE GT MASK
                mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
                class_shape = [_num_instance, self.num_frames, 1]
                gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)
                gt_classes_per_video = torch.zeros(class_shape, dtype=torch.int64, device=self.device)
                for f_i, targets_per_frame in enumerate(targets_per_video['instances']):
                    targets_per_frame = targets_per_frame.to(self.device)
                    h, w = targets_per_frame.image_size
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor[:_num_instance]
                    gt_classes_per_video[:, f_i, 0] = targets_per_frame.gt_classes[:_num_instance]

                # os.environ['TIME033'] = '%.2f'%(time.process_time()); print('3.2-3.3:',float(os.environ['TIME033'])-float(os.environ['TIME032']))

                if os.environ['SINGLE_MATCH'] != 'True':
                    # Attention: targets_per_frame is the last frame
                    gt_classes_per_video = targets_per_frame.gt_classes[:_num_instance]          # N,

                gt_instances.append({"labels": gt_classes_per_video}) 
                gt_masks_per_video = gt_masks_per_video.float()          # N, num_frames, H, W
                gt_instances[-1].update({"masks": gt_masks_per_video})

                # os.environ['TIME034'] = '%.2f'%(time.process_time()); print('3.3-3.4:',float(os.environ['TIME034'])-float(os.environ['TIME033']))
                if os.environ['TRAIN_MP3D'] == 'True':
                    plane_para_shape = [_num_instance, self.num_frames, 3]
                else:
                    plane_para_shape = [_num_instance, self.num_frames, 4]
                gt_plane_para_per_video = torch.zeros(plane_para_shape, device=self.device)
                for f_i, targets_per_frame in enumerate(targets_per_video['planes_paras']):
                    targets_per_frame = targets_per_frame.to(self.device) # P, 4
                    gt_plane_para_per_video[:, f_i, :] = targets_per_frame[:_num_instance]
                gt_instances[-1].update({"gt_plane_para": gt_plane_para_per_video}) # P_min, T, 4
                # os.environ['TIME035'] = '%.2f'%(time.process_time()); print('3.4-3.5:',float(os.environ['TIME035'])-float(os.environ['TIME034']))

                if os.environ['TRAIN_MP3D'] == 'True':
                    gt_instances[-1].update({"gt_cam_pose": targets_per_video['cam_pose']})
                else:
                    gt_instances[-1].update({"gt_cam_pose": torch.stack(targets_per_video['cam_pose']).to(self.device)}) # T, 4, 4
                gt_instances[-1].update({"nonplane_num": targets_per_video['nonplane_num']}) # 1

                # os.environ['TIME036'] = '%.2f'%(time.process_time()); print('3.5-3.6:',float(os.environ['TIME036'])-float(os.environ['TIME035']))
                
                gt_instances[-1].update({"gt_depth_from_plane": depths.tensor[bid*T:bid*T+T]}) # T, H, W
                if os.environ['SEG_NONPLANE'] == 'True':
                    # method 1
                    # gt_instances[-1].update({"gt_depth_from_plane_and_nonplane": torch.stack(targets_per_video['depth_from_plane_and_nonplane']).to(self.pixel_mean)}) # T, H, W

                    # method 2
                    # this_depths = []
                    # for fdepth in targets_per_video['depth_from_plane_and_nonplane']:
                    #     this_depths.append(fdepth.to(self.device))
                    # this_depths = ImageList.from_tensors(this_depths, self.size_divisibility)
                    # gt_instances[-1].update({"gt_depth_from_plane_and_nonplane": this_depths.tensor}) # T, H, W

                    # method 3
                    gt_instances[-1].update({"gt_depth_from_plane_and_nonplane": depths2.tensor[bid*T:bid*T+T]}) # T, H, W

                # os.environ['TIME037'] = '%.2f'%(time.process_time()); print('3.6-3.7:',float(os.environ['TIME037'])-float(os.environ['TIME036']))
                # delete to speed up
                # gt_instances[-1].update({"gt_depth_map": torch.stack(targets_per_video['depth_map']).to(self.device)}) # T, H, W 
                # gt_instances[-1].update({"K_inv_dot_xy_1": targets_per_video['K_inv_dot_xy_1'].to(self.device)})     # 3, H, W
                # gt_instances[-1].update({"K_inv_dot_xy_1_2": targets_per_video['K_inv_dot_xy_1_2'].to(self.device)}) # 3, H/2, W/2
                # gt_instances[-1].update({"K_inv_dot_xy_1_4": targets_per_video['K_inv_dot_xy_1_4'].to(self.device)}) # 3, H/4, W/4
                # gt_instances[-1].update({"K_inv_dot_xy_1_8": targets_per_video['K_inv_dot_xy_1_8'].to(self.device)}) # 3, H/8, W/8

                if os.environ['TRAIN_PHASE'] == '2':
                    gt_instances[-1].update({"gt_tgt_image": tgt_images.tensor[bid:bid+1]}) # 1, 3, H, W
                    gt_instances[-1].update({"gt_tgt_depth": tgt_depths.tensor[bid:bid+1]}) # 1, H, W
                    gt_instances[-1].update({"gt_src_image": src_image_orig.tensor[bid*T:bid*T+T]}) # T, 3, H, W
                    gt_instances[-1].update({"gt_tgt_G_src_tgt": torch.stack(targets_per_video['tgt_G_src_tgt']).to(self.device)})
                    gt_instances[-1].update({"gt_tgt_cam_pose": targets_per_video['tgt_cam_pose'].to(self.device)})
                    gt_instances[-1].update({"tgt_filename": targets_per_video['tgt_filename']})
                
                # os.environ['TIME036'] = '%.2f'%(time.process_time()); print('3.5-3.6:',float(os.environ['TIME036'])-float(os.environ['TIME035']))
                             
        return gt_instances

    def inference_video(self, pred_cls, pred_masks, img_size, output_height, output_width):
        # print(pred_cls.shape, pred_masks.shape, img_size, output_height, output_width)
        # torch.Size([100, 134]) torch.Size([100, 2, 256, 384]) (256, 384) 480 640
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(10, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            pred_masks = pred_masks[topk_indices]

            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )

            masks = pred_masks > 0.

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output

    
    def plane_para4_trans(self, src_plane_para_BQ4, G_src_tgt_B44):
        # G_src_tgt = gt_src_pose @ torch.linalg.inv(gt_tgt_pose)
        B,Q,_ = src_plane_para_BQ4.shape

        G_src_tgt_transpose_B44 = G_src_tgt_B44.transpose(1,2)
        G_src_tgt_transpose_BQ44 = G_src_tgt_transpose_B44[:,None,...].repeat(1,Q,1,1)
        src_plane_para_BQ41 = src_plane_para_BQ4[...,None]

        src_plane_para_BQ4[:,:,3] = -src_plane_para_BQ4[:,:,3]
        tgt_view_plane_para_BQ4 = torch.matmul(G_src_tgt_transpose_BQ44, src_plane_para_BQ41)[:,:,:,0]
        tgt_view_plane_para_BQ4[:,:,3] = -tgt_view_plane_para_BQ4[:,:,3]
        src_plane_para_BQ4[:,:,3] = -src_plane_para_BQ4[:,:,3]
        return tgt_view_plane_para_BQ4

    def plane_para_3to4(self,plane_para_3):
        if len(plane_para_3.shape) == 2:
            offset = torch.reciprocal(torch.norm(plane_para_3, dim=1, keepdim=True))
            plane_para_3 = plane_para_3 * offset
            plane_para_4 = torch.cat([plane_para_3,offset],dim=1)
        elif len(plane_para_3.shape) == 3:
            offset = torch.reciprocal(torch.norm(plane_para_3, dim=2, keepdim=True))
            plane_para_3 = plane_para_3 * offset
            plane_para_4 = torch.cat([plane_para_3,offset],dim=2)

        sign_correct = False
        if sign_correct:
            # the third in plane_para_3 has to be +
            # the third and the fourth in plane_para_4 has to be the same sign
            # two plane_para_4 with only sign difference, will be the same after converting to plane_para_3

            # below is not correct
            if len(plane_para_3.shape) == 2:
                plane_para_4abs = torch.abs(plane_para_4)
                sign = plane_para_4/plane_para_4abs
                sign = sign[:,1:2]
                plane_para_4 = plane_para_4 * sign.repeat(1,4)
            elif len(plane_para_3.shape) == 3:
                plane_para_4abs = torch.abs(plane_para_4)
                sign = plane_para_4/plane_para_4abs
                sign = sign[:,:,1:2]
                plane_para_4 = plane_para_4 * sign.repeat(1,1,4)
        return plane_para_4

    def plane_para_4to3(self,plane_para_4):
        if len(plane_para_4.shape) == 2:
            plane_para_3 = plane_para_4[:,:3] / plane_para_4[:,3:]
        elif len(plane_para_4.shape) == 3:
            plane_para_3 = plane_para_4[:,:,:3] / plane_para_4[:,:,3:]
        return plane_para_3

    def inference_scannet_video(self, pred_cls, pred_masks_orig, pred_plane_para, pred_alpha, pred_rgb, target):
        # print(pred_cls.shape, pred_masks.shape)
        # torch.Size([100, 134]) torch.Size([100, 2, 256, 384]) (256, 384) 480 640
        scores, labels = F.softmax(pred_cls, dim=-1).max(-1)

        device_type = pred_masks_orig.float()

        mask_pred = pred_masks_orig.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold) # filter 1
        cur_scores = scores[keep]
        cur_classes = labels[keep] # P
        cur_masks = mask_pred[keep] # P, 2, H, W
        cur_masks_orig = pred_masks_orig[keep]

        if not os.environ['ONLY_SEG'] == 'True':
            # pred_plane_rgba_keep = pred_plane_rgba[keep]
            pred_plane_para_keep = pred_plane_para[keep] # P, 3
            if os.environ['TRAIN_PHASE'] in ['1','2']:
                pred_alpha_keep = pred_alpha[keep] # P, T, H, W
            if os.environ['TRAIN_PHASE'] in ['2']:
                pred_rgb_keep = pred_rgb[keep] # P, T, 3, H, W

            if cur_masks.shape[1] == 2:
                cam_pose0 = target['gt_cam_pose'][0][None,...].to(device_type)
                cam_pose1 = target['gt_cam_pose'][1][None,...].to(device_type)
                G_src_tgt = cam_pose0 @ torch.linalg.inv(cam_pose1) # 1 4 4

                pred_plane_para0_3 = pred_plane_para_keep[None,...].to(device_type) # 1 P 3
                pred_plane_para0_4 = self.plane_para_3to4(pred_plane_para0_3) # 1 P 4
                pred_plane_para1_4 = self.plane_para4_trans(pred_plane_para0_4,G_src_tgt) # 1 P 4
                pred_plane_para1_3 = self.plane_para_4to3(pred_plane_para1_4) # 1 P 3

                pred_plane_para_keep = torch.cat([pred_plane_para0_3[0][:,None,:],pred_plane_para1_3[0][:,None,:]],dim=1) # P, 2, 3
            else:
                pred_plane_para_keep = pred_plane_para_keep[:,None,:].to(device_type)

        cur_prob_masks = cur_scores.view(-1, 1, 1, 1) * cur_masks # P, 2, H, W

        panoptic_segs = []
        segments_infos = []

        for fi in range(cur_masks.shape[1]):
            h, w = cur_masks.shape[-2:]
            panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
            segments_info = []

            current_segment_id = 0
            if cur_masks.shape[0] == 0:
                # We didn't detect any mask :(
                panoptic_segs.append(panoptic_seg)
                segments_infos.append(segments_info)
            else:
                # 2 to 1
                this_cur_classes = cur_classes # [P]
                this_cur_masks = cur_masks[:,fi,:,:] # [P, H, W]
                this_cur_masks_orig = cur_masks_orig[:,fi,:,:] # [P, H, W]
                this_cur_prob_masks = cur_prob_masks[:,fi,:,:] # [P, H, W]
                if not os.environ['ONLY_SEG'] == 'True':
                    this_pred_plane_para_keep = pred_plane_para_keep[:,fi,:] # [P,3]
                if os.environ['TRAIN_PHASE'] in ['2']:
                    this_pred_plane_RGBA_keep = torch.cat([pred_rgb_keep,pred_alpha_keep[:,:,None,:,:]], dim=2)[:,fi,:,:,:] # P 4 H W

                # take argmax
                cur_mask_ids = this_cur_prob_masks.argmax(0) # h, w
                stuff_memory_list = {}

                for k in range(this_cur_classes.shape[0]): # P
                    if not os.environ['ONLY_SEG'] == 'True':
                        this_plane_para = this_pred_plane_para_keep[k]
                    else:
                        this_plane_para = None
                    if os.environ['TRAIN_PHASE'] in ['2']:
                        this_plane_RGBA = this_pred_plane_RGBA_keep[k]
                    else:
                        this_plane_RGBA = None
                    # this_plane_rgba = pred_plane_rgba_keep[k]
                    this_plane_mask_orig = this_cur_masks_orig[k]
                    pred_class = this_cur_classes[k].item()
                    isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values() # 0 & 50 both true
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (this_cur_masks[k] >= 0.5).sum().item()
                    if os.environ['ALL_COVER'] == 'True':
                        mask = (cur_mask_ids == k) & (this_cur_masks[k] > 0.) # 0.25) # filter 2
                    else:
                        mask = (cur_mask_ids == k) & (this_cur_masks[k] >= 0.5) # 0.25) # filter 2

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0: # filter 3
                        if mask_area / original_area < self.overlap_threshold:
                            continue

                        # merge stuff regions
                        if not isthing:
                            assert False
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = current_segment_id + 1

                        current_segment_id += 1
                        panoptic_seg[mask] = current_segment_id
                        # panoptic_seg_global[mask_global] = current_segment_id

                        cate_id = 90
                        isplane = False
                        if int(pred_class) == 0:
                            cate_id = k
                            isplane = True
                        elif int(pred_class) == 50:
                            cate_id = 50+k
                            isplane = False

                        segments_info.append(
                            {
                                "id": current_segment_id,
                                "isthing": bool(isthing),
                                "category_id": int(cate_id),
                                "plane_para": this_plane_para,
                                "plane_RGBA": this_plane_RGBA,
                                "isplane": bool(isplane),
                                # "rgba": this_plane_rgba,
                                "mask_orig": this_plane_mask_orig,
                            }
                        )

                panoptic_segs.append(panoptic_seg)
                segments_infos.append(segments_info)
            
        return panoptic_segs, segments_infos