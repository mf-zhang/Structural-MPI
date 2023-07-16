# Copyright (c) Facebook, Inc. and its affiliates.
from traceback import print_tb
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms.functional as fn

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher

import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d
from piqa import SSIM
from ssim import SSIM2
import time



def check_tensor_naninf(atensor,name):
    if not torch.isfinite(atensor).all():
        print('zmf',name,"contains inf")
        print(atensor)
        return True
    if torch.isnan(atensor).any():
        print('zmf',name,"contains nan")
        print(atensor)
        return True
    return False


@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
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
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
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

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
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

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        

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
        nonplane_depth_weight = cfg.MODEL.MASK_FORMER.NONPLANE_DEPTH_WEIGHT
        tgt_view_rgb_weight = cfg.MODEL.MASK_FORMER.TGT_VIEW_RGB_WEIGHT
        tgt_view_ssim_weight = cfg.MODEL.MASK_FORMER.TGT_VIEW_SSIM_WEIGHT
        tgt_view_depth_weight = cfg.MODEL.MASK_FORMER.TGT_VIEW_DEPTH_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight, "loss_plane_para_L1": plane_para_L1_weight, "loss_plane_para_cos": plane_para_cos_weight, "loss_plane_para_depth": plane_para_depth_weight, "loss_nonplane_depth": nonplane_depth_weight, "loss_tgt_view_rgb": tgt_view_rgb_weight, "loss_tgt_view_ssim": tgt_view_ssim_weight} #, "loss_tgt_view_depth": tgt_view_depth_weight,}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks", "plane_para", "nonplane_depth", "tgt_view"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            # piqa_ssim = SSIM().cuda()
            piqa_ssim = SSIM2(size_average=True).cuda()
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
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

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]

        images = ImageList.from_tensors(images, self.size_divisibility)


        features = self.backbone(images.tensor)

        conv1_out = self.conv1(images.tensor)
        conv1_out = self.bn1(conv1_out)
        conv1_out = self.relu(conv1_out)
        features['conv1_out'] = conv1_out


        outputs = self.sem_seg_head(features) 



        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                gt_planes_paras = [x["planes_paras"].to(self.device) for x in batched_inputs]
                gt_depth_map = [x["depth_map"].to(self.device) for x in batched_inputs]
                K_inv_dot_xy_1 = [x["K_inv_dot_xy_1"].to(self.device) for x in batched_inputs]
                K_inv_dot_xy_1_2 = [x["K_inv_dot_xy_1_2"].to(self.device) for x in batched_inputs]
                K_inv_dot_xy_1_4 = [x["K_inv_dot_xy_1_4"].to(self.device) for x in batched_inputs]
                K_inv_dot_xy_1_8 = [x["K_inv_dot_xy_1_8"].to(self.device) for x in batched_inputs]
                pixel_plane_para = [x["pixel_plane_para"].to(self.device) for x in batched_inputs]
                depth_from_plane = [x["depth_from_plane"].to(self.device) for x in batched_inputs]
                mask_nonplane_valid = [x["mask_nonplane_valid"].to(self.device) for x in batched_inputs]
                filename = [x["filename"] for x in batched_inputs]


                src_view_image = [x["image"] for x in batched_inputs]
                tgt_view_filename = [x["tgt_filename"] for x in batched_inputs]
                tgt_view_rgb = [x["tgt_image"].to(self.device) for x in batched_inputs]
                tgt_view_depth_map = [x["tgt_depth_map"].to(self.device) for x in batched_inputs]
                G_src_tgt = [x["G_src_tgt"].to(self.device) for x in batched_inputs]
                
                
                targets = self.prepare_targets(gt_instances, gt_planes_paras, gt_depth_map, K_inv_dot_xy_1,  K_inv_dot_xy_1_2,  K_inv_dot_xy_1_4,  K_inv_dot_xy_1_8, pixel_plane_para, depth_from_plane, images, filename, mask_nonplane_valid, tgt_view_filename, tgt_view_rgb, tgt_view_depth_map, G_src_tgt, src_view_image)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)




            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            # print(outputs.keys())
            # assert False
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            pred_plane_paras = outputs["pred_plane_para"]
            # pred_nonplane_rgba = outputs['pred_nonplane_rgba'][0]
            # pred_plane_rgba_results = outputs['pred_plane_rgba'][0] # 1, 100, 4, H/4, W/4
            

            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )



            del outputs
            

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size, pred_plane_para, pred_plane_rgba in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes, pred_plane_paras, pred_plane_rgba_results
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                # if self.sem_seg_postprocess_before_inference:
                #     mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                #         mask_pred_result, image_size, height, width
                #     )
                #     mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result, pred_plane_para, pred_plane_rgba)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                    processed_results[-1]["pred_nonplane_rgba"] = pred_nonplane_rgba
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets(self, targets, planes_paras, depth_map, K_inv_dot_xy_1, K_inv_dot_xy_1_2, K_inv_dot_xy_1_4, K_inv_dot_xy_1_8, pixel_plane_para, depth_from_plane, images, filename, mask_nonplane_valid, tgt_view_filename, tgt_view_rgb, tgt_view_depth_map, G_src_tgt, src_view_image):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []

        assert len(targets) == len(planes_paras) == len(depth_map) # bs

        for idx, targets_per_image in enumerate(targets):
            # pad gt
            gt_masks = targets_per_image.gt_masks

            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            gt_plane_para = planes_paras[idx]
            gt_depth_map = depth_map[idx]
            gt_K_inv_dot_xy_1 = K_inv_dot_xy_1[idx]
            gt_K_inv_dot_xy_1_2 = K_inv_dot_xy_1_2[idx]
            gt_K_inv_dot_xy_1_4 = K_inv_dot_xy_1_4[idx]
            gt_K_inv_dot_xy_1_8 = K_inv_dot_xy_1_8[idx]
            gt_pixel_plane_para = pixel_plane_para[idx]
            gt_depth_from_plane = depth_from_plane[idx]
            gt_filename = filename[idx]
            gt_mask_nonplane_valid = mask_nonplane_valid[idx]
            
            gt_tgt_view_filename = tgt_view_filename[idx]
            gt_tgt_view_rgb = tgt_view_rgb[idx]
            gt_tgt_view_depth_map = tgt_view_depth_map[idx]
            gt_G_src_tgt = G_src_tgt[idx]
            gt_src_view_image = src_view_image[idx]



            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "gt_plane_para": gt_plane_para,
                    "gt_depth_map": gt_depth_map,
                    "K_inv_dot_xy_1": gt_K_inv_dot_xy_1,
                    "K_inv_dot_xy_1_2": gt_K_inv_dot_xy_1_2,
                    "K_inv_dot_xy_1_4": gt_K_inv_dot_xy_1_4,
                    "K_inv_dot_xy_1_8": gt_K_inv_dot_xy_1_8,
                    "gt_pixel_plane_para": gt_pixel_plane_para,
                    "gt_depth_from_plane": gt_depth_from_plane,
                    "filename": gt_filename,
                    "mask_nonplane_valid": gt_mask_nonplane_valid,

                    "gt_tgt_view_filename": gt_tgt_view_filename,
                    "gt_tgt_view_rgb": gt_tgt_view_rgb,
                    "gt_tgt_view_depth_map": gt_tgt_view_depth_map,
                    "gt_G_src_tgt": gt_G_src_tgt,
                    "gt_src_view_image": gt_src_view_image,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred_orig, pred_plane_para, pred_plane_rgba):


        # print(mask_cls[0])                    # 131 neg nums & 2 positive nums at #1 # 133
        # print(F.softmax(mask_cls, dim=-1)[0]) # converted to <1 and sum=1, #1 or # 133 always > 0.9
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        # print(scores.shape,scores) # confidence of 100 predictions being #1 or #133
        # print(labels.shape,labels) # 1 or 133

        mask_pred = mask_pred_orig.sigmoid() # to 0~1

        # print(self.sem_seg_head.num_classes, self.object_mask_threshold) # 133, 0.8
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_masks_orig = mask_pred_orig[keep]
        pred_plane_rgba_keep = pred_plane_rgba[keep]
        pred_plane_para_keep = pred_plane_para[keep]

        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks




        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        # panoptic_seg_global = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info #, panoptic_seg_global
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0) # h, w

            stuff_memory_list = {}

            for k in range(cur_classes.shape[0]):
                this_plane_para = pred_plane_para_keep[k]
                this_plane_rgba = pred_plane_rgba_keep[k]
                this_plane_mask_orig = cur_masks_orig[k]
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5) # 0.25) # 0.5)
                # mask_global = cur_mask_ids == k

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    # print(mask_area,original_area,mask_area / original_area,self.overlap_threshold) 0.8
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id
                    # panoptic_seg_global[mask_global] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                            "plane_para": this_plane_para,
                            "rgba": this_plane_rgba,
                            "mask_orig": this_plane_mask_orig,
                        }
                    )

            return panoptic_seg, segments_info #, panoptic_seg_global

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
