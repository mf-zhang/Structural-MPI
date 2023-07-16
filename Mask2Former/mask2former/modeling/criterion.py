# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
from ensurepip import bootstrap
import logging

import torch, os, time
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
import mpi_rendering
from ssim import SSIM2
from piqa import SSIM

def drawDepthImage(depth, maxDepth=5):
    import cv2
    import numpy as np
    depthImage = np.clip(depth / maxDepth * 255, 0, 255).astype(np.uint8)
    depthImage = cv2.applyColorMap(255 - depthImage, colormap=cv2.COLORMAP_JET)
    return depthImage

def drawDepth(depth,addr):
    import cv2
    import numpy as np
    assert depth.dim() == 3 or depth.dim() == 2
    if depth.dim() == 3:
        depth = depth[0].cpu().numpy()
    else:
        depth = depth.cpu().numpy()

    depth_color = drawDepthImage(depth)
    depth_mask = depth > 1e-4
    depth_mask = depth_mask[:, :, np.newaxis]
    depth_color = depth_color * depth_mask
    cv2.imwrite(addr, depth_color)

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

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks

sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

def plane_para_loss_L1(src_param, target_param):
    loss_param_l1 = torch.mean(torch.sum(torch.abs(target_param - src_param), dim=1))
    return loss_param_l1

plane_para_loss_L1_jit = torch.jit.script(
    plane_para_loss_L1
)  # type: torch.jit.ScriptModule

def plane_para_loss_cos(src_param, target_param):
    similarity = torch.nn.functional.cosine_similarity(src_param, target_param, dim=1)  # N
    loss_param_cos = torch.mean(1-similarity)
    # angle = torch.mean(torch.acos(torch.clamp(similarity, -1, 1)))
    return loss_param_cos

plane_para_loss_cos_jit = torch.jit.script(
    plane_para_loss_cos
)  # type: torch.jit.ScriptModule

def depth_loss(input, target, mask):
    if mask is not None:
        input = input[mask]
        target = target[mask]
    g = torch.log(input) - torch.log(target)
    # n, c, h, w = g.shape
    # norm = 1/(h*w)
    # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

    Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
    return 10 * torch.sqrt(Dg)

depth_loss_jit = torch.jit.script(
    depth_loss
)  # type: torch.jit.ScriptModule

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, piqa_ssim):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        # zmf
        self.ssim2 = SSIM2(size_average=True).cuda()
        self.ssim = piqa_ssim

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        bootstrap_alpha = (os.environ['BOOT'] == 'true') and ("pred_plane_rgba" in outputs)

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        if bootstrap_alpha:
            src_alpha_0 = outputs["pred_plane_rgba"][0][:,:,3,:,:]
            src_alpha_0 = src_alpha_0[src_idx][:, None]
            src_alpha_1 = outputs["pred_plane_rgba"][1][:,:,3,:,:]
            src_alpha_1 = src_alpha_1[src_idx][:, None]
            src_alpha_2 = outputs["pred_plane_rgba"][2][:,:,3,:,:]
            src_alpha_2 = src_alpha_2[src_idx][:, None]
            src_alpha_3 = outputs["pred_plane_rgba"][3][:,:,3,:,:]
            src_alpha_3 = src_alpha_3[src_idx][:, None]


        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        if bootstrap_alpha:
            point_logits_0 = point_sample(
                src_alpha_0,
                point_coords,
                align_corners=False,
            ).squeeze(1)
            point_logits_1 = point_sample(
                src_alpha_1,
                point_coords,
                align_corners=False,
            ).squeeze(1)
            point_logits_2 = point_sample(
                src_alpha_2,
                point_coords,
                align_corners=False,
            ).squeeze(1)
            point_logits_3 = point_sample(
                src_alpha_3,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        
        loss_mask = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks)
        loss_dice = dice_loss_jit(point_logits, point_labels, num_masks)

        if bootstrap_alpha:
            loss_mask = loss_mask + sigmoid_ce_loss_jit(point_logits_0, point_labels, num_masks) + sigmoid_ce_loss_jit(point_logits_1, point_labels, num_masks) + sigmoid_ce_loss_jit(point_logits_2, point_labels, num_masks) + sigmoid_ce_loss_jit(point_logits_3, point_labels, num_masks)
            loss_mask = loss_mask / 5
            loss_dice = loss_dice + dice_loss_jit(point_logits_0, point_labels, num_masks) + dice_loss_jit(point_logits_1, point_labels, num_masks) + dice_loss_jit(point_logits_2, point_labels, num_masks) + dice_loss_jit(point_logits_3, point_labels, num_masks)
            loss_dice = loss_dice / 5

        losses = {
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
        }

        del src_masks
        del target_masks
        return losses

    def loss_plane_para(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        # print(outputs.keys()) # dict_keys(['pred_logits', 'pred_plane_para', 'pred_masks', 'aux_outputs'])
        # print(outputs['pred_plane_para'].shape) # 3, 100, 3
        # print(outputs['pred_logits'].shape) # 3, 100, 134
        # print(outputs['pred_masks'].shape) # 3, 100, 256, 256

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"] # 3, 100, 256, 256
        src_masks = src_masks[src_idx]
        src_plane_para = outputs['pred_plane_para'][src_idx]

        # print(src_masks.shape) # torch.Size([20, 256, 256])
        # print(src_plane_para.shape) # torch.Size([20, 3])
        



        masks             = [t["masks"] for t in targets]
        target_plane_para = [t["gt_plane_para"][:,:,None] for t in targets]
        # target_depth_map  = [t["gt_depth_map"] for t in targets]
        target_depth_map  = [t["gt_depth_from_plane"] for t in targets]
        K_inv_dot_xy_1    = [t["K_inv_dot_xy_1"] for t in targets]
        filenames         = [t["filename"] for t in targets]
        print_filename = ''
        for filename in filenames:
            print_filename += filename
            print_filename += '  '

        # TODO use valid to mask invalid areas due to padding in loss
        # for i in range(len(masks)):
        #     print('in',masks[i].shape,len(masks))
        # in torch.Size([4, 1024, 1024]) 3
        # in torch.Size([7, 1024, 1024]) 3
        # in torch.Size([26, 1024, 1024]) 3
        target_masks_nest, valid = nested_tensor_from_tensor_list(masks).decompose()

        target_masks_nest = target_masks_nest.to(src_masks)
        # print(target_masks.shape) # torch.Size([3, 26, 1024, 1024])
        target_masks = target_masks_nest[tgt_idx]
        # print(target_masks.shape) # torch.Size([42, 1024, 1024])

        # for i in range(len(target_plane_para)):
        #     print('in',target_plane_para[i].shape,target_plane_para[i])
        target_plane_para, valid = nested_tensor_from_tensor_list(target_plane_para).decompose()
        target_plane_para = target_plane_para.to(src_plane_para)
        # print(target_plane_para.shape) # torch.Size([3, 26, 4, 1])
        target_plane_para = target_plane_para[tgt_idx][:,:,0]
        # print(gt_plane_para.shape) # torch.Size([42, 4])
        
        # check_tensor_naninf(target_depth_map[0],'before '+print_filename)
        target_depth_map = torch.stack(target_depth_map).to(src_plane_para)
        # check_tensor_naninf(target_depth_map,'after '+print_filename)

        K_inv_dot_xy_1   = torch.stack(K_inv_dot_xy_1).to(src_plane_para) 




        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        # print(target_plane_para[:5])
        target_plane_para = target_plane_para[:,:3] / target_plane_para[:,3:]

        ###################################################################
        # with torch.no_grad():
        #     # sample point_coords
        #     point_coords = get_uncertain_point_coords_with_randomness(
        #         src_masks,
        #         lambda logits: calculate_uncertainty(logits),
        #         self.num_points,
        #         self.oversample_ratio,
        #         self.importance_sample_ratio,
        #     )
        #     # get gt labels
        #     point_labels = point_sample(
        #         target_masks,
        #         point_coords,
        #         align_corners=False,
        #     ).squeeze(1)

        # point_logits = point_sample(
        #     src_masks,
        #     point_coords,
        #     align_corners=False,
        # ).squeeze(1)
        ####################################################################

        # print(target_depth_map.shape) # torch.Size([b, 480, 640])
        # print(K_inv_dot_xy_1.shape) # torch.Size([b, 3, 480, 640])
        gt_depths = target_depth_map[:,None]  # b, 1, 480, 640 depth_from_plane

        b, _, h, w = gt_depths.shape

        assert b == len(targets)

        losses_Q = 0.

        have_naninf = False
        for bi in range(b):
            num_planes = len(indices[bi][0])

            segmentation = target_masks_nest[bi] # b, max_num, h, w
            device = segmentation.device

            depth = gt_depths[bi]  # 1, h, w
            # check_tensor_naninf(depth,'depth '+print_filename)
            k_inv_dot_xy1_map = (K_inv_dot_xy_1[bi]).clone().view(3, h, w).to(device)
            # check_tensor_naninf(k_inv_dot_xy1_map,'k_inv_dot_xy1_map '+print_filename)
            gt_pts_map = k_inv_dot_xy1_map * depth  # 3, h, w
            # check_tensor_naninf(gt_pts_map,'gt_pts_map '+print_filename)

            indices_bi = indices[bi]
            idx_out = indices_bi[0]
            idx_tgt = indices_bi[1]
            assert idx_tgt.max() + 1 == num_planes

            # select pixel with segmentation
            loss_bi = 0.
            for i in range(num_planes):
                gt_plane_idx = int(idx_tgt[i])
                # print('hello',h,w,segmentation.shape)
                mask = segmentation[gt_plane_idx, :, :].view(1, h, w)
                mask = mask > 0

                pts = torch.masked_select(gt_pts_map, mask).view(3, -1)  # 3, plane_pt_num
                # bad = check_tensor_naninf(pts,'pts '+print_filename)

                pred_plane_idx = int(idx_out[i])

                param = outputs['pred_plane_para'][bi][pred_plane_idx].view(1, 3)
                # param = targets[bi][gt_plane_idx, 1:].view(1, 3)
                
                #########################################
                # param_gt = targets[bi][gt_plane_idx, 1:4].view(1, 3)
                # gt_err = torch.mean(torch.abs(torch.matmul(param_gt, pts) - 1))  # 1, plane_pt_num
                # print(gt_err)
                #########################################
                # bad = check_tensor_naninf(param,'param '+print_filename)
                loss = torch.abs(torch.matmul(param, pts) - 1)  # 1, plane_pt_num
                # bad = check_tensor_naninf(loss,'loss '+print_filename)
                # if bad: continue
                loss = loss.mean()
                loss_bi += loss
            loss_bi = loss_bi / float(num_planes)
            losses_Q += loss_bi




        

        

        losses = {
            # "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            # "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
            "loss_plane_para_L1": plane_para_loss_L1_jit(src_plane_para, target_plane_para),
            "loss_plane_para_cos": plane_para_loss_cos_jit(src_plane_para, target_plane_para),
        }
        losses['loss_plane_para_depth'] = losses_Q / float(b)

        del src_masks
        del target_masks
        return losses

    def loss_nonplane_depth(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_nonplane_rgba" in outputs

        # print(outputs.keys()) # dict_keys(['pred_logits', 'pred_plane_para', 'pred_masks', 'pred_nonplane_rgba', 'aux_outputs'])
        # print(outputs['pred_plane_para'].shape) # b, 100, 3
        # print(outputs['pred_logits'].shape) # b, 100, 134
        # print(outputs['pred_masks'].shape) # b, 100, h/4, w/4
        # print(outputs['pred_nonplane_rgba'][0].shape) # b, 32, 4, h, w
        # print(outputs['pred_nonplane_rgba'][1].shape) # b, 32, 4, h/2, w/2
        # print(outputs['pred_nonplane_rgba'][2].shape) # b, 32, 4, h/4, w/4
        # print(outputs['pred_nonplane_rgba'][3].shape) # b, 32, 4, h/8, w/8

        B, S, _, H_im, W_im = outputs['pred_nonplane_rgba'][0].shape
        device_type = outputs['pred_nonplane_rgba'][0]
        use_deep_supervision = True
        start_disparity = 0.999
        end_disparity = 0.001
        disparity_src = torch.linspace(
            start_disparity, end_disparity, S, dtype=device_type.dtype,
            device=device_type.device
        ).unsqueeze(0).repeat(B, 1)
        upsample_list = \
            [nn.Identity(),
             nn.Upsample(size=(int(H_im / 2), int(W_im / 2))),
             nn.Upsample(size=(int(H_im / 4), int(W_im / 4))),
             nn.Upsample(size=(int(H_im / 8), int(W_im / 8)))]
        src_depth_map_gt  = [t["gt_depth_from_plane"] for t in targets]
        src_depth_map_gt = torch.stack(src_depth_map_gt).to(device_type)
        src_mask_nonplane_valid_gt  = [t["mask_nonplane_valid"] for t in targets]
        src_mask_nonplane_valid_gt = torch.stack(src_mask_nonplane_valid_gt).to(device_type)

        loss_total = 0
        loss_cnt = 0
        losses = {}
        for scale_i in range(4):
            if (not use_deep_supervision) and scale_i > 0: continue

            mpi_rgba = outputs['pred_nonplane_rgba'][scale_i]
            _, _, _, H, W = mpi_rgba.shape

            src_depth_map_gt_scaled = upsample_list[scale_i](src_depth_map_gt[:,None,...])
            src_mask_nonplane_valid_gt_scaled = upsample_list[scale_i](src_mask_nonplane_valid_gt.type(torch.float)[:,None,...]).type(torch.bool)

            
            K_640 = torch.tensor([[577.,   0., 320.], [  0., 577., 240.], [  0.,   0.,   1.]])[None,...].repeat(B,1,1)
            K_640_inv = torch.inverse(K_640)
            f = 577 * (640/W)
            K = torch.tensor([[f,   0., W/2], [  0., f, H/2], [  0.,   0.,   1.]])[None,...].repeat(B,1,1)
            K_inv = torch.inverse(K)

            xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
                mpi_rendering.HomographySample(H, W, device=device_type.device).meshgrid,
                disparity_src,
                K_inv
            )
            xyz_src_BS3HW_640 = mpi_rendering.get_src_xyz_from_plane_disparity(
                mpi_rendering.HomographySample(H, W, device=device_type.device).meshgrid2,
                disparity_src,
                K_640_inv
            )

            # print(xyz_src_BS3HW.shape) # 2, 32, 3, 256, 384
            # print((xyz_src_BS3HW[:, :, 2:] == xyz_src_BS3HW_640[:, :, 2:]).all()) # True

            mpi_plane_rgb_src_1P3HW = mpi_rgba[:, :, 0:3, :, :]  # BxSx3xHxW
            mpi_plane_alpha_src_1P1HW = mpi_rgba[:, :, 3:, :, :]  # BxSx1xHxW
            src_imgs_syn_1, src_depth_syn_1, blend_weights, weights = mpi_rendering.render(
                mpi_plane_rgb_src_1P3HW,
                mpi_plane_alpha_src_1P1HW,
                xyz_src_BS3HW,
                use_alpha=True,
            )
            # print(src_imgs_syn_1.shape, src_depth_syn_1.shape) # 2, 3, 256, 384; 2, 1, 256, 384
            # print(torch.min(src_depth_map_gt_scaled),torch.mean(src_depth_map_gt_scaled),torch.max(src_depth_map_gt_scaled))
            # print(torch.min(src_depth_syn_1),        torch.mean(src_depth_syn_1),        torch.max(src_depth_syn_1))

            show_depth = False
            if show_depth:
                show_P4HW = torch.cat([mpi_plane_rgb_src_1P3HW,mpi_plane_alpha_src_1P1HW],dim=2)[0]
                P,_,H,W = show_P4HW.shape
                for p in range(P):
                    tmp_im = show_P4HW[p]
                    tmp_im = (tmp_im*255.).type(torch.uint8).cpu().numpy()
                    print(tmp_im.shape)
                    tmp_im = tmp_im.transpose(1,2,0)
                    print(tmp_im.shape)
                    import cv2
                    cv2.imwrite('./tmp/%d.png'%p, tmp_im)
                

            
            mask_type = 'nonplane'
            if mask_type == 'global':
                mask = src_depth_map_gt_scaled > 1e-3
            elif mask_type == 'nonplane':
                mask = src_mask_nonplane_valid_gt_scaled

            if show_depth:
                # drawDepth(mask[0,0].float(),'./showmask.png')
                drawDepth(src_depth_map_gt_scaled[0,0],'./tmp/gt_depth.png')
                drawDepth(src_depth_syn_1[0,0].detach(),'./tmp/pred_depth.png')
                time.sleep(222)
            loss_total += depth_loss(src_depth_syn_1, src_depth_map_gt_scaled, mask=mask.to(torch.bool))
            loss_cnt += 1

        losses['loss_nonplane_depth'] = loss_total / float(loss_cnt)
        return losses

        h, w = targets[0]['masks'].shape[-2:]

        losses_Q = 0
        for bi in range(outputs['pred_logits'].shape[0]): # for every image
            gt_depth_map = targets[bi]['gt_depth_from_plane']

            # drawDepth(gt_depth_map, './zmf_show/depth1.png')

            mask_cls = outputs['pred_logits'][bi]
            scores, labels = F.softmax(mask_cls, dim=-1).max(-1)

            mask_pred = outputs['pred_masks'][bi]
            mask_pred = mask_pred.sigmoid() # to 0~1
            keep = labels.ne(133) & (scores > 0.8)
            # print(labels,scores)
            # print(keep)

            pred_plane_para = outputs['pred_plane_para'][bi]

            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            pred_plane_para_keep = pred_plane_para[keep]

            # print(cur_masks.shape)
            if cur_masks[None,...].shape[1] == 0:
                continue
            cur_masks = F.interpolate(
                cur_masks[None,...],
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )[0]
            # print(cur_masks.shape)

            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
            cur_mask_ids = cur_prob_masks.argmax(0)
            K_inv_dot_xy_1 = targets[bi]['K_inv_dot_xy_1']
            # print(torch.unique(cur_mask_ids))

            # tmp_mask = gt_depth_map.ne(0).float()
            # drawDepth(tmp_mask,'./zmf_show/mask2.png')

            loss_bi = 0
            for k in range(cur_classes.shape[0]): # for every plane
                param = pred_plane_para_keep[k].view(1, 3)
                mask = gt_depth_map.ne(0) & (cur_mask_ids == k)
                mask_area = mask.sum().item()
                if mask_area == 0:
                    continue

                depth = gt_depth_map[None,...]  # 1, h, w
                
                k_inv_dot_xy1_map = K_inv_dot_xy_1.clone().view(3, h, w).to(depth)
                gt_pts_map = k_inv_dot_xy1_map * depth  # 3, h, w

                
                # check_tensor_naninf(gt_pts_map,'gt_pts_map')
                # check_tensor_naninf(mask,'mask')

                pts = torch.masked_select(gt_pts_map, mask).view(3, -1)  # 3, plane_pt_num

                # check_tensor_naninf(param,'param')
                # check_tensor_naninf(pts,'pts')

                loss = torch.abs(torch.matmul(param.to(pts), pts) - 1)  # 1, plane_pt_num

                loss = loss.mean()
                # check_tensor_naninf(loss,'loss') ###################
                loss_bi += loss
            plane_num = cur_classes.shape[0]
            # print('zmfzmf',plane_num)
            # check_tensor_naninf(loss_bi,'loss_bi')
            loss_bi = loss_bi / float(plane_num)
            losses_Q += loss_bi

        b = outputs['pred_logits'].shape[0]
        # check_tensor_naninf(losses_Q,'losses_Q')
        # print('zmfzmfzmf',b)
        losses = {
            "loss_nonplane_depth": losses_Q / float(b),
        }

        return losses

    def loss_tgt_view(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        # print(src_masks.shape, target_masks.shape) # torch.Size([18, 1, 64, 96]) torch.Size([18, 1, 256, 384])



        # print(outputs.keys()) # dict_keys(['pred_logits', 'pred_plane_para', 'pred_masks', 'pred_nonplane_rgba', 'pred_plane_rgba', 'aux_outputs'])
        # for t in targets: print(t.keys()); exit() # dict_keys(['labels', 'masks', 'gt_plane_para', 'gt_depth_map', 'K_inv_dot_xy_1', 'gt_pixel_plane_para', 'gt_depth_from_plane', 'filename', 'mask_nonplane_valid', 'gt_tgt_view_filename', 'gt_tgt_view_rgb', 'gt_tgt_view_depth_map', 'gt_G_src_tgt', 'gt_src_view_image'])
        
        # ZMF START

        device_type = outputs["pred_masks"].float()
        B, _, _, _ = outputs["pred_masks"].shape




        losses_rgb_l1 = 0.
        losses_rgb_ssim = 0.
        losses_depth = 0.

        deep_super = True
        if deep_super: scales = [0,1,2,3]
        else: scales = [0]

        loss_cnt = 0

        for bi in range(B):
            P = outputs["pred_masks"][bi][indices[bi][0]].shape[0]
            
            im_idx = 12345
            if torch.randint(0,20,[1])[0] == 1 and bi == 0 and False:
                save_images = True
                print('save')
            else:
                save_images = False
            
            
            for scale in scales:
                # prepare orig data
                pred_src_plane_masks_Phw    = outputs["pred_masks"][bi][indices[bi][0]].to(device_type)
                pred_src_plane_RGBA_P4HW    = outputs["pred_plane_rgba"][scale][bi][indices[bi][0]].sigmoid().to(device_type)
                pred_src_plane_para_P3      = outputs['pred_plane_para'][bi][indices[bi][0]].to(device_type)
                pred_src_nonplane_RGBA_S4HW = outputs["pred_nonplane_rgba"][scale][bi].to(device_type)

                gt_tgt_view_rgb_3hw         = (targets[bi]['gt_tgt_view_rgb']/255.).to(device_type)
                gt_tgt_view_depth_map_hw    = targets[bi]['gt_tgt_view_depth_map'].to(device_type)
                gt_G_src_tgt_44             = targets[bi]['gt_G_src_tgt'].to(device_type)
                K_inv_dot_xy_1_3HxW         = targets[bi]['K_inv_dot_xy_1'].view(3,-1).to(device_type)
                K_inv_dot_xy_1_3HxW_2         = targets[bi]['K_inv_dot_xy_1_2'].view(3,-1).to(device_type)
                K_inv_dot_xy_1_3HxW_4         = targets[bi]['K_inv_dot_xy_1_4'].view(3,-1).to(device_type)
                K_inv_dot_xy_1_3HxW_8         = targets[bi]['K_inv_dot_xy_1_8'].view(3,-1).to(device_type)
                K_inv_dot_xy_1 = [K_inv_dot_xy_1_3HxW,K_inv_dot_xy_1_3HxW_2,K_inv_dot_xy_1_3HxW_4,K_inv_dot_xy_1_3HxW_8]
                gt_src_view_rgb_3hw         = (targets[bi]['gt_src_view_image']/255.).to(device_type)

                _,_,H,W = pred_src_plane_RGBA_P4HW.shape
                gt_src_view_rgb_3HW = F.interpolate(gt_src_view_rgb_3hw[:,None,...], size=(H,W), mode="bilinear")[:,0]
                gt_tgt_view_rgb_3HW = F.interpolate(gt_tgt_view_rgb_3hw[:,None,...], size=(H,W), mode="bilinear")[:,0]
                gt_tgt_view_depth_map_HW = F.interpolate(gt_tgt_view_depth_map_hw[None,None,...], size=(H,W), mode="bilinear")[0,0]


                results, vis_results = mpi_rendering.render_everything(pred_src_plane_masks_Phw, pred_src_plane_RGBA_P4HW, pred_src_plane_para_P3, 
                                                                        pred_src_nonplane_RGBA_S4HW, gt_G_src_tgt_44, gt_src_view_rgb_3HW, K_inv_dot_xy_1,
                                                                        device_type, save_images, scale)
                
                # 0 target view
                if save_images:
                    from detectron2.utils.events import get_event_storage
                    P,_, _, _ = pred_src_plane_RGBA_P4HW.shape
                    S, _, H, W = pred_src_nonplane_RGBA_S4HW.shape

                    for i in range(P):
                        this_plane_RGBA_4HW = vis_results['pred_src_plane_RGBA_P4HW'][i]
                        see_this_plane_pred_RGBA_4HW = (this_plane_RGBA_4HW*255).type(torch.uint8) #.cpu().numpy() # [:,:,[2,1,0,3]]
                        try:
                            storage = get_event_storage()
                            storage.put_image("{}/plane_{}_{}".format(im_idx,i,scale), see_this_plane_pred_RGBA_4HW)
                        except:
                            self.writer.add_image("{}/plane_{}_{}".format(im_idx,i,scale), see_this_plane_pred_RGBA_4HW, 0)
                            
                    
                    for i in range(S):
                        this_plane_RGBA_4HW = vis_results['pred_src_nonplane_RGB_1S4HW'][0,i]
                        see_this_plane_debug_RGBA_4HW = (this_plane_RGBA_4HW*255).type(torch.uint8) #.cpu().numpy() # [:,:,[2,1,0,3]]
                        try:
                            storage.put_image("{}/nonplane_{}_{}".format(im_idx,i,scale), see_this_plane_debug_RGBA_4HW)
                        except:
                            self.writer.add_image("{}/nonplane_{}_{}".format(im_idx,i,scale), see_this_plane_debug_RGBA_4HW, 0)
                    
                    
                    # pred_tgt_depth_HW = vis_results['pred_tgt_depth_HW']
                    # pred_tgt_depth_HW3 = self.drawDepth(pred_tgt_depth_HW.cpu().numpy())
                    # see_pred_tgt_depth_3HW = cv2.resize(pred_tgt_depth_HW3,(int(300/H*W),300)).transpose(2, 0, 1)
                    # try:
                    #     storage.put_image("results/{}_tgt_depth".format(im_idx), see_pred_tgt_depth_3HW)
                    # except:
                    #     self.writer.add_image("results/{}_tgt_depth".format(im_idx), see_pred_tgt_depth_3HW, 0)

                    # gt_tgt_view_depth_map_HW3 = self.drawDepth(gt_tgt_view_depth_map_HW.cpu().numpy())
                    # see_gt_tgt_view_depth_map_HW3 = cv2.resize(gt_tgt_view_depth_map_HW3,(int(300/H*W),300)).transpose(2, 0, 1)
                    # try:
                    #     storage.put_image("results/{}_tgt_depth_gt".format(im_idx), see_gt_tgt_view_depth_map_HW3)
                    # except:
                    #     self.writer.add_image("results/{}_tgt_depth_gt".format(im_idx), see_gt_tgt_view_depth_map_HW3, 0)

                    
                    pred_tgt_RGB_3HW = vis_results['pred_tgt_RGB_3HW']
                    see_pred_tgt_RGB_3HW = (pred_tgt_RGB_3HW*255).type(torch.uint8) #.cpu().numpy() #[:,:,[2,1,0]]
                    see_gt = (gt_tgt_view_rgb_3HW*255.).type(torch.uint8) #.cpu().numpy() #[:,:,[2,1,0]]
                    try:
                        storage.put_image("results/{}_tgt_im_pred_{}".format(im_idx,scale), see_pred_tgt_RGB_3HW)
                        storage.put_image("results/{}_tgt_im_gt_{}".format(im_idx,scale), see_gt)
                    except:
                        self.writer.add_image("results/{}_tgt_im_pred_{}".format(im_idx,scale), see_pred_tgt_RGB_3HW, 0)
                        self.writer.add_image("results/{}_tgt_im_gt_{}".format(im_idx,scale), see_gt, 0)
            
                if False:
                    if save_images:
                        pred_src_plane_RGBA_P4HW[:,:3,:,:] = gt_src_view_rgb_3hw[None,...].repeat(P,1,1,1)
                        pred_src_nonplane_RGBA_S4HW[:,:3,:,:] = gt_src_view_rgb_3hw[None,...].repeat(S,1,1,1)

                        for i in range(P):
                            # see alpha
                            # drawDepth(gt_masks_PHW[i].detach(),                   './zmf_debug/b%d_p%d_seemask_gt.png'%(bi,i))
                            drawDepth(pred_src_plane_RGBA_P4HW[i,3,...].detach(), './zmf_debug/b%d_p%d_seemask_alpha_init.png'%(bi,i))

                            # see rgb
                            import cv2
                            import numpy as np
                            see_src_rgb = (gt_src_view_rgb_3hw*255).permute(1,2,0).cpu().numpy().astype(np.uint8)[:,:,[2,1,0]]
                            cv2.imwrite('./zmf_debug/b%d_seergb_src_gt.png'%bi,see_src_rgb)
                            this_plane_RGBA_4HW = pred_src_plane_RGBA_P4HW[i]
                            see_this_plane_pred_RGBA_4HW = (this_plane_RGBA_4HW*255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)[:,:,[2,1,0,3]]
                            cv2.imwrite('./zmf_debug/b%d_p%d_seergb_src_pred_init.png'%(bi,i),see_this_plane_pred_RGBA_4HW)
                            
                    if save_images:
                        for i in range(S):
                            # see alpha
                            drawDepth(pred_all_plane.detach(),                       './zmf_debug/b%d_seenon_allplanes.png'%(bi))
                            drawDepth(pred_src_nonplane_A_1S1HW[0,i,0,...].detach(), './zmf_debug/b%d_s%d_seenon_alpha_init.png'%(bi,i))

                            # see rgb
                            import cv2
                            import numpy as np
                            debug_this_plane_RGBA_4HW = torch.cat([pred_src_nonplane_RGB_1S3HW,pred_src_nonplane_A_1S1HW],dim=2)[0,i,]
                            see_this_plane_debug_RGBA_4HW = (debug_this_plane_RGBA_4HW*255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)[:,:,[2,1,0,3]]
                            cv2.imwrite('./zmf_debug/b%d_s%d_seenon_src_debug.png'%(bi,i),see_this_plane_debug_RGBA_4HW)
                    
                    if save_images: 
                        drawDepth(pred_tgt_depth_HW.detach(), './zmf_debug/b%d_tgt_depth_syn.png'%bi)
                        drawDepth(gt_tgt_view_depth_map_HW, './zmf_debug/b%d_tgt_depth_gt.png'%bi)
                    
                    if save_images:
                        see_pred_tgt_RGB_3HW = (pred_tgt_RGB_3HW.permute(1,2,0)*255).type(torch.uint8).detach().cpu().numpy()[:,:,[2,1,0]]
                        import cv2
                        cv2.imwrite('./zmf_debug/b%d_tgt_rgb_syn.png'%bi, see_pred_tgt_RGB_3HW)
                        see_gt = (gt_tgt_view_rgb_3HW.permute(1,2,0)*255.).type(torch.uint8).detach().cpu().numpy()[:,:,[2,1,0]]
                        cv2.imwrite('./zmf_debug/b%d_tgt_rgb_gt.png'%bi,see_gt)


                # pred_tgt_RGB_3HW = results['pred_tgt_RGB_3HW']
                # pred_tgt_vmask_HW = results['pred_tgt_vmask_HW']
                # # RGB L1 loss
                # loss_map = torch.abs(pred_tgt_RGB_3HW - gt_tgt_view_rgb_3HW) * pred_tgt_vmask_HW[None,...].repeat(3,1,1)
                # losses_rgb_l1 += loss_map.mean()

                pred_tgt_RGB_3HW = results['pred_tgt_RGB_3HW']
                pred_tgt_vmask_1HW = results['pred_tgt_vmask_HW'][None,...].to(device_type)
                # RGB L1 loss
                loss_map = torch.abs(pred_tgt_RGB_3HW - gt_tgt_view_rgb_3HW) * pred_tgt_vmask_1HW
                losses_rgb_l1 += loss_map.mean()

                # RGB SSIM loss
                # losses_rgb_ssim2 += (1 - self.ssim2(pred_tgt_RGB_3HW[None,...], gt_tgt_view_rgb_3HW[None,...]))

                pred_tgt_RGB_3HW = torch.clamp(pred_tgt_RGB_3HW, 0., 1.)
                gt_tgt_view_rgb_3HW = torch.clamp(gt_tgt_view_rgb_3HW, 0., 1.)

                losses_rgb_ssim += (1 - self.ssim(pred_tgt_RGB_3HW[None,...], gt_tgt_view_rgb_3HW[None,...]))

                loss_cnt += 1

                # print((pred_tgt_RGB_3HW*pred_tgt_vmask_1HW).requires_grad,(gt_tgt_view_rgb_3HW*pred_tgt_vmask_1HW).requires_grad,losses_rgb_l1.requires_grad,pred_tgt_vmask_1HW.requires_grad)
                # print(pred_tgt_RGB_3HW.requires_grad,gt_tgt_view_rgb_3HW.requires_grad,losses_rgb_ssim.requires_grad)

                # TGT depth loss
                # depth_mask_HW = gt_tgt_view_depth_map_HW > 1e-3
                # depth_mask_HW = torch.logical_and(depth_mask_HW, pred_tgt_vmask_HW)
                # if save_images: drawDepth(depth_mask_HW.detach().float(), './zmf_debug/b%d_tgt_depthmap.png'%bi)
                # losses_depth += depth_loss(pred_tgt_depth_HW, gt_tgt_view_depth_map_HW, mask=depth_mask_HW)
        losses = {
            "loss_tgt_view_rgb": losses_rgb_l1 / loss_cnt,
            "loss_tgt_view_ssim": losses_rgb_ssim / loss_cnt,
            # "loss_tgt_view_depth": losses_depth / loss_cnt,
        }
        return losses 




        


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'plane_para': self.loss_plane_para,
            'nonplane_depth': self.loss_nonplane_depth,
            'tgt_view': self.loss_tgt_view,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))
        


        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss in ['plane_para','nonplane_depth','tgt_view']:
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


    
