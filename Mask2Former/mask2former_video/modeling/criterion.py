# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch,time,os
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from mask2former.utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

import mpi_rendering
from ssim import SSIM2


def drawDepthImage(depth, maxDepth=10):
    import numpy as np
    import cv2
    depthImage = np.clip(depth / maxDepth * 255, 0, 255).astype(np.uint8)
    depthImage = cv2.applyColorMap(255 - depthImage, colormap=cv2.COLORMAP_JET)
    return depthImage

def drawDepth(depth):
    # assert depth.dim() == 3 or depth.dim() == 2
    # if depth.dim() == 3:
    #     depth = depth[0].cpu().numpy()
    # else:
    #     depth = depth.cpu().numpy()

    depth_color = drawDepthImage(depth)
    depth_mask = depth > 1e-4
    depth_mask = depth_mask[:, :, None]
    depth_color = depth_color * depth_mask
    # cv2.imwrite(addr, depth_color)
    depth_color[:, :, [2, 0]] = depth_color[:, :, [0, 2]]
    return depth_color

def drawDepthSave(depth,addr):
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
    # print('dice input num_masks',inputs.shape, num_masks)
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

    # print('ce input target nummask',inputs.shape,targets.shape,num_masks)

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

def depth_loss(inputd, target, mask):

    inputd = torch.clamp(inputd, 1e-3, 100.)

    if mask is not None:
        inputd = inputd[mask]
        target = target[mask]
    g = torch.log(inputd) - torch.log(target)

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


class VideoSetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, HW):
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

        (self.cfg_h,self.cfg_w) = HW

        self.K_inv_dot_xy_1,self.K_inv_dot_xy_1_2,self.K_inv_dot_xy_1_4,self.K_inv_dot_xy_1_8 = None, None, None, None

        if os.environ['TRAIN_PHASE'] == '2':
            self.ssim = SSIM2(size_average=True).cuda()
        
    def precompute_K_inv_dot_xy_1(self, h, w, device):
        focal_length = 517.97
        offset_x = 320
        offset_y = 240

        K = [[focal_length, 0, offset_x],
                [0, focal_length, offset_y],
                [0, 0, 1]]

        K_inv = torch.linalg.inv(torch.tensor(K)).to(device=device)

        # full
        K_inv_dot_xy_1 = torch.zeros((3, h, w),device=device)
        for y in range(h):
            for x in range(w):
                yy = float(y) / h * 480
                xx = float(x) / w * 640

                ray = torch.mm(K_inv, torch.tensor([xx, yy, 1],device=device).reshape(3, 1))
                K_inv_dot_xy_1[:, y, x] = ray[:, 0]

        # half
        h = int(h/2)
        w = int(w/2)
        K_inv_dot_xy_1_2 = torch.zeros((3, h, w),device=device)
        # xy_map = torch.zeros((2, h, w))
        for y in range(h):
            for x in range(w):
                yy = float(y) / h * 480
                xx = float(x) / w * 640

                ray = torch.mm(K_inv, torch.tensor([xx, yy, 1],device=device).reshape(3, 1))
                K_inv_dot_xy_1_2[:, y, x] = ray[:, 0]
                # xy_map[0, y, x] = float(x) / w
                # xy_map[1, y, x] = float(y) / h

        # 1/4
        h = int(h/2)
        w = int(w/2)
        K_inv_dot_xy_1_4 = torch.zeros((3, h, w),device=device)
        # xy_map = torch.zeros((2, h, w))
        for y in range(h):
            for x in range(w):
                yy = float(y) / h * 480
                xx = float(x) / w * 640

                ray = torch.mm(K_inv, torch.tensor([xx, yy, 1],device=device).reshape(3, 1))
                K_inv_dot_xy_1_4[:, y, x] = ray[:, 0]
                # xy_map[0, y, x] = float(x) / w
                # xy_map[1, y, x] = float(y) / h

        # 1/8
        h = int(h/2)
        w = int(w/2)
        K_inv_dot_xy_1_8 = torch.zeros((3, h, w),device=device)
        # xy_map = torch.zeros((2, h, w))
        for y in range(h):
            for x in range(w):
                yy = float(y) / h * 480
                xx = float(x) / w * 640

                ray = torch.mm(K_inv, torch.tensor([xx, yy, 1],device=device).reshape(3, 1))
                K_inv_dot_xy_1_8[:, y, x] = ray[:, 0]
                # xy_map[0, y, x] = float(x) / w
                # xy_map[1, y, x] = float(y) / h


        return K_inv_dot_xy_1, K_inv_dot_xy_1_2, K_inv_dot_xy_1_4, K_inv_dot_xy_1_8

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs

        if not os.environ['SINGLE_MATCH'] == 'True':
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
        else:
            src_logits = outputs["pred_logits"].float() # B,Q,134
            target_classes = torch.full(
                src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
            ) # B,Q

            # import random
            # r = int(random.random()*1000)

            for fi in range(int(os.environ['SAMPLING_FRAME_NUM'])): 
                # video_maskformer_model.py prepare_targets only prepare labels for the final (2nd) frame.
                # this is reasonalbe because when there are different instances in 1&2 frame,
                # this loss function will assign different labels (plane & background) to the same query.
                # this is still not reasonable because it assigns a wrong label to queries finding instances only appearing in the 1st frame
                # However, original Mask2Former-VIS throws away unmatched instances and assign wrong labels to queries finding them, too.

                idx = self._get_src_permutation_idx(indices[fi])
                target_classes_o = torch.cat([t["labels"][:,fi,0][J] for t, (_, J) in zip(targets, indices[fi])])
                target_classes[idx] = target_classes_o # B,Q

            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
            losses = {"loss_ce": loss_ce}
            return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        alpha_loss = (os.environ['TRAIN_PHASE'] == '1') and ('is_aux' not in outputs)

        if not os.environ['SINGLE_MATCH'] == 'True':
            src_idx = self._get_src_permutation_idx(indices)
            src_masks = outputs["pred_masks"] # B, Q, T, H, W
            src_masks = src_masks[src_idx] # P, T, H, W
            
            # Modified to handle video
            target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)]).to(src_masks)

            # print('los',src_masks.shape,target_masks.shape) # P, T, H, W



                    

            # No need to upsample predictions as we are using normalized coordinates :)
            # NT x 1 x H x W
            src_masks = src_masks.flatten(0, 1)[:, None]
            target_masks = target_masks.flatten(0, 1)[:, None]
            # print('los2',src_masks.shape,target_masks.shape) # P*T, 1, H, W
        else:
            s, t = [], []
            if alpha_loss:
                sa = []
            for i in range(len(indices)):
                src_idx = self._get_src_permutation_idx(indices[i])
                tgt_idx = self._get_tgt_permutation_idx(indices[i])

                src_masks = outputs["pred_masks"][:,:,i,:,:]
                src_masks = src_masks[src_idx]
                
                if alpha_loss:
                    src_alpha = outputs["pred_alpha"][:,:,i,:,:]
                    src_alpha = src_alpha[src_idx]

                masks = [t["masks"][:,i,:,:] for t in targets]
                # TODO use valid to mask invalid areas due to padding in loss
                target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
                target_masks = target_masks.to(src_masks)
                target_masks = target_masks[tgt_idx]

                first_train_see = False
                if first_train_see:
                    import cv2,random
                    if random.random() < 1: #0.001:
                        P,H,W = src_masks.shape
                        for p in range(P):
                            pred_mask = (src_masks[p].sigmoid() * 255).type(torch.uint8).detach().cpu().numpy()
                            gt_mask   = (target_masks[p] * 255).type(torch.uint8).detach().cpu().numpy()
                            cv2.imwrite('./tmp/frame_%d_plane_%d_pred.png'%(i,p), pred_mask)
                            cv2.imwrite('./tmp/frame_%d_plane_%d_gt.png'%(i,p), gt_mask)
                    if i == 1:
                        exit()
                

                # No need to upsample predictions as we are using normalized coordinates :)
                # N x 1 x H x W
                s.append(src_masks[:, None])
                t.append(target_masks[:, None])
                if alpha_loss:
                    sa.append(src_alpha[:, None])

            src_masks = torch.cat(s,dim=0)
            target_masks = torch.cat(t,dim=0)
            if alpha_loss:
                src_alphas = torch.cat(sa,dim=0)


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

        if alpha_loss:
            point_logits_alpha = point_sample(
                src_alphas,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        loss_mask = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks)
        loss_dice = dice_loss_jit(point_logits, point_labels, num_masks)

        if alpha_loss:
            loss_mask = loss_mask + sigmoid_ce_loss_jit(point_logits_alpha, point_labels, num_masks)
            loss_mask = loss_mask / 2
            loss_dice = loss_dice + dice_loss_jit(point_logits_alpha, point_labels, num_masks)
            loss_dice = loss_dice / 2

        losses = {
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
        }

        del src_masks
        del target_masks
        return losses

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

    def get_plane_params_in_global(self, planes, camera_info):
        """
        input:
        @planes: plane params
        @camera_info: plane params from camera info, type = dict, must contain 'position' and 'rotation' as keys
        output:
        plane parameters in global frame.
        """
        import quaternion
        tran = torch.FloatTensor(camera_info["position"]).to(planes)
        rot = quaternion.from_float_array(camera_info["rotation"])
        start = torch.ones((len(planes), 3)).to(planes) * tran
        end = planes * torch.tensor([1, -1, -1]).to(planes)  # suncg2habitat
        end = (
            torch.mm(
                torch.FloatTensor(quaternion.as_rotation_matrix(rot)).to(planes),
                (end).T,
            ).T
            + tran
        )  # cam2world
        a = end
        b = end - start
        planes_world = ((a * b).sum(dim=1) / (torch.norm(b, dim=1) + 1e-5) ** 2).view(-1, 1) * b
        return planes_world

    def get_plane_params_in_local(self, planes, camera_info):
        """
        input: 
        @planes: plane params
        @camera_info: plane params from camera info, type = dict, must contain 'position' and 'rotation' as keys
        output:
        plane parameters in global frame.
        """
        import quaternion
        import numpy as np
        tran = torch.FloatTensor(camera_info["position"]).to(planes)
        rot = quaternion.from_float_array(camera_info["rotation"]) # np.array(camera_info['rotation'])
        b = planes
        a = torch.ones((len(planes),3)).to(planes)*tran
        planes_world = a + b - ((a*b).sum(axis=1) / torch.linalg.norm(b, axis=1)**2).reshape(-1,1)*b
        end = (torch.tensor(quaternion.as_rotation_matrix(rot.inverse())).to(planes)@(planes_world - tran).T).T #world2cam
        planes_local = end*torch.tensor([1, -1, -1]).to(planes)# habitat2suncg
        return planes_local

    def loss_plane_para(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        device_type = outputs["pred_masks"]
        if self.K_inv_dot_xy_1 == None:
            self.K_inv_dot_xy_1,self.K_inv_dot_xy_1_2,self.K_inv_dot_xy_1_4,self.K_inv_dot_xy_1_8 = self.precompute_K_inv_dot_xy_1(h=self.cfg_h, w=self.cfg_w, device=device_type.device)
            print('should once')

        # print(outputs.keys()) # dict_keys(['pred_logits', 'pred_plane_para', 'pred_masks', 'aux_outputs'])
        # print(outputs['pred_plane_para'].shape) # B, 100, 3
        # print(outputs['pred_logits'].shape) # B, 100, 134
        # print(outputs['pred_masks'].shape) # B, 100, 256, 256
        # print(len(targets),targets[0]['gt_cam_pose'].shape,targets[0]['gt_plane_para'].shape) # B   2,4,4   P,2,4

        if len(indices) == 2:
            cam_pose0 = [t["gt_cam_pose"][0] for t in targets]
            cam_pose1 = [t["gt_cam_pose"][1] for t in targets]

            if os.environ['TRAIN_MP3D'] == 'True':
                plane_para_1 = []
                pred_plane_para0_3 = outputs['pred_plane_para'].to(device_type) # B Q 3
                for bi in range(len(cam_pose0)):
                    cam_info0 = cam_pose0[bi]
                    cam_info1 = cam_pose1[bi]




                    glob_plane_para0 = self.get_plane_params_in_global(pred_plane_para0_3[bi].to(device_type),cam_info0)
                    this_plane_para1 = self.get_plane_params_in_local(glob_plane_para0, cam_info1)
                    plane_para_1.append(this_plane_para1)

                pred_plane_para1_3 = torch.stack(plane_para_1)

                pred_plane_para = [pred_plane_para0_3,pred_plane_para1_3]
            else:
                cam_pose0 = torch.stack(cam_pose0).to(device_type) # B 4 4
                cam_pose1 = torch.stack(cam_pose1).to(device_type) # B 4 4
                G_src_tgt = cam_pose0 @ torch.linalg.inv(cam_pose1) # B 4 4
                
                verify = False
                if verify:
                    cam_pose0 = targets[0]['gt_cam_pose'][0].to(device_type) # 4 4
                    cam_pose1 = targets[0]['gt_cam_pose'][1].to(device_type) # 4 4
                    G_src_tgt = cam_pose0 @ torch.linalg.inv(cam_pose1)[None,...] # 1 4 4

                    gt_src_plane_para_1Q4 = targets[0]['gt_plane_para'][:,0,:][None,...]
                    gt_tgt_plane_para_1Q4 = targets[0]['gt_plane_para'][:,1,:][None,...]
                    tgt_plane_para_1Q4 = self.plane_para4_trans(gt_src_plane_para_1Q4, G_src_tgt)
                    print('src',gt_src_plane_para_1Q4)
                    print('gt tgt',gt_tgt_plane_para_1Q4)
                    print('cal tgt',tgt_plane_para_1Q4)
                    print('showing that plane_para4 R&t function is correct (not matching is ok)')
                    
                    gt_src_plane_para_1Q3 = self.plane_para_4to3(gt_src_plane_para_1Q4)
                    # The sign is uncertain when 3 -> 4
                    gt_src_plane_para_1Q4 = self.plane_para_3to4(gt_src_plane_para_1Q3)
                    gt_src_plane_para_1Q4_m = -gt_src_plane_para_1Q4

                    tgt_plane_para_1Q4 = self.plane_para4_trans(gt_src_plane_para_1Q4, G_src_tgt)
                    tgt_plane_para_1Q4_m = self.plane_para4_trans(gt_src_plane_para_1Q4_m, G_src_tgt)
                    print('1',tgt_plane_para_1Q4)
                    print('1_m',tgt_plane_para_1Q4_m)
                    print('showing that there is still only sign difference after R&t')
                    
                    tgt_plane_para_1Q3 = self.plane_para_4to3(tgt_plane_para_1Q4)
                    tgt_plane_para_1Q3_m = self.plane_para_4to3(tgt_plane_para_1Q4_m)
                    print('2',tgt_plane_para_1Q3)
                    print('2_m',tgt_plane_para_1Q3_m)
                    print('showing that they are the same again after converting to para3')
                    exit()


                pred_plane_para0_3 = outputs['pred_plane_para'].to(device_type) # B Q 3
                pred_plane_para0_4 = self.plane_para_3to4(pred_plane_para0_3) # B Q 4
                pred_plane_para1_4 = self.plane_para4_trans(pred_plane_para0_4,G_src_tgt) # B Q 4
                pred_plane_para1_3 = self.plane_para_4to3(pred_plane_para1_4) # B Q 3

                pred_plane_para = [pred_plane_para0_3,pred_plane_para1_3]
        else:
            if os.environ['TRAIN_MP3D'] == 'True':
                pred_plane_para0_3 = outputs['pred_plane_para'].to(device_type) # B Q 3
                pred_plane_para = [pred_plane_para0_3]
            else:
                cam_pose0 = [t["gt_cam_pose"][0] for t in targets]
                cam_pose0 = torch.stack(cam_pose0).to(device_type) # B 4 4
                pred_plane_para0_3 = outputs['pred_plane_para'].to(device_type) # B Q 3
                pred_plane_para = [pred_plane_para0_3]

        loss_l1, loss_l1_cnt = 0., 0
        loss_cos, loss_cos_cnt = 0., 0
        loss_depth, loss_depth_cnt = 0., 0
        
        for fi in range(len(indices)):
            # os.environ['TIME0401'] = '%.2f'%(time.process_time())
            src_idx = self._get_src_permutation_idx(indices[fi])
            tgt_idx = self._get_tgt_permutation_idx(indices[fi])
            # src_masks = outputs["pred_masks"][:,:,fi,:,:]
            pred_plane_para_P3 = pred_plane_para[fi][src_idx]

            second_frame_no_nonplane_loss = True

            gt_plane_para = [t["gt_plane_para"][:,fi,:,None] for t in targets]
            gt_nonplane_num = [t["nonplane_num"][fi] for t in targets]
            # print(0,fi,gt_nonplane_num)
            # print(1,fi,gt_plane_para)
            gt_plane_para, valid = nested_tensor_from_tensor_list(gt_plane_para).decompose()
            gt_plane_para = gt_plane_para.to(device_type)
            # print(2,fi,gt_plane_para.shape,gt_plane_para)
            gt_plane_para_P4 = gt_plane_para[tgt_idx][:,:,0]
            # print(3,fi,gt_plane_para_P4.shape,gt_plane_para_P4)
            if os.environ['TRAIN_MP3D'] == 'True':
                gt_plane_para_P3 = gt_plane_para_P4
            else:
                gt_plane_para_P3 = self.plane_para_4to3(gt_plane_para_P4)

            loss_l1 += plane_para_loss_L1_jit(pred_plane_para_P3, gt_plane_para_P3)
            loss_l1_cnt += 1
            loss_cos += plane_para_loss_cos_jit(pred_plane_para_P3, gt_plane_para_P3)
            loss_cos_cnt += 1

            # loss depth
            masks = [t["masks"][:,fi,:,:] for t in targets]
            target_masks_nest, valid = nested_tensor_from_tensor_list(masks).decompose()
            target_masks_nest = target_masks_nest.to(device_type)

            if os.environ['SEG_NONPLANE'] != 'True':
                target_depth_map  = [t["gt_depth_from_plane"][fi] for t in targets]
            else:
                target_depth_map  = [t["gt_depth_from_plane_and_nonplane"][fi] for t in targets]

            # filenames         = [t["filename"] for t in targets]
            # print_filename = ''
            # for filename in filenames:
            #     print_filename += filename
            #     print_filename += '  '

            target_depth_map = torch.stack(target_depth_map).to(device_type)
            # check_tensor_naninf(target_depth_map,'after '+print_filename)

            gt_depths = target_depth_map[:,None]  # b, 1, 480, 640 depth_from_plane
            b, _, h, w = gt_depths.shape

            assert b == len(targets)
            have_naninf = False
            
            for bi in range(b):
                # os.environ['TIME041'] = '%.2f'%(time.process_time())
                P = len(indices[fi][bi][0])
                segmentation = target_masks_nest[bi] # b, max_num, h, w

                depth = gt_depths[bi]  # 1, h, w
                # check_tensor_naninf(depth,'depth '+print_filename)
                # os.environ['TIME042'] = '%.2f'%(time.process_time()); print('4.1-4.2:',float(os.environ['TIME042'])-float(os.environ['TIME041']))
                gt_pts_map = self.K_inv_dot_xy_1 * depth  # 3, h, w
                # check_tensor_naninf(gt_pts_map,'gt_pts_map '+print_filename)
                # os.environ['TIME043'] = '%.2f'%(time.process_time()); print('4.2-4.3:',float(os.environ['TIME043'])-float(os.environ['TIME042']))

                indices_bi = indices[fi][bi]
                idx_out = indices_bi[0]
                idx_tgt = indices_bi[1]
                # assert idx_tgt.max() + 1 == P
                # os.environ['TIME044'] = '%.2f'%(time.process_time()); print('4.3-4.4:',float(os.environ['TIME044'])-float(os.environ['TIME043']))

                # select pixel with segmentation
                loss_bi = 0.
                
                for i in range(P):
                    gt_plane_idx = int(idx_tgt[i])
                    mask = segmentation[gt_plane_idx, :, :].view(1, h, w)
                    mask = mask > 0

                    pts = torch.masked_select(gt_pts_map, mask).view(3, -1)  # 3, plane_pt_num
                    # bad = check_tensor_naninf(pts,'pts '+print_filename)

                    pred_plane_idx = int(idx_out[i])

                    param = pred_plane_para[fi][bi][pred_plane_idx].view(1, 3)
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
                loss_bi = loss_bi / P
                loss_depth += loss_bi
                loss_depth_cnt += 1
                # os.environ['TIME0498'] = '%.2f'%(time.process_time()); print('4.3-4.98:',float(os.environ['TIME0498'])-float(os.environ['TIME043']))

            # os.environ['TIME0499'] = '%.2f'%(time.process_time()); print('4.01-4.99:',float(os.environ['TIME0499'])-float(os.environ['TIME0401']))
        losses = {
            # "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            # "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
            # "loss_plane_para_L1": plane_para_loss_L1_jit(src_plane_para, target_plane_para),
            # "loss_plane_para_cos": plane_para_loss_cos_jit(src_plane_para, target_plane_para),
        }
        losses['loss_plane_para_L1'] = loss_l1 / loss_l1_cnt
        losses['loss_plane_para_cos'] = loss_cos / loss_cos_cnt
        losses['loss_plane_para_depth'] = loss_depth / loss_depth_cnt
        # del src_masks
        # del target_masks
        return losses

    def loss_tgt_view(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        assert "pred_alpha" in outputs
        assert "pred_rgb" in outputs

        device_type = outputs["pred_masks"]

        # print(outputs.keys()) # dict_keys(['pred_logits', 'pred_plane_para', 'pred_masks', 'aux_outputs'])
        # print(outputs['pred_plane_para'].shape) # B, 100, 3
        # print(outputs['pred_logits'].shape) # B, 100, 134
        # print(outputs['pred_masks'].shape) # B, 100, 256, 256
        # print(len(targets),targets[0]['gt_cam_pose'].shape,targets[0]['gt_plane_para'].shape) # B   2,4,4   P,2,4

        if len(indices) == 2:
            cam_pose0 = [t["gt_cam_pose"][0] for t in targets]
            cam_pose1 = [t["gt_cam_pose"][1] for t in targets]
            cam_pose0 = torch.stack(cam_pose0).to(device_type) # B 4 4
            cam_pose1 = torch.stack(cam_pose1).to(device_type) # B 4 4
            G_src_tgt = cam_pose0 @ torch.linalg.inv(cam_pose1) # B 4 4
            
            verify = False
            if verify:
                cam_pose0 = targets[0]['gt_cam_pose'][0].to(device_type) # 4 4
                cam_pose1 = targets[0]['gt_cam_pose'][1].to(device_type) # 4 4
                G_src_tgt = cam_pose0 @ torch.linalg.inv(cam_pose1)[None,...] # 1 4 4

                gt_src_plane_para_1Q4 = targets[0]['gt_plane_para'][:,0,:][None,...]
                gt_tgt_plane_para_1Q4 = targets[0]['gt_plane_para'][:,1,:][None,...]
                tgt_plane_para_1Q4 = self.plane_para4_trans(gt_src_plane_para_1Q4, G_src_tgt)
                print('src',gt_src_plane_para_1Q4)
                print('gt tgt',gt_tgt_plane_para_1Q4)
                print('cal tgt',tgt_plane_para_1Q4)
                print('showing that plane_para4 R&t function is correct (not matching is ok)')
                
                gt_src_plane_para_1Q3 = self.plane_para_4to3(gt_src_plane_para_1Q4)
                # The sign is uncertain when 3 -> 4
                gt_src_plane_para_1Q4 = self.plane_para_3to4(gt_src_plane_para_1Q3)
                gt_src_plane_para_1Q4_m = -gt_src_plane_para_1Q4

                tgt_plane_para_1Q4 = self.plane_para4_trans(gt_src_plane_para_1Q4, G_src_tgt)
                tgt_plane_para_1Q4_m = self.plane_para4_trans(gt_src_plane_para_1Q4_m, G_src_tgt)
                print('1',tgt_plane_para_1Q4)
                print('1_m',tgt_plane_para_1Q4_m)
                print('showing that there is still only sign difference after R&t')
                
                tgt_plane_para_1Q3 = self.plane_para_4to3(tgt_plane_para_1Q4)
                tgt_plane_para_1Q3_m = self.plane_para_4to3(tgt_plane_para_1Q4_m)
                print('2',tgt_plane_para_1Q3)
                print('2_m',tgt_plane_para_1Q3_m)
                print('showing that they are the same again after converting to para3')
                exit()

            pred_plane_para0_3 = outputs['pred_plane_para'].to(device_type) # B Q 3
            pred_plane_para0_4 = self.plane_para_3to4(pred_plane_para0_3) # B Q 4
            pred_plane_para1_4 = self.plane_para4_trans(pred_plane_para0_4,G_src_tgt) # B Q 4
            pred_plane_para1_3 = self.plane_para_4to3(pred_plane_para1_4) # B Q 3

            pred_plane_para = [pred_plane_para0_3,pred_plane_para1_3]
        else:
            cam_pose0 = [t["gt_cam_pose"][0] for t in targets]
            cam_pose0 = torch.stack(cam_pose0).to(device_type) # B 4 4
            pred_plane_para0_3 = outputs['pred_plane_para'].to(device_type) # B Q 3
            pred_plane_para = [pred_plane_para0_3]

        loss_l1, loss_l1_cnt = 0., 0
        loss_ssim, loss_ssim_cnt = 0., 0
        loss_depth, loss_depth_cnt = 0., 0
        loss_l1_merge, loss_l1_merge_cnt = 0., 0.
        loss_ssim_merge, loss_ssim_merge_cnt = 0., 0.
        if os.environ['SAMPLING_FRAME_NUM'] == '2':
            render_tgt_RGBs_f1 = []
            render_tgt_As_f1 = []
            render_tgt_Ds_f1 = []
            render_tgt_masks_f1 = []
            render_tgt_RGBs_f2 = []
            render_tgt_As_f2 = []
            render_tgt_Ds_f2 = []
            render_tgt_masks_f2 = []
            tgt_RGB_gt = []
        
        for fi in range(len(indices)):
            # os.environ['TIME0401'] = '%.2f'%(time.process_time())
            src_idx = self._get_src_permutation_idx(indices[fi])
            tgt_idx = self._get_tgt_permutation_idx(indices[fi])
            pred_plane_para_P3 = pred_plane_para[fi][src_idx]

            B, _, _, _, _ = outputs["pred_masks"].shape # B Q T h w
            # src_masks = outputs["pred_masks"][:,:,fi,:,:] 
            # src_masks = src_masks[src_idx]

            # masks = [t["masks"] for t in targets]
            # target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
            # target_masks = target_masks.to(src_masks)
            # target_masks = target_masks[tgt_idx]

            # No need to upsample predictions as we are using normalized coordinates :)
            # N x 1 x H x W
            # src_masks = src_masks[:, None] # torch.Size([56, 1, 64, 96])
            # target_masks = target_masks[:, None]
            
            for bi in range(B):
                P, _, _ = outputs["pred_masks"][bi][:,fi,:,:][indices[fi][bi][0]].shape
            
                im_idx = 12345
                if torch.randint(0,20,[1])[0] == 1 and bi == 0:
                    save_images = False
                    # print('save')
                else:
                    save_images = False

            
                # prepare orig data
                pred_class_P134             = outputs["pred_logits"][bi][indices[fi][bi][0]].to(device_type)
                _, pred_class_P             = F.softmax(pred_class_P134, dim=-1).max(-1)
                pred_src_plane_masks_Phw    = outputs["pred_masks"][bi][:,fi,:,:][indices[fi][bi][0]].to(device_type)
                pred_src_RGB_P3HW           = outputs["pred_rgb"][bi][:,fi,:,:,:][indices[fi][bi][0]].sigmoid().to(device_type)
                pred_src_alpha_P1HW         = outputs["pred_alpha"][bi][:,fi,None,:,:][indices[fi][bi][0]].sigmoid().to(device_type)
                pred_src_RGBA_P4HW          = torch.cat([pred_src_RGB_P3HW,pred_src_alpha_P1HW],dim=1)
                pred_src_plane_para_P3      = outputs['pred_plane_para'][bi][indices[fi][bi][0]].to(device_type)

                gt_src_view_rgb_3hw         = (targets[bi]['gt_src_image']/255.).to(device_type)[fi]
                gt_src_view_depth_map_hw    = targets[bi]['gt_depth_from_plane'].to(device_type)[fi]
                gt_tgt_view_rgb_3hw         = (targets[bi]['gt_tgt_image']/255.).to(device_type)[0]
                gt_tgt_view_depth_map_hw    = targets[bi]['gt_tgt_depth'].to(device_type)[0]
                gt_G_src_tgt_44             = targets[bi]['gt_tgt_G_src_tgt'][fi].to(device_type)
                K_inv_dot_xy_1_3HxW         = self.K_inv_dot_xy_1.view(3,-1)
                K_inv_dot_xy_1              = [K_inv_dot_xy_1_3HxW]

                # K_inv_dot_xy_1_3HxW         = targets[bi]['K_inv_dot_xy_1'].view(3,-1).to(device_type)
                # K_inv_dot_xy_1_3HxW_2         = targets[bi]['K_inv_dot_xy_1_2'].view(3,-1).to(device_type)
                # K_inv_dot_xy_1_3HxW_4         = targets[bi]['K_inv_dot_xy_1_4'].view(3,-1).to(device_type)
                # K_inv_dot_xy_1_3HxW_8         = targets[bi]['K_inv_dot_xy_1_8'].view(3,-1).to(device_type)
                # K_inv_dot_xy_1 = [K_inv_dot_xy_1_3HxW,K_inv_dot_xy_1_3HxW_2,K_inv_dot_xy_1_3HxW_4,K_inv_dot_xy_1_3HxW_8]
                
                _,_,H,W = pred_src_RGBA_P4HW.shape



                use_src_view_rgb = os.environ['USE_ORIG_RGB'] == 'True'
                results, vis_results = mpi_rendering.render_everything(pred_src_plane_masks_Phw, pred_src_RGBA_P4HW, pred_src_plane_para_P3, 
                                                                        None, gt_G_src_tgt_44, gt_src_view_rgb_3hw, K_inv_dot_xy_1,
                                                                        device_type, save_images, use_src_view_rgb=use_src_view_rgb, pred_class_P=pred_class_P)

                if os.environ['SAMPLING_FRAME_NUM'] == '2':
                    if fi == 0:
                        render_tgt_RGBs_f1.append(results['pred_tgt_RGB_3HW'])
                        render_tgt_Ds_f1.append(results['pred_tgt_depth_HW'])
                        render_tgt_As_f1.append(results['pred_tgt_alpha_acc_1HW'])
                        render_tgt_masks_f1.append(results['pred_tgt_vmask_HW'][None,...].to(device_type))
                    if fi == 1:
                        render_tgt_RGBs_f2.append(results['pred_tgt_RGB_3HW'])
                        render_tgt_Ds_f2.append(results['pred_tgt_depth_HW'])
                        render_tgt_As_f2.append(results['pred_tgt_alpha_acc_1HW'])
                        render_tgt_masks_f2.append(results['pred_tgt_vmask_HW'][None,...].to(device_type))
                        tgt_RGB_gt.append(gt_tgt_view_rgb_3hw)


                # print(results.keys(),vis_results.keys()) # dict_keys(['pred_tgt_vmask_HW', 'pred_src_RGB_3HW', 'pred_tgt_RGB_3HW', 'pred_src_depth_HW', 'pred_tgt_depth_HW']) dict_keys(['pred_src_plane_RGBA_P4HW'])
                
                # 0 target view
                if save_images:
                    from detectron2.utils.events import get_event_storage
                    P,_, H, W = pred_src_RGBA_P4HW.shape

                    # src planes
                    for pi in range(P):
                        this_plane_RGBA_4HW = vis_results['pred_src_plane_RGBA_P4HW'][pi]
                        see_this_plane_pred_RGBA_4HW = (this_plane_RGBA_4HW*255).type(torch.uint8) #.cpu().numpy() # [:,:,[2,1,0,3]]
                        try:
                            storage = get_event_storage()
                            storage.put_image("{}/plane_{}_{}".format(im_idx,pi,fi), see_this_plane_pred_RGBA_4HW)
                        except:
                            self.writer.add_image("{}/plane_{}_{}".format(im_idx,pi,fi), see_this_plane_pred_RGBA_4HW, 0)

                    # src depth
                    if True:
                        pred_src_depth_HW = results['pred_src_depth_HW']
                        pred_src_depth_HW3 = drawDepth(pred_src_depth_HW.detach().cpu().numpy())
                        see_pred_src_depth_3HW = pred_src_depth_HW3.transpose(2, 0, 1)
                        gt_src_view_depth_map_HW3 = drawDepth(gt_src_view_depth_map_hw.detach().cpu().numpy())
                        see_gt_src_view_depth_map_3HW = gt_src_view_depth_map_HW3.transpose(2, 0, 1)
                        try:
                            storage.put_image("results/{}_src_depth_{}".format(im_idx,fi), see_pred_src_depth_3HW)
                            storage.put_image("results/{}_src_depth_gt_{}".format(im_idx,fi), see_gt_src_view_depth_map_3HW)
                        except:
                            self.writer.add_image("results/{}_src_depth_{}".format(im_idx,fi), see_pred_src_depth_3HW, 0)
                            self.writer.add_image("results/{}_src_depth_gt_{}".format(im_idx,fi), see_gt_src_view_depth_map_3HW, 0)

                    # src rgb
                    if True:
                        pred_src_RGB_3HW = results['pred_src_RGB_3HW']
                        see_pred_src_RGB_3HW = (pred_src_RGB_3HW*255).type(torch.uint8) #.cpu().numpy() #[:,:,[2,1,0]]
                        see_gt = (gt_src_view_rgb_3hw*255.).type(torch.uint8) #.cpu().numpy() #[:,:,[2,1,0]]
                        try:
                            storage.put_image("results/{}_src_im_pred_{}".format(im_idx,fi), see_pred_src_RGB_3HW)
                            storage.put_image("results/{}_src_im_gt_{}".format(im_idx,fi), see_gt)
                        except:
                            self.writer.add_image("results/{}_src_im_pred_{}".format(im_idx,fi), see_pred_src_RGB_3HW, 0)
                            self.writer.add_image("results/{}_src_im_gt_{}".format(im_idx,fi), see_gt, 0)
            
                    # tgt depth
                    if True:
                        pred_tgt_depth_HW = results['pred_tgt_depth_HW']
                        pred_tgt_depth_HW3 = drawDepth(pred_tgt_depth_HW.detach().cpu().numpy())
                        see_pred_tgt_depth_3HW = pred_tgt_depth_HW3.transpose(2, 0, 1)
                        gt_tgt_view_depth_map_HW3 = drawDepth(gt_tgt_view_depth_map_hw.detach().cpu().numpy())
                        see_gt_tgt_view_depth_map_3HW = gt_tgt_view_depth_map_HW3.transpose(2, 0, 1)
                        try:
                            storage.put_image("results/{}_tgt_depth_{}".format(im_idx,fi), see_pred_tgt_depth_3HW)
                            storage.put_image("results/{}_tgt_depth_gt_{}".format(im_idx,fi), see_gt_tgt_view_depth_map_3HW)
                        except:
                            self.writer.add_image("results/{}_tgt_depth_{}".format(im_idx,fi), see_pred_tgt_depth_3HW, 0)
                            self.writer.add_image("results/{}_tgt_depth_gt_{}".format(im_idx,fi), see_gt_tgt_view_depth_map_3HW, 0)

                    # tgt rgb
                    if True:
                        pred_tgt_RGB_3HW = results['pred_tgt_RGB_3HW']
                        see_pred_tgt_RGB_3HW = (pred_tgt_RGB_3HW*255).type(torch.uint8) #.cpu().numpy() #[:,:,[2,1,0]]
                        see_gt = (gt_tgt_view_rgb_3hw*255.).type(torch.uint8) #.cpu().numpy() #[:,:,[2,1,0]]
                        try:
                            storage.put_image("results/{}_tgt_im_pred_{}".format(im_idx,fi), see_pred_tgt_RGB_3HW)
                            storage.put_image("results/{}_tgt_im_gt_{}".format(im_idx,fi), see_gt)
                        except:
                            self.writer.add_image("results/{}_tgt_im_pred_{}".format(im_idx,fi), see_pred_tgt_RGB_3HW, 0)
                            self.writer.add_image("results/{}_tgt_im_gt_{}".format(im_idx,fi), see_gt, 0)
            
                    # valid mask
                    if True:
                        pred_vmask_HW = results['pred_tgt_vmask_HW']
                        pred_vmask_HW3 = drawDepth(pred_vmask_HW.detach().cpu().numpy())
                        see_pred_vmask_3HW = pred_vmask_HW3.transpose(2, 0, 1)
                        try:
                            storage.put_image("results/{}_vmask_{}".format(im_idx,fi), see_pred_vmask_3HW)
                        except:
                            self.writer.add_image("results/{}_vmask_{}".format(im_idx,fi), see_pred_vmask_3HW, 0)

                debug_see = False
                pred_tgt_RGB_3HW = results['pred_tgt_RGB_3HW']
                pred_tgt_vmask_1HW = results['pred_tgt_vmask_HW'][None,...].to(device_type)
                # RGB L1 loss
                if debug_see:
                    import cv2
                    pred_tgt_RGB_HW3_see = (pred_tgt_RGB_3HW*255).type(torch.uint8).detach().cpu().numpy().transpose(1,2,0)
                    gt_tgt_view_rgb_hw3_see = (gt_tgt_view_rgb_3hw*255).type(torch.uint8).detach().cpu().numpy().transpose(1,2,0)
                    pred_tgt_vmask_HW_see = (pred_tgt_vmask_1HW[0]*255).type(torch.uint8).detach().cpu().numpy()
                    cv2.imwrite('./zmf_debug/1.png',pred_tgt_RGB_HW3_see)
                    cv2.imwrite('./zmf_debug/2.png',gt_tgt_view_rgb_hw3_see)
                    cv2.imwrite('./zmf_debug/3.png',pred_tgt_vmask_HW_see)
                loss_map = torch.abs(pred_tgt_RGB_3HW - gt_tgt_view_rgb_3hw) * pred_tgt_vmask_1HW
                loss_l1 += loss_map.mean()
                loss_l1_cnt += 1

                # RGB SSIM loss
                pred_tgt_RGB_3HW = pred_tgt_RGB_3HW * pred_tgt_vmask_1HW.repeat(3,1,1)
                gt_tgt_view_rgb_3hw = gt_tgt_view_rgb_3hw * pred_tgt_vmask_1HW.repeat(3,1,1)
                if debug_see:
                    import cv2
                    pred_tgt_RGB_HW3_see = (pred_tgt_RGB_3HW*255).type(torch.uint8).detach().cpu().numpy().transpose(1,2,0)
                    gt_tgt_view_rgb_hw3_see = (gt_tgt_view_rgb_3hw*255).type(torch.uint8).detach().cpu().numpy().transpose(1,2,0)
                    cv2.imwrite('./zmf_debug/4.png',pred_tgt_RGB_HW3_see)
                    cv2.imwrite('./zmf_debug/5.png',gt_tgt_view_rgb_hw3_see)
                pred_tgt_RGB_3HW = torch.clamp(pred_tgt_RGB_3HW, 0., 1.)
                gt_tgt_view_rgb_3hw = torch.clamp(gt_tgt_view_rgb_3hw, 0., 1.)
                assert(pred_tgt_RGB_3HW.dtype == torch.float32) and (gt_tgt_view_rgb_3hw.dtype == torch.float32)
                loss_ssim += (1 - self.ssim(pred_tgt_RGB_3HW[None,...], gt_tgt_view_rgb_3hw[None,...]))
                loss_ssim_cnt += 1

                # SRC depth loss
                pred_src_depth_HW = results['pred_src_depth_HW']
                depth_mask_HW = gt_src_view_depth_map_hw > 1e-3
                depth_mask2_HW = pred_src_depth_HW > 1e-3
                depth_mask_HW = torch.logical_and(depth_mask_HW, depth_mask2_HW)
                # depth_mask_HW = torch.logical_and(depth_mask_HW, pred_tgt_vmask_1HW[0])
                if debug_see:
                    import cv2
                    pred_src_depth_HW_see = drawDepth(pred_src_depth_HW.detach().cpu().numpy())
                    gt_src_view_depth_map_hw_see = drawDepth(gt_src_view_depth_map_hw.detach().cpu().numpy())
                    depth_mask_HW_see = (depth_mask_HW*255).type(torch.uint8).detach().cpu().numpy()
                    cv2.imwrite('./zmf_debug/6.png',pred_src_depth_HW_see)
                    cv2.imwrite('./zmf_debug/7.png',gt_src_view_depth_map_hw_see)
                    cv2.imwrite('./zmf_debug/8.png',depth_mask_HW_see)
                    exit()
                loss_depth += depth_loss(pred_src_depth_HW, gt_src_view_depth_map_hw, mask=depth_mask_HW)
                loss_depth_cnt += 1


                # # TGT depth loss
                # depth_mask_HW = gt_tgt_view_depth_map_hw > 1e-3
                # depth_mask_HW = torch.logical_and(depth_mask_HW, pred_tgt_vmask_1HW[0])
                # pred_tgt_depth_HW = results['pred_tgt_depth_HW']
                # loss_depth += depth_loss(pred_tgt_depth_HW, gt_tgt_view_depth_map_hw, mask=depth_mask_HW)
                # loss_depth_cnt += 1

        if os.environ['SAMPLING_FRAME_NUM'] == '2':
            assert len(render_tgt_RGBs_f1) == B
            
            for bi in range(B):
                render_tgt_RGBs = []
                render_tgt_As = []
                render_tgt_Ds = []
                render_tgt_masks = []

                render_tgt_RGBs.append(render_tgt_RGBs_f1[bi]) #.append(results['pred_tgt_RGB_3HW'])
                render_tgt_Ds.append(render_tgt_Ds_f1[bi]) #.append(results['pred_tgt_depth_HW'])
                render_tgt_As.append(render_tgt_As_f1[bi]) #.append(results['pred_tgt_alpha_acc_1HW'])
                render_tgt_masks.append(render_tgt_masks_f1[bi]) #.append(results['pred_tgt_vmask_HW'][None,...].to(device_type))

                render_tgt_RGBs.append(render_tgt_RGBs_f2[bi]) #.append(results['pred_tgt_RGB_3HW'])
                render_tgt_Ds.append(render_tgt_Ds_f2[bi]) #.append(results['pred_tgt_depth_HW'])
                render_tgt_As.append(render_tgt_As_f2[bi]) #.append(results['pred_tgt_alpha_acc_1HW'])
                render_tgt_masks.append(render_tgt_masks_f2[bi]) #.append(results['pred_tgt_vmask_HW'][None,...].to(device_type))

                debug_see = False
                if debug_see:
                    import cv2
                    cv2.imwrite('./zmf_debug/render_b%d_1.png'%bi,render_tgt_RGBs[0].permute(1,2,0).detach().cpu().numpy()*255)
                    cv2.imwrite('./zmf_debug/render_b%d_2.png'%bi,render_tgt_RGBs[1].permute(1,2,0).detach().cpu().numpy()*255)

                render_tgt_RGBs_T3HW = torch.stack(render_tgt_RGBs,dim=0)
                render_tgt_As_T1HW = torch.stack(render_tgt_As,dim=0)
                render_tgt_Ds_T1HW = torch.stack(render_tgt_Ds,dim=0)[:,None,:,:]

                T,_,H,W = render_tgt_RGBs_T3HW.shape
                out_frame = (render_tgt_RGBs[0] + render_tgt_RGBs[1]) / (1e-10+render_tgt_As_T1HW[0]+render_tgt_As_T1HW[1])
                gt_tgt_RGB_3HW = tgt_RGB_gt[bi]
                out_mask = torch.logical_or(render_tgt_masks[0],render_tgt_masks[1])


                if debug_see:
                    import cv2
                    cv2.imwrite('./zmf_debug/render_b%d_3.png'%bi,out_frame.permute(1,2,0).detach().cpu().numpy()*255)
                    cv2.imwrite('./zmf_debug/render_b%d_4.png'%bi,out_mask[0].float().detach().cpu().numpy()*255)
                    cv2.imwrite('./zmf_debug/render_b%d_5.png'%bi,gt_tgt_RGB_3HW.permute(1,2,0).detach().cpu().numpy()*255)

                loss_map = torch.abs(out_frame - gt_tgt_RGB_3HW) * out_mask
                loss_l1_merge += loss_map.mean()
                loss_l1_merge_cnt += 1

                # RGB SSIM loss
                out_frame = out_frame * out_mask.repeat(3,1,1)
                gt_tgt_RGB_3HW = gt_tgt_RGB_3HW * out_mask.repeat(3,1,1)
                out_frame = torch.clamp(out_frame, 0., 1.)
                gt_tgt_RGB_3HW = torch.clamp(gt_tgt_RGB_3HW, 0., 1.)
                assert(out_frame.dtype == torch.float32) and (gt_tgt_RGB_3HW.dtype == torch.float32)
                loss_ssim_merge += (1 - self.ssim(out_frame[None,...], gt_tgt_RGB_3HW[None,...]))
                loss_ssim_merge_cnt += 1
                if debug_see:
                    import cv2
                    pred_tgt_RGB_HW3_see = (out_frame*255).type(torch.uint8).detach().cpu().numpy().transpose(1,2,0)
                    gt_tgt_view_rgb_hw3_see = (gt_tgt_RGB_3HW*255).type(torch.uint8).detach().cpu().numpy().transpose(1,2,0)
                    cv2.imwrite('./zmf_debug/render_b%d_6.png'%bi,pred_tgt_RGB_HW3_see)
                    cv2.imwrite('./zmf_debug/render_b%d_7.png'%bi,gt_tgt_view_rgb_hw3_see)
                

        if os.environ['SAMPLING_FRAME_NUM'] == '2':
            losses = {
                "loss_tgt_view_rgb": loss_l1 / loss_l1_cnt,
                "loss_tgt_view_ssim": loss_ssim / loss_ssim_cnt,
                "loss_tgt_view_depth": loss_depth / loss_depth_cnt,
                "loss_tgt_view_rgb_merge": loss_l1_merge / loss_l1_merge_cnt,
                "loss_tgt_view_ssim_merge": loss_ssim_merge / loss_ssim_merge_cnt,
            }
        else:
            losses = {
                "loss_tgt_view_rgb": loss_l1 / loss_l1_cnt,
                "loss_tgt_view_ssim": loss_ssim / loss_ssim_cnt,
                "loss_tgt_view_depth": loss_depth / loss_depth_cnt,
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
        frame_num = outputs_without_aux['pred_masks'].shape[2] # torch.Size([10, 100, 1, 64, 96])
        if os.environ['SINGLE_MATCH'] == 'True':
            indices = []
            for fi in range(frame_num):
                out_1 = {}
                out_1['pred_logits'] = outputs_without_aux['pred_logits']
                out_1['pred_masks'] = outputs_without_aux['pred_masks'][:,:,fi,:,:]
                # print(targets[0]['labels'].shape) # 11
                # print(targets[0]['masks'].shape) # 11, 2, H, W

                tar_1 = []
                for i in range(len(targets)):
                    dic_1 = {}
                    dic_1['labels'] = targets[i]['labels'][:,fi,0]
                    dic_1['masks'] = targets[i]['masks'][:,fi,:,:]
                    tar_1.append(dic_1)


                indices1 = self.matcher(out_1, tar_1)
                indices.append(indices1)
        else:
            indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item() * frame_num

        

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            # time1 = time.process_time()
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))
            # time2 = time.process_time()
            # print('overall',loss,time2-time1)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):

                if os.environ['SINGLE_MATCH'] == 'True':
                    indices = []
                    for fi in range(frame_num):
                        out_1 = {}
                        out_1['pred_logits'] = aux_outputs['pred_logits']
                        out_1['pred_masks'] = aux_outputs['pred_masks'][:,:,fi,:,:]

                        tar_1 = []
                        for ii in range(len(targets)):
                            dic_1 = {}
                            dic_1['labels'] = targets[ii]['labels'][:,fi,0]
                            dic_1['masks'] = targets[ii]['masks'][:,fi,:,:]
                            tar_1.append(dic_1)

                        indices1 = self.matcher(out_1, tar_1)
                        indices.append(indices1)
                else:
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
