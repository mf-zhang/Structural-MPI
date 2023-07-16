# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import io
import itertools
import json
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
from typing import Optional
from PIL import Image
from tabulate import tabulate

from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.visualizer import ColorMode, Visualizer
import torch, cv2
from torch.utils.tensorboard import SummaryWriter
import mpi_rendering


from detectron2.utils.events import get_event_storage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

from .evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


class ScanNetPanopticEvaluator_evalonly(DatasetEvaluator):
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(self, dataset_name: str, output_dir: Optional[str] = None):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        }

        self._output_dir = output_dir
        if self._output_dir is not None:
            PathManager.mkdirs(self._output_dir)
        self.writer = SummaryWriter('./zmf_show')

    def reset(self):
        self._predictions = []

    def _convert_category_id(self, segment_info):
        isthing = segment_info.pop("isthing", None)
        if isthing is None:
            # the model produces panoptic category id directly. No more conversion needed
            return segment_info
        if isthing is True:
            segment_info["category_id"] = self._thing_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = self._stuff_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        return segment_info

    def get_plane_parameters(self, plane, segmentation):
        # plane = plane[:,:3] / plane[:,3:]

        # plane[:, 0] = -plane[:, 0]
        # plane[:, 1] = -plane[:, 1]

        # tmp = plane[:, 0].copy()
        # plane[:, 0] = plane[:, 1]
        # plane[:, 1] = tmp

        h, w = segmentation.shape[1:]

        plane_parameters2 = np.ones((3, h, w))
        for i in range(segmentation.shape[0]):
            plane_mask = segmentation[i]
            plane_mask = plane_mask.astype(np.float32)
            cur_plane_param_map = np.ones((3, h, w)) * plane[i, :].reshape(3, 1, 1)
            plane_parameters2 = plane_parameters2 * (1-plane_mask) + cur_plane_param_map * plane_mask

        # # plane_instance parameter, padding zero to fix size
        # plane_instance_parameter = np.concatenate((plane, np.zeros((20 - plane.shape[0], 3))), axis=0)
        return plane_parameters2 #, valid_region, plane_instance_parameter

    def plane2depth(self, plane_parameters, segmentation, gt_depth, h=480, w=640):

        depth_map = 1. / np.sum(self.K_inv_dot_xy_1.reshape(3, -1) * plane_parameters.reshape(3, -1), axis=0)
        depth_map = depth_map.reshape(h, w)

        planar_area = np.zeros(segmentation.shape[1:]).astype(bool)
        for i in range(segmentation.shape[0]):
            planar_area = np.logical_or(planar_area,segmentation[i])
        planar_area = planar_area.astype(int)

        # replace non planer region depth using sensor depth map
        depth_map[planar_area == 0] = gt_depth[planar_area == 0]
        return depth_map

    def mask_nhw_to_hw(self, mask_nhw):
        # print(mask_nhw.shape)
        # print(mask_nhw[0].shape)
        mask_hw = np.zeros((mask_nhw.shape[1],mask_nhw.shape[2]))
        for i in range(mask_nhw.shape[0]):
            this_mask = mask_nhw[i].astype(bool)
            mask_hw[this_mask] = i+1

        return mask_hw

    def drawDepthImage(self, depth, maxDepth=5):
        depthImage = np.clip(depth / maxDepth * 255, 0, 255).astype(np.uint8)
        depthImage = cv2.applyColorMap(255 - depthImage, colormap=cv2.COLORMAP_JET)
        return depthImage

    def drawDepth(self, depth):
        # assert depth.dim() == 3 or depth.dim() == 2
        # if depth.dim() == 3:
        #     depth = depth[0].cpu().numpy()
        # else:
        #     depth = depth.cpu().numpy()

        depth_color = self.drawDepthImage(depth)
        depth_mask = depth > 1e-4
        depth_mask = depth_mask[:, :, None]
        depth_color = depth_color * depth_mask
        # cv2.imwrite(addr, depth_color)
        depth_color[:, :, [2, 0]] = depth_color[:, :, [0, 2]]
        return depth_color

    def plot_depth_recall_curve(self, method_recalls, type='', save_path=None, method_color=None):
        assert type in ['pixel', 'PIXEL', 'Pixel', 'plane', 'PLANE', 'Plane']
        depth_threshold = np.arange(0, 0.65, 0.05)
        title = 'Per-'+type+' Recall(%)'

        pre_defined_recalls = {}
        if type in ['pixel', 'PIXEL', 'Pixel']:
            recall_planeAE = np.array(
                [0., 30.59, 51.88, 62.83, 68.54, 72.13, 74.28, 75.38, 76.57, 77.08, 77.35, 77.54, 77.86])
            pre_defined_recalls['PlaneAE'] = recall_planeAE

            recall_planeNet = np.array(
                [0., 22.79, 42.19, 52.71, 58.92, 62.29, 64.31, 65.20, 66.10, 66.71, 66.96, 67.11, 67.14])
            pre_defined_recalls['PlaneNet'] = recall_planeNet

        else:
            recall_planeAE = np.array(
                [0., 22.93, 40.17, 49.40, 54.58, 57.75, 59.72, 60.92, 61.84, 62.23, 62.56, 62.76, 62.93])
            pre_defined_recalls['PlaneAE'] = recall_planeAE

            recall_planeNet = np.array(
                [0., 15.78, 29.15, 37.48, 42.34, 45.09, 46.91, 47.77, 48.54, 49.02, 49.33, 49.53, 49.59])
            pre_defined_recalls['PlaneNet'] = recall_planeNet

        plt.figure(figsize=(5, 4))
        plt.xlabel('Depth Threshold', fontsize=18)
        plt.ylabel(title, fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        markers = {'PlaneNet': 'o', 'PlaneAE': '*'}
        colors = {'PlaneNet': 'gray', 'PlaneAE': '#FFCC99'}
        for method_name, recalls in pre_defined_recalls.items():
            assert len(depth_threshold) == len(recalls)
            plt.plot(depth_threshold, recalls, linewidth=3, marker=markers[method_name],label=method_name, color=colors[method_name])

        for method_name, recalls in method_recalls.items():
            assert len(depth_threshold) == len(recalls)
            if method_color is not None:
                plt.plot(depth_threshold, recalls, linewidth=3, marker='^', color=method_color[method_name], label=method_name)
            else:
                plt.plot(depth_threshold, recalls, linewidth=3, marker='^', label=method_name, color='#FF6666')

        plt.legend(loc='lower right', fontsize=16)
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        x_major_locator = MultipleLocator(0.1)
        y_major_locator = MultipleLocator(20)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'depth_recall_%s.png'%(type)))
        else:
            plt.savefig('./depth_recall_%s.png'%(type))
        plt.close()
        return './depth_recall_%s.png'%(type)

    def plot_normal_recall_curve(self, method_recalls, type='', save_path=None, method_color=None):
        assert type in ['pixel', 'PIXEL', 'Pixel', 'plane', 'PLANE', 'Plane']
        normal_threshold = np.linspace(0.0, 30, 13)
        title = 'Per-'+type+' Recall(%)'

        pre_defined_recalls = {}
        if type in ['pixel', 'PIXEL', 'Pixel']:
            recall_planeAE = np.array(
                [0., 30.20, 59.89, 69.79, 73.59, 75.67, 76.8, 77.3, 77.42, 77.57, 77.76, 77.85, 78.03])
            pre_defined_recalls['PlaneAE'] = recall_planeAE

            recall_planeNet = np.array(
                [0., 19.68, 43.78, 57.55, 63.36, 65.27, 66.03, 66.64, 66.99, 67.16, 67.20, 67.26, 67.29])
            pre_defined_recalls['PlaneNet'] = recall_planeNet
        else:
            recall_planeAE = np.array(
                [0., 20.05, 42.66, 51.85, 55.92, 58.34, 59.52, 60.35, 60.75, 61.23, 61.64, 61.84, 61.93])
            pre_defined_recalls['PlaneAE'] = recall_planeAE

            recall_planeNet = np.array(
                [0., 12.49, 29.70, 40.21, 44.92, 46.77, 47.71, 48.44, 48.83, 49.09, 49.20, 49.31, 49.38])
            pre_defined_recalls['PlaneNet'] = recall_planeNet

        plt.figure(figsize=(5, 4))
        plt.xlabel('Normal Threshold', fontsize=18)
        plt.ylabel(title, fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        markers = {'PlaneNet': 'o', 'PlaneAE': '*', 'PlaneRCNN': '.'}
        colors = {'PlaneNet': 'gray', 'PlaneAE': '#FFCC99', 'PlaneRCNN': 'mediumaquamarine'}
        for method_name, recalls in pre_defined_recalls.items():
            assert len(normal_threshold) == len(recalls)
            plt.plot(normal_threshold, recalls, linewidth=3, marker=markers[method_name], label=method_name,
                    color=colors[method_name])

        for method_name, recalls in method_recalls.items():
            assert len(normal_threshold) == len(recalls)
            if method_color is not None:
                plt.plot(normal_threshold, recalls, linewidth=3, marker='^', color=method_color[method_name], label=method_name)
            else:
                plt.plot(normal_threshold, recalls, linewidth=3, marker='^', label=method_name, color='#FF6666')

        plt.legend(loc='lower right', fontsize=16)
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        x_major_locator = MultipleLocator(5)
        y_major_locator = MultipleLocator(20)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'normal_recall_%s.png'%(type)))
        else:
            plt.savefig('./normal_recall_%s.png'%(type))
        plt.close()

        return './normal_recall_%s.png'%(type)

    def alpha_composition(self, alpha_BK1HW, value_BKCHW):
        """
        composition equation from 'Single-View View Synthesis with Multiplane Images'
        K is the number of planes, k=0 means the nearest plane, k=K-1 means the farthest plane
        :param alpha_BK1HW: alpha at each of the K planes
        :param value_BKCHW: rgb/disparity at each of the K planes
        :return:
        """
        B, K, _, H, W = alpha_BK1HW.size()
        alpha_comp_cumprod = torch.cumprod(1 - alpha_BK1HW, dim=1)  # BxKx1xHxW

        preserve_ratio = torch.cat((torch.ones((B, 1, 1, H, W), dtype=alpha_BK1HW.dtype, device=alpha_BK1HW.device),
                                    alpha_comp_cumprod[:, 0:K-1, :, :, :]), dim=1)  # BxKx1xHxW
        weights = alpha_BK1HW * preserve_ratio  # BxKx1xHxW
        value_composed = torch.sum(value_BKCHW * weights, dim=1, keepdim=False)  # Bx3xHxW

        return value_composed, weights

    def render_novel_view(self, mpi_all_rgb_src, mpi_all_sigma_src,
                            para_BS3, G_tgt_src,
                            K_src_inv, K_tgt):

        B, S, _, H, W = mpi_all_rgb_src.shape

        # src view normal&d
        tmp_distance_para_BS = torch.norm(para_BS3, dim=2)
        src_view_distance_para_BS = tmp_distance_para_BS
        tmp_distance_para_BS1 = torch.norm(para_BS3, dim=2, keepdim=True)
        src_view_norm_para_BS3 = para_BS3 / tmp_distance_para_BS1

        # tgt view normal&d
        # G_src_tgt = torch.inverse(G_tgt_src)
        G_src_tgt_inverse_transpose = G_tgt_src.transpose(1,2)
        src_view_plane_para = torch.cat([src_view_norm_para_BS3,tmp_distance_para_BS[...,None]],dim=2)
        tgt_view_plane_para = torch.matmul(G_src_tgt_inverse_transpose[:,None,:,:].repeat(1,S,1,1).to(para_BS3), src_view_plane_para[...,None].to(para_BS3))

        tgt_view_norm_para_BS3 = tgt_view_plane_para[:,:,:3,0]
        tgt_view_distance_para_BS = tgt_view_plane_para[:,:,3,0]
        para_tgt_BS3 = tgt_view_norm_para_BS3 * tgt_view_distance_para_BS[...,None]

        # # Apply scale factor
        # if scale_factor is None:
        #     with torch.no_grad():
        #         G_tgt_src = torch.clone(G_tgt_src)
        #         G_tgt_src[:, 0:3, 3] = G_tgt_src[:, 0:3, 3] * 0 #scale_factor.view(-1, 1)
        #         # print(scale_factor)

        # xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
        #     mpi_rendering.HomographySample(H, W, device=torch.device("cuda:0")).meshgrid,
        #     src_view_distance_para_BS, # disparity_all_src,
        #     K_src_inv
        # )

        K_640 = torch.tensor([[577.,   0., 320.], [  0., 577., 240.], [  0.,   0.,   1.]])[None,...].repeat(B,1,1)
        K_640_inv = torch.inverse(K_640)
        mesh2 = mpi_rendering.HomographySample(H, W, device=para_BS3.device).meshgrid2

        xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_para(
            mesh2,
            para_BS3, # disparity_all_src,
            K_640_inv.to(para_BS3)
        )


        xyz_tgt_BS3HW = mpi_rendering.get_tgt_xyz_from_plane_disparity(
            xyz_src_BS3HW,
            G_tgt_src
        )

        # Bx1xHxW, Bx3xHxW, Bx1xHxW
        tgt_imgs_syn, tgt_depth_syn, tgt_mask_syn, sub_ims_tgt = mpi_rendering.render_tgt_rgb_depth(
            src_view_norm_para_BS3,
            mpi_rendering.HomographySample(H, W, device=para_BS3.device), # self.homography_sampler_list[scale],
            mpi_all_rgb_src.to(para_BS3),
            mpi_all_sigma_src.to(para_BS3),
            src_view_distance_para_BS.to(para_BS3), # disparity_all_src,
            xyz_tgt_BS3HW,
            G_tgt_src,
            K_src_inv,
            K_tgt,
            use_alpha=True,
            is_bg_depth_inf=False
        )
        tgt_disparity_syn = torch.reciprocal(tgt_depth_syn)

        return {
            # "tgt_imgs_syn": tgt_imgs_syn,
            "tgt_disparity_syn": tgt_disparity_syn,
            "tgt_mask_syn": tgt_mask_syn,
            "sub_ims_tgt": sub_ims_tgt,
            "para_tgt_BS3": para_tgt_BS3
        }

    def zmf_render(self, sub_ims_BS4HW, plane_para_BS3, k_inv_dot_xy1):

        B = sub_ims_BS4HW.shape[0]
        S = sub_ims_BS4HW.shape[1]
        h = sub_ims_BS4HW.shape[3]
        w = sub_ims_BS4HW.shape[4]

        rgb_BS3HW = []
        sigma_BS1HW = []

        for batchi in range(B):
            
            depth_maps_inv = torch.matmul(plane_para_BS3[batchi], k_inv_dot_xy1.to(plane_para_BS3))
            depth_maps_inv = torch.clamp(depth_maps_inv, min=0.1, max=1e4)
            depth_maps = 1. / depth_maps_inv  # (valid_plane_num, h*w)


            sub_ims = sub_ims_BS4HW.permute(0,3,4,1,2)[batchi] # h, w, S, 4
            sub_ims = torch.reshape(sub_ims,(h*w,S,4)).to(plane_para_BS3)

            sub_ims_R = sub_ims[:,:,0]
            sub_ims_G = sub_ims[:,:,1]
            sub_ims_B = sub_ims[:,:,2]
            sub_ims_A = sub_ims[:,:,3]

            depth_order = torch.argsort(depth_maps.t(), dim=1)
            sub_ims_R = torch.gather(sub_ims_R, 1, depth_order)
            sub_ims_G = torch.gather(sub_ims_G, 1, depth_order)
            sub_ims_B = torch.gather(sub_ims_B, 1, depth_order)
            sub_ims_A = torch.gather(sub_ims_A, 1, depth_order)


            sub_RGBs = torch.stack([sub_ims_R,sub_ims_G,sub_ims_B])
            sub_RGBs = sub_RGBs.permute(2,0,1)
            rgb_BS3HW.append(torch.reshape(sub_RGBs,(S,3,h,w)))
            
            sub_ims_A = sub_ims_A[None, ...]
            sub_ims_A = sub_ims_A.permute(2,0,1)
            sigma_BS1HW.append(torch.reshape(sub_ims_A,(S,1,h,w)))

        rgb_BS3HW = torch.stack(rgb_BS3HW)
        sigma_BS1HW = torch.stack(sigma_BS1HW)

        imgs_syn, weights = self.alpha_composition(sigma_BS1HW/255., rgb_BS3HW/255.)

        
        imgs_syn = imgs_syn*255.
        imgs_syn = imgs_syn.type(torch.uint8)


        return imgs_syn

    def process(self, inputs, outputs):

        for idx, (input, output) in enumerate(zip(inputs, outputs)):
            


            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img_torch = panoptic_img.cpu()
            panoptic_img = panoptic_img_torch.numpy()




            if len(segments_info) == 0:
                print('zmf: this eval sample has no predicted plane.')
                continue
            if len(input['segments_info']) == 0:
                print('zmf: this eval sample has no gt plane.',input['file_name'])
                continue

            h, w = panoptic_img.shape
            self.K_inv_dot_xy_1 = input['K_inv_dot_xy_1'].cpu().numpy()


            pred_nplanes_masks = np.zeros([np.max(panoptic_img), h, w], dtype=np.uint8)
            for i in range(0, np.max(panoptic_img)):
                seg = panoptic_img == i+1

                pred_nplanes_masks[i, :, :] = seg.reshape(h, w)

            if len(segments_info) == 0:
                depth_from_pred_plane_hw = np.zeros([h, w])
                depth_from_pred_plane_hw3 = self.drawDepth(depth_from_pred_plane_hw)
            else:
                nplanes_paras = torch.stack([x['plane_para'] for x in segments_info])

                pixel_plane_para = self.get_plane_parameters(nplanes_paras.cpu().numpy(),pred_nplanes_masks)
                depth_map0 = np.zeros([h, w])
                depth_from_pred_plane_hw = self.plane2depth(pixel_plane_para,pred_nplanes_masks,depth_map0,h=h,w=w)
                depth_from_pred_plane_hw3 = self.drawDepth(depth_from_pred_plane_hw)

            # 1 SAVE SEG IMAGES
            # 2 SAVE DEPTH FROM PLANE

            im_idx = int(os.path.basename(input['file_name']).split('.')[0])

            view_gap = 20 if os.environ['DATASET_CHOICE'] == '3' else 2
            if im_idx % view_gap == 0:
                zmf_im = np.swapaxes(input['image'].cpu().numpy(),0,1)
                zmf_im = np.swapaxes(zmf_im,1,2)

                if os.environ['PLANE_UNDERSTAND_ONLY'] == 'False':
                
                    # Prepare src view sub images
                    orig_im = input['image']
                    all_sub_im = []
                    S = torch.max(panoptic_img_torch)
                    for maski in range(1,torch.max(panoptic_img_torch)+1):
                        tar_mask = torch.ones(panoptic_img_torch.shape)*maski
                        this_mask = (panoptic_img_torch == tar_mask).type(torch.uint8)[None, :,:]
                        this_im = orig_im * this_mask
                        this_im = torch.cat((this_im,(this_mask*255).type(torch.uint8)),dim=0) # RGBA
                        # if maski != 0: self.writer.add_image("pano/pred/{}_subim".format(im_idx), this_im, maski-1)
                        if maski != 0: all_sub_im.append(this_im)
                    sub_ims_orig = torch.stack(all_sub_im)
                    
                    # Render src view image
                    K_inv_dot_xy_1_3hxw = input['K_inv_dot_xy_1'].view(3,-1).float()
                    imgs_syn = self.zmf_render(sub_ims_orig[None,:,:,:,:], nplanes_paras[None,:,:], K_inv_dot_xy_1_3hxw)
                    self.writer.add_image("pano/pred/{}_src".format(im_idx), imgs_syn[0], 0)


                    # Prepare tgt view
                    G_src_tgt = input['G_src_tgt'][None,:]
                    K = input['intrinsic'][None,:]

                    tgt_image = input['tgt_image'].type(torch.uint8)
                    self.writer.add_image("pano/pred/{}_tgt_GT".format(im_idx), tgt_image, 0)
                    K_inv = torch.inverse(K)
                    G_tgt_src = torch.inverse(G_src_tgt)

                    
                    orig_rgba_BS4HW = sub_ims_orig[None,...]
                    orig_rgb_BS3HW = orig_rgba_BS4HW[:,:,:3,:,:]
                    orig_a_BS1HW = orig_rgba_BS4HW[:,:,3:,:,:]
                    
                    result_dic = self.render_novel_view(orig_rgb_BS3HW/255.,orig_a_BS1HW/255.,para_BS3=nplanes_paras[None,...],G_tgt_src=G_tgt_src, K_src_inv=K_inv,K_tgt=K)

                    # tgt_imgs_syn = result_dic["tgt_imgs_syn"]
                    tgt_disparity_syn = result_dic["tgt_disparity_syn"]
                    tgt_depth_syn = torch.reciprocal(tgt_disparity_syn)
                    tgt_mask_syn = result_dic["tgt_mask_syn"]
                    rgb_tgt_valid_mask = torch.ge(tgt_mask_syn, 2).to(torch.float32)
                    # loss_map = torch.abs(tgt_imgs_syn - tgt_imgs_scaled) * rgb_tgt_valid_mask
                    # loss_rgb_tgt = loss_map.mean()

                    sub_ims_tgt = result_dic["sub_ims_tgt"]
                    para_tgt_BS3 = result_dic["para_tgt_BS3"]
                    tgt_imgs_zmf = self.zmf_render(sub_ims_tgt, para_tgt_BS3, K_inv_dot_xy_1_3hxw)
                    self.writer.add_image("pano/pred/{}_tgt".format(im_idx), tgt_imgs_zmf[0], 0)

                    tgt_depth_syn_hw3 = self.drawDepth(tgt_depth_syn[0,0].cpu().numpy()).transpose(2, 0, 1)
                    tgt_mask_syn_hw3 = self.drawDepth(rgb_tgt_valid_mask[0,0].cpu().numpy()).transpose(2, 0, 1)
                    self.writer.add_image("pano/pred/{}_tgt_depth".format(im_idx), tgt_depth_syn_hw3, 0)
                    self.writer.add_image("pano/pred/{}_tgt_valid".format(im_idx), tgt_mask_syn_hw3, 0)

                    gt_depth_map_tar = input['tgt_depth_map'].cpu().numpy()
                    gt_see_depth_tar = self.drawDepth(gt_depth_map_tar)
                    if gt_see_depth_tar.shape[0] > 300:
                        gt_see_depth_tar = cv2.resize(gt_see_depth_tar,(400,300)).transpose(2, 0, 1)
                    else:
                        gt_see_depth_tar = gt_see_depth_tar.transpose(2, 0, 1)
                    # storage.put_image("pano/pred/{}_depth_gt".format(im_idx), gt_see_depth_tar)
                    self.writer.add_image("pano/pred/{}_tgt_depth_gt".format(im_idx), gt_see_depth_tar, 0)





                assert zmf_im.shape[:2] == panoptic_img_torch.shape

                
                metadata = MetadataCatalog.get("scannet_val_panoptic")
                v_pred = Visualizer(zmf_im, metadata, instance_mode=ColorMode.IMAGE)
                v_pred = v_pred.draw_panoptic_seg_predictions(panoptic_img_torch, segments_info)

                vis_img = v_pred.get_image()
                if vis_img.shape[0] > 300:
                    vis_img = cv2.resize(vis_img,(400,300)).transpose(2, 0, 1)
                else:
                    vis_img = vis_img.transpose(2, 0, 1)

                self.writer.add_image("pano/pred/{}".format(im_idx), vis_img, 0)

                # 2 SAVE DEPTH FROM PLANE
                
                if depth_from_pred_plane_hw3.shape[0] > 300:
                    see_depth_from_pred_plane = cv2.resize(depth_from_pred_plane_hw3,(400,300)).transpose(2, 0, 1)
                else:
                    see_depth_from_pred_plane = depth_from_pred_plane_hw3.transpose(2, 0, 1)

                self.writer.add_image("pano/pred/{}_depth".format(im_idx), see_depth_from_pred_plane, 0)

                gt_depth_map = input['depth_from_plane'].cpu().numpy()
                gt_see_depth = self.drawDepth(gt_depth_map)
                if gt_see_depth.shape[0] > 300:
                    gt_see_depth = cv2.resize(gt_see_depth,(400,300)).transpose(2, 0, 1)
                else:
                    gt_see_depth = gt_see_depth.transpose(2, 0, 1)
                self.writer.add_image("pano/pred/{}_depth_gt".format(im_idx), gt_see_depth, 0)



            targets_per_image = input['instances']
            gt_masks = targets_per_image.gt_masks.cpu().numpy()



            # EVAL 1
            plane_info = self.evaluateMasks(panoptic_img, gt_masks, device=None, pred_non_plane_idx=0,printInfo=False)

            # EVAL 2
            gt_depth_from_plane = input['depth_from_plane'].cpu().numpy()

            gt_masks_hw = self.mask_nhw_to_hw(gt_masks)

            depth_pixel_recall, depth_plane_recall = self.eval_plane_recall_depth(input['file_name'], panoptic_img.copy(), gt_masks_hw.copy(), depth_from_pred_plane_hw, gt_depth_from_plane)

            # EVAL 3
            instance_param = nplanes_paras.cpu().numpy()
            gt_plane_instance_parameter = input['planes_paras'].cpu().numpy()
            gt_plane_instance_parameter = gt_plane_instance_parameter[:,:3] / gt_plane_instance_parameter[:,3:]
            normal_plane_recall, normal_pixel_recall = self.eval_plane_recall_normal(input['file_name'], panoptic_img.copy(), gt_masks_hw.copy(),
                                                                            instance_param, gt_plane_instance_parameter)




            

            if segments_info is None:
                # If "segments_info" is None, we assume "panoptic_img" is a
                # H*W int32 image storing the panoptic_id in the format of
                # category_id * label_divisor + instance_id. We reserve -1 for
                # VOID label, and add 1 to panoptic_img since the official
                # evaluation script uses 0 for VOID label.
                label_divisor = self._metadata.label_divisor
                segments_info = []

                # print(label_divisor, np.unique(panoptic_img))
                for panoptic_label in np.unique(panoptic_img):
                    if panoptic_label == -1:
                        # VOID region.
                        continue
                    pred_class = panoptic_label // label_divisor
                    isthing = (
                        pred_class in self._metadata.thing_dataset_id_to_contiguous_id.values()
                    )
                    segments_info.append(
                        {
                            "id": int(panoptic_label) + 1,
                            "category_id": int(pred_class),
                            "isthing": bool(isthing),
                        }
                    )
                # Official evaluation script uses 0 for VOID label.
                panoptic_img += 1

            


            self._predictions.append(
                {

                    "eval_info": plane_info,
                    "depth_pixel_recall": depth_pixel_recall, 
                    "depth_plane_recall": depth_plane_recall,
                    "normal_pixel_recall": normal_pixel_recall, 
                    "normal_plane_recall": normal_plane_recall
                }
            )

    # https://github.com/davisvideochallenge/davis/blob/master/python/lib/davis/measures/jaccard.py
    def eval_iou(self, annotation, segmentation):
        """ Compute region similarity as the Jaccard Index.

        Arguments:
            annotation   (ndarray): binary annotation   map.
            segmentation (ndarray): binary segmentation map.

        Return:
            jaccard (float): region similarity

        """
        annotation = annotation.astype(np.bool)
        segmentation = segmentation.astype(np.bool)

        if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
            return 1
        else:
            return np.sum((annotation & segmentation)) / \
                np.sum((annotation | segmentation), dtype=np.float32)

    # https://github.com/svip-lab/PlanarReconstruction/blob/master/utils/metric.py
    # https://github.com/art-programmer/PlaneNet/blob/master/utils.py#L2115
    def eval_plane_recall_depth(self, filename, predSegmentations, gtSegmentations, predDepths, gtDepths, threshold=0.5):
        # predNumPlanes = pred_plane_num  # actually, it is the maximum number of the predicted planes
        predNumPlanes = np.max(predSegmentations)
        gtNumPlanes = len(np.unique(gtSegmentations))-1
        predSegmentations -= 1
        gtSegmentations -= 1
        
        # print(np.unique(predSegmentations),predNumPlanes) # [-1  0  1  2  3  4  5  6  7] 8
        # print(np.unique(gtSegmentations),gtNumPlanes) # [-1.  0.  1.  2.  3.  4.] 5


        if len(gtSegmentations.shape) == 2:
            gtSegmentations = (np.expand_dims(gtSegmentations, -1) == np.arange(gtNumPlanes)).astype(np.float32)  # h, w, gtNumPlanes
        if len(predSegmentations.shape) == 2:
            predSegmentations = (np.expand_dims(predSegmentations, -1) == np.arange(predNumPlanes)).astype(np.float32)  # h, w, predNumPlanes

        planeAreas = gtSegmentations.sum(axis=(0, 1))  # gt plane pixel number

        intersectionMask = np.expand_dims(gtSegmentations, -1) * np.expand_dims(predSegmentations, 2) > 0.5  # h, w, gtNumPlanes, predNumPlanes

        depthDiffs = gtDepths - predDepths  # h, w
        depthDiffs = depthDiffs[:, :, np.newaxis, np.newaxis]  # h, w, 1, 1

        intersection = np.sum((intersectionMask).astype(np.float32), axis=(0, 1))  # gtNumPlanes, predNumPlanes

        planeDiffs = np.abs(depthDiffs * intersectionMask).sum(axis=(0, 1)) / np.maximum(intersection, 1e-4)  # gtNumPlanes, predNumPlanes
        planeDiffs[intersection < 1e-4] = 1

        union = np.sum(
            ((np.expand_dims(gtSegmentations, -1) + np.expand_dims(predSegmentations, 2)) > 0.5).astype(np.float32),
            axis=(0, 1))  # gtNumPlanes, predNumPlanes
        planeIOUs = intersection / np.maximum(union, 1e-4)  # gtNumPlanes, predNumPlanes

        numPredictions = int(predSegmentations.max(axis=(0, 1)).sum())

        numPixels = planeAreas.sum()
        IOUMask = (planeIOUs > threshold).astype(np.float32)

        minDiff = np.min(planeDiffs * IOUMask + 1000000 * (1 - IOUMask), axis=1)

        stride = 0.05
        pixelRecalls = []
        planeStatistics = []
        for step in range(int(0.61 / stride + 1)):
            diff = step * stride

            if numPixels == 0: 
                print('zmf: numPixels',numPixels,filename)
            pixelRecalls.append(np.minimum((intersection * (planeDiffs <= diff).astype(np.float32) * IOUMask).sum(1), planeAreas).sum() / numPixels)
            planeStatistics.append(((minDiff <= diff).sum(), gtNumPlanes, numPredictions))
        return pixelRecalls, planeStatistics

    # https://github.com/svip-lab/PlanarReconstruction/blob/master/utils/metric.py
    def eval_plane_recall_normal(self, filename, segmentation, gt_segmentation, param, gt_param, threshold=0.5):
        """
        :param segmentation: label map for plane segmentation [h, w] where 20 indicate non-planar
        :param gt_segmentation: ground truth label for plane segmentation where 20 indicate non-planar
        :param threshold: value for iou
        :return: percentage of correctly predicted ground truth planes correct plane
        """
        depth_threshold_list = np.linspace(0.0, 30, 13)

        segmentation -= 1
        gt_segmentation -= 1

        pred_plane_idxs = np.unique(segmentation)
        pred_plane_idx_max = pred_plane_idxs[-1]
        plane_num = pred_plane_idx_max + 1
        gt_plane_num = len(np.unique(gt_segmentation)) - 1


        # 13: 0:0.05:0.6
        plane_recall = np.zeros((gt_plane_num, len(depth_threshold_list)))
        pixel_recall = np.zeros((gt_plane_num, len(depth_threshold_list)))

        plane_area = 0.0



        # check if plane is correctly predict
        for i in range(gt_plane_num):
            gt_plane = gt_segmentation == i
            plane_area += np.sum(gt_plane)

            for j in range(plane_num):
                pred_plane = segmentation == j
                iou = self.eval_iou(gt_plane, pred_plane)

                if iou > threshold:
                    # mean degree difference over overlap region:
                    gt_p = gt_param[i]
                    pred_p = param[j]

                    n_gt_p = gt_p / np.linalg.norm(gt_p)
                    n_pred_p = pred_p / np.linalg.norm(pred_p)

                    angle = np.arccos(np.clip(np.dot(n_gt_p, n_pred_p), -1.0, 1.0))
                    degree = np.degrees(angle)
                    depth_diff = degree

                    # compare with threshold difference
                    plane_recall[i] = (depth_diff < depth_threshold_list).astype(np.float32)
                    pixel_recall[i] = (depth_diff < depth_threshold_list).astype(np.float32) * \
                                    (np.sum(gt_plane * pred_plane))
                    break
        
        if plane_area == 0: 
            print('zmf: plane_area',plane_area,filename)
        pixel_recall = np.sum(pixel_recall, axis=0).reshape(-1) / plane_area

        plane_recall_new = np.zeros((len(depth_threshold_list), 3))
        plane_recall = np.sum(plane_recall, axis=0).reshape(-1, 1)
        plane_recall_new[:, 0:1] = plane_recall
        plane_recall_new[:, 1] = gt_plane_num
        plane_recall_new[:, 2] = plane_num

        return plane_recall_new, pixel_recall

    # https://github.com/svip-lab/PlanarReconstruction/blob/master/utils/metric.py
    # https://github.com/yi-ming-qian/interplane/blob/master/utils/metric.py
    def evaluateMasks(self, predSegmentations_np, gtSegmentations, device, pred_non_plane_idx, gt_non_plane_idx=20, printInfo=False):
        """
        :param predSegmentations:
        :param gtSegmentations:
        :param device:
        :param pred_non_plane_idx:
        :param gt_non_plane_idx:
        :param printInfo:
        :return:
        """
        predSegmentations = torch.from_numpy(predSegmentations_np) #.to(device)
        gtSegmentations = torch.from_numpy(gtSegmentations) #.to(device)

        # print(gtSegmentations.shape)

        pred_masks = []
        if pred_non_plane_idx == 0:
            # print(np.unique(predSegmentations_np))
            # print(np.max(predSegmentations_np)+1)
            for i in range(1, np.max(predSegmentations_np)+1):
                mask_i = predSegmentations == i
                mask_i = mask_i.float()
                if mask_i.sum() > 0:
                    pred_masks.append(mask_i)
        else:
            assert False
            assert pred_non_plane_idx == -1 or pred_non_plane_idx == 0
            for i in range(gt_non_plane_idx + 1, 100):
                mask_i = predSegmentations == i
                mask_i = mask_i.float()
                if mask_i.sum() > 0:
                    pred_masks.append(mask_i)
        
        if len(pred_masks) == 0:
            pred_masks.append(predSegmentations)
            # print('err',predSegmentations)
            # print('err2',predSegmentations.shape)
        predMasks = torch.stack(pred_masks, dim=0)
        # print(predMasks.shape)

        # gt_masks = []
        # if gt_non_plane_idx > 0:
        #     for i in range(gt_non_plane_idx):
        #         mask_i = gtSegmentations == i
        #         mask_i = mask_i.float()
        #         if mask_i.sum() > 0:
        #             gt_masks.append(mask_i)
        # else:
        #     assert pred_non_plane_idx == -1 or pred_non_plane_idx == 0
        #     for i in range(gt_non_plane_idx+1, 100):
        #         mask_i = gtSegmentations == i
        #         mask_i = mask_i.float()
        #         if mask_i.sum() > 0:
        #             gt_masks.append(mask_i)
        # gtMasks = torch.stack(gt_masks, dim=0)
        gtMasks = gtSegmentations

        valid_mask = (gtMasks.max(0)[0]).unsqueeze(0)

        gtMasks = torch.cat([gtMasks, torch.clamp(1 - gtMasks.sum(0, keepdim=True), min=0)], dim=0)  # M+1, H, W
        predMasks = torch.cat([predMasks, torch.clamp(1 - predMasks.sum(0, keepdim=True), min=0)], dim=0)  # N+1, H, W

        intersection = (gtMasks.unsqueeze(1) * predMasks * valid_mask).sum(-1).sum(-1).float()
        union = (torch.max(gtMasks.unsqueeze(1), predMasks) * valid_mask).sum(-1).sum(-1).float()

        N = intersection.sum()

        RI = 1 - ((intersection.sum(0).pow(2).sum() + intersection.sum(1).pow(2).sum()) / 2 - intersection.pow(2).sum()) / (
                N * (N - 1) / 2)
        joint = intersection / N
        marginal_2 = joint.sum(0)
        marginal_1 = joint.sum(1)
        H_1 = (-marginal_1 * torch.log2(marginal_1 + (marginal_1 == 0).float())).sum()
        H_2 = (-marginal_2 * torch.log2(marginal_2 + (marginal_2 == 0).float())).sum()

        B = (marginal_1.unsqueeze(-1) * marginal_2)
        log2_quotient = torch.log2(torch.clamp(joint, 1e-8) / torch.clamp(B, 1e-8)) * (torch.min(joint, B) > 1e-8).float()
        MI = (joint * log2_quotient).sum()
        voi = H_1 + H_2 - 2 * MI

        IOU = intersection / torch.clamp(union, min=1)
        SC = ((IOU.max(-1)[0] * torch.clamp((gtMasks * valid_mask).sum(-1).sum(-1), min=1e-4)).sum() / N + (
                IOU.max(0)[0] * torch.clamp((predMasks * valid_mask).sum(-1).sum(-1), min=1e-4)).sum() / N) / 2
        info = [RI.item(), voi.item(), SC.item()]
        if printInfo:
            print('mask statistics', info)
            pass
        return info

    def evaluate(self):
        comm.synchronize()


        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # print(self._predictions) # RI, VOI, SC

        all_RI = []
        all_VOI = []
        all_SC = []
        pixelDepth_recall_curve = np.zeros((13))
        planeDepth_recall_curve = np.zeros((13, 3))
        pixelNorm_recall_curve = np.zeros((13))
        planeNorm_recall_curve = np.zeros((13, 3))
        for p in self._predictions:
            eval_info = p['eval_info']
            all_RI.append(eval_info[0])
            all_VOI.append(eval_info[1])
            all_SC.append(eval_info[2])
            depth_pixel_recall = p['depth_pixel_recall']
            depth_plane_recall = p['depth_plane_recall']
            normal_pixel_recall = p['normal_pixel_recall']
            normal_plane_recall = p['normal_plane_recall']
            pixelDepth_recall_curve += np.array(depth_pixel_recall)
            planeDepth_recall_curve += np.array(depth_plane_recall)
            pixelNorm_recall_curve += normal_pixel_recall
            planeNorm_recall_curve += normal_plane_recall

        all_RI = np.array(all_RI)
        all_VOI = np.array(all_VOI)
        all_SC = np.array(all_SC)
        depth_recalls_pixel = {"Mask2Former": pixelDepth_recall_curve / len(self._predictions) * 100}
        depth_recalls_plane = {"Mask2Former": planeDepth_recall_curve[:, 0] / planeDepth_recall_curve[:, 1] * 100}
        see_depth_pixel = self.plot_depth_recall_curve(depth_recalls_pixel, type='pixel')
        see_depth_plane = self.plot_depth_recall_curve(depth_recalls_plane, type='plane')
        see_depth_pixel = cv2.imread(see_depth_pixel).transpose(2, 0, 1)
        see_depth_plane = cv2.imread(see_depth_plane).transpose(2, 0, 1)

        normal_recalls_pixel = {"Mask2Former": pixelNorm_recall_curve / len(self._predictions) * 100}
        normal_recalls_plane = {"Mask2Former": planeNorm_recall_curve[:, 0] / planeNorm_recall_curve[:, 1] * 100}
        see_normal_pixel = self.plot_normal_recall_curve(normal_recalls_pixel, type='pixel')
        see_normal_plane = self.plot_normal_recall_curve(normal_recalls_plane, type='plane')
        see_normal_pixel = cv2.imread(see_normal_pixel).transpose(2, 0, 1)
        see_normal_plane = cv2.imread(see_normal_plane).transpose(2, 0, 1)

        # storage = get_event_storage()
        # storage.put_scalar("Mean RI on %d samples"%len(self._predictions), all_RI.mean())
        self.writer.add_scalar("Mean RI on %d samples"%len(self._predictions), all_RI.mean(), 0)
        # storage.put_scalar("Mean VOI on %d samples"%len(self._predictions), all_VOI.mean())
        self.writer.add_scalar("Mean VOI on %d samples"%len(self._predictions), all_VOI.mean(), 0)
        # storage.put_scalar("Mean SC on %d samples"%len(self._predictions), all_SC.mean())
        self.writer.add_scalar("Mean SC on %d samples"%len(self._predictions), all_SC.mean(), 0)

        # vis_img = cv2.resize(vis_img,(400,300)).transpose(2, 0, 1)
        # print(see_depth_pixel.shape)
        # storage.put_image("stat/eval/depth_pixel", see_depth_pixel)
        self.writer.add_image("stat/eval/depth_pixel", see_depth_pixel, 0)
        # storage.put_image("stat/eval/depth_plane", see_depth_plane)
        self.writer.add_image("stat/eval/depth_plane", see_depth_plane, 0)
        # storage.put_image("stat/eval/normal_pixel", see_normal_pixel)
        self.writer.add_image("stat/eval/normal_pixel", see_normal_pixel, 0)
        # storage.put_image("stat/eval/normal_plane", see_normal_plane)
        self.writer.add_image("stat/eval/normal_plane", see_normal_plane, 0)

        return


def _print_panoptic_results(pq_res):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    logger.info("Panoptic Evaluation Results:\n" + table)

if __name__ == "__main__":
    from detectron2.utils.logger import setup_logger

    logger = setup_logger()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json")
    parser.add_argument("--gt-dir")
    parser.add_argument("--pred-json")
    parser.add_argument("--pred-dir")
    args = parser.parse_args()

    from panopticapi.evaluation import pq_compute

    with contextlib.redirect_stdout(io.StringIO()):
        pq_res = pq_compute(
            args.gt_json, args.pred_json, gt_folder=args.gt_dir, pred_folder=args.pred_dir
        )
        _print_panoptic_results(pq_res)
