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

from detectron2.utils.events import get_event_storage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

from .evaluator import DatasetEvaluator
import mpi_rendering
# from ssim import SSIM2
from piqa import SSIM,PSNR,LPIPS

logger = logging.getLogger(__name__)


class ScanNetPanopticEvaluator(DatasetEvaluator):
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

        try:
            tmp = get_event_storage()
        except:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter('./zmf_eval_only')
        # self.ssim = SSIM(size_average=True).cuda()
        self.ssim = SSIM().cuda()
        self.psnr = PSNR()
        self.lpips = LPIPS().cuda()

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

        H, W = segmentation.shape[1:]

        plane_parameters2 = np.ones((3, H, W))
        for i in range(segmentation.shape[0]):
            plane_mask = segmentation[i]
            plane_mask = plane_mask.astype(np.float32)
            cur_plane_param_map = np.ones((3, H, W)) * plane[i, :].reshape(3, 1, 1)
            plane_parameters2 = plane_parameters2 * (1-plane_mask) + cur_plane_param_map * plane_mask

        # # plane_instance parameter, padding zero to fix size
        # plane_instance_parameter = np.concatenate((plane, np.zeros((20 - plane.shape[0], 3))), axis=0)
        return plane_parameters2 #, valid_region, plane_instance_parameter

    def plane2depth(self, plane_parameters, segmentation, gt_depth, H=480, W=640):

        depth_map = 1. / np.sum(self.K_inv_dot_xy_1.reshape(3, -1) * plane_parameters.reshape(3, -1), axis=0)
        depth_map = depth_map.reshape(H, W)

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
        depth_mask = depth_mask[:, :, np.newaxis]
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

    def process(self, inputs, outputs):

        for idx, (input, output) in enumerate(zip(inputs, outputs)):

            im_idx = int(os.path.basename(input['file_name']).split('.')[0])
            view_gap = 20 if os.environ['DATASET_CHOICE'] == '3' else 2
            if im_idx % view_gap == 0:
                save_images = True
            else:
                save_images = False

            render_wo_nonplane = False



            # PREPARE EVALUATION ITEMS

            # 1 segmentation
            panoptic_img_HW, segments_info = output["panoptic_seg"]
            panoptic_img_torch_HW = panoptic_img_HW.cpu()
            panoptic_img_HW = panoptic_img_torch_HW.numpy()
            H, W = panoptic_img_HW.shape

            if len(segments_info) == 0:
                print('zmf: this eval sample has no predicted plane.')
                continue
            if len(input['segments_info']) == 0:
                print('zmf: this eval sample has no gt plane.',input['file_name'])
                continue

            device_type = segments_info[0]['mask_orig']

            # 2 plane depth
            self.K_inv_dot_xy_1 = input['K_inv_dot_xy_1'].cpu().numpy()
            pred_plane_mask_PHW = np.zeros([np.max(panoptic_img_HW), H, W], dtype=np.uint8)
            for i in range(0, np.max(panoptic_img_HW)):
                seg = panoptic_img_HW == i+1
                pred_plane_mask_PHW[i, :, :] = seg.reshape(H, W)
            pred_src_plane_para_P3 = torch.stack([x['plane_para'] for x in segments_info]).to(device_type)
            pred_plane_para_pixel_3HW = self.get_plane_parameters(pred_src_plane_para_P3.cpu().numpy(),pred_plane_mask_PHW)
            depth_map0_HW = np.zeros([H, W])
            pred_plane_depth_HW = self.plane2depth(pred_plane_para_pixel_3HW,pred_plane_mask_PHW,depth_map0_HW,H=H,W=W)
            pred_plane_depth_HW3 = self.drawDepth(pred_plane_depth_HW)

            # 3 target view
            pred_plane_mask_orig_PHW = torch.stack([x['mask_orig'] for x in segments_info]).to(device_type)
            pred_src_plane_RGBA_P4hw = torch.stack([x['rgba'] for x in segments_info]).sigmoid().to(device_type)
            pred_src_nonplane_RGBA_S4HW = output["pred_nonplane_rgba"][0].to(device_type)
            gt_tgt_view_rgb_3HW         = (input['tgt_image']/255.).to(device_type)
            gt_tgt_view_depth_map_HW    = input['tgt_depth_map'].to(device_type)
            gt_G_src_tgt_44             = input['G_src_tgt'].to(device_type)
            gt_src_view_rgb_3HW         = (input['image']/255.).to(device_type)
            K_inv_dot_xy_1_13HxW         = input['K_inv_dot_xy_1'].view(3,-1).to(device_type)[None,...]

            if render_wo_nonplane:
                pred_src_nonplane_RGBA_S4HW = torch.zeros(pred_src_nonplane_RGBA_S4HW.shape).to(device_type)
            results, vis_results = mpi_rendering.render_everything(pred_plane_mask_orig_PHW, pred_src_plane_RGBA_P4hw, pred_src_plane_para_P3, 
                                                                    pred_src_nonplane_RGBA_S4HW, gt_G_src_tgt_44, gt_src_view_rgb_3HW, K_inv_dot_xy_1_13HxW,
                                                                    device_type, save_images)


            # SAVE VARIOUS IMAGES
            if save_images:
                # 0 target view
                if True:
                    P,_, h, w = pred_src_plane_RGBA_P4hw.shape
                    S, _, H, W = pred_src_nonplane_RGBA_S4HW.shape

                    for i in range(P):
                        this_plane_RGBA_4HW = vis_results['pred_src_plane_RGBA_P4HW'][i]
                        see_this_plane_pred_RGBA_4HW = (this_plane_RGBA_4HW*255).cpu().numpy().astype(np.uint8) # [:,:,[2,1,0,3]]
                        try:
                            storage = get_event_storage()
                            storage.put_image("{}/plane_{}".format(im_idx,i), see_this_plane_pred_RGBA_4HW)
                        except:
                            self.writer.add_image("{}/plane_{}".format(im_idx,i), see_this_plane_pred_RGBA_4HW, 0)
                            
                    
                    for i in range(S):
                        this_plane_RGBA_4HW = vis_results['pred_src_nonplane_RGB_1S4HW'][0,i]
                        see_this_plane_debug_RGBA_4HW = (this_plane_RGBA_4HW*255).cpu().numpy().astype(np.uint8) # [:,:,[2,1,0,3]]
                        try:
                            storage.put_image("{}/nonplane_{}".format(im_idx,i), see_this_plane_debug_RGBA_4HW)
                        except:
                            self.writer.add_image("{}/nonplane_{}".format(im_idx,i), see_this_plane_debug_RGBA_4HW, 0)
                    
                    

                    pred_tgt_depth_HW = vis_results['pred_tgt_depth_HW']
                    pred_tgt_depth_HW3 = self.drawDepth(pred_tgt_depth_HW.cpu().numpy())
                    see_pred_tgt_depth_3HW = cv2.resize(pred_tgt_depth_HW3,(int(300/H*W),300)).transpose(2, 0, 1)
                    try:
                        storage.put_image("results/{}_tgt_depth".format(im_idx), see_pred_tgt_depth_3HW)
                    except:
                        self.writer.add_image("results/{}_tgt_depth".format(im_idx), see_pred_tgt_depth_3HW, 0)

                    gt_tgt_view_depth_map_HW3 = self.drawDepth(gt_tgt_view_depth_map_HW.cpu().numpy())
                    see_gt_tgt_view_depth_map_HW3 = cv2.resize(gt_tgt_view_depth_map_HW3,(int(300/H*W),300)).transpose(2, 0, 1)
                    try:
                        storage.put_image("results/{}_tgt_depth_gt".format(im_idx), see_gt_tgt_view_depth_map_HW3)
                    except:
                        self.writer.add_image("results/{}_tgt_depth_gt".format(im_idx), see_gt_tgt_view_depth_map_HW3, 0)

                    
                
                    pred_tgt_RGB_3HW = vis_results['pred_tgt_RGB_3HW']
                    pred_tgt_vmask_3HW = vis_results['pred_tgt_vmask_HW'].float().repeat(3,1,1)
                    zeros_3HW = torch.zeros(pred_tgt_RGB_3HW.shape).to(pred_tgt_RGB_3HW)

                    pred_tgt_RGB_3HW = torch.where(pred_tgt_vmask_3HW>0.1,pred_tgt_RGB_3HW,zeros_3HW)
                    see_pred_tgt_RGB_3HW = (pred_tgt_RGB_3HW*255).type(torch.uint8).cpu().numpy() #[:,:,[2,1,0]]
                    

                    gt_tgt_view_rgb_3HW = torch.where(pred_tgt_RGB_3HW>0.1,gt_tgt_view_rgb_3HW,zeros_3HW)
                    see_gt = (gt_tgt_view_rgb_3HW*255.).type(torch.uint8).cpu().numpy() #[:,:,[2,1,0]]
                    
                    pred_tgt_vmask_3HW = pred_tgt_vmask_3HW / torch.max(pred_tgt_vmask_3HW)
                    see_mask = (pred_tgt_vmask_3HW*255.).type(torch.uint8).cpu().numpy()
                    try:
                        storage.put_image("results/{}_tgt_im_pred".format(im_idx), see_pred_tgt_RGB_3HW)
                        storage.put_image("results/{}_tgt_im_gt".format(im_idx), see_gt)
                        storage.put_image("results/{}_tgt_mask".format(im_idx), see_mask)
                    except:
                        self.writer.add_image("results/{}_tgt_im_pred".format(im_idx), see_pred_tgt_RGB_3HW, 0)
                        self.writer.add_image("results/{}_tgt_im_gt".format(im_idx), see_gt, 0)
                        self.writer.add_image("results/{}_tgt_mask".format(im_idx), see_mask, 0)
                
                # 1 save segmentation
                if True:
                    src_image_HW3 = input['image'].permute(1,2,0).cpu().numpy()
                    metadata = MetadataCatalog.get("scannet_val_panoptic")
                    show_pred_seg = Visualizer(src_image_HW3, metadata, instance_mode=ColorMode.IMAGE)
                    show_pred_seg = show_pred_seg.draw_panoptic_seg_predictions(panoptic_img_torch_HW, segments_info)
                    
                    see_pred_seg_3HW = show_pred_seg.get_image()
                    see_pred_seg_3HW = cv2.resize(see_pred_seg_3HW,(int(300/H*W),300)).transpose(2, 0, 1)

                    try:
                        storage = get_event_storage()
                        storage.put_image("results/{}_src_seg".format(im_idx), see_pred_seg_3HW)
                        storage.put_image("results/{}_src_im_gt".format(im_idx), input['image'])
                    except:
                        self.writer.add_image("results/{}_src_seg".format(im_idx), see_pred_seg_3HW, 0)
                        self.writer.add_image("results/{}_src_im_gt".format(im_idx), input['image'], 0)

                # 2 save plane depth
                if True:
                    see_pred_plane_depth_3HW = cv2.resize(pred_plane_depth_HW3,(int(300/H*W),300)).transpose(2, 0, 1)
                    try:
                        storage.put_image("results/{}_src_depth_plane".format(im_idx), see_pred_plane_depth_3HW)
                    except:
                        self.writer.add_image("results/{}_src_depth_plane".format(im_idx), see_pred_plane_depth_3HW, 0)
                
                # 3 save gt depth
                if True:
                    gt_depth_map = input['depth_map'].cpu().numpy()
                    gt_depth_HW3 = self.drawDepth(gt_depth_map)
                    see_gt_depth_3HW = cv2.resize(gt_depth_HW3,(int(300/H*W),300)).transpose(2, 0, 1)
                    try:
                        storage.put_image("results/{}_src_depth_gt".format(im_idx), see_gt_depth_3HW)
                    except:
                        self.writer.add_image("results/{}_src_depth_gt".format(im_idx), see_gt_depth_3HW, 0)

                # 4 save nonplane depth
                if True:
                    mpi_rgba = output['pred_nonplane_rgba']
                    pred_device = mpi_rgba.device
                    B, S, _, _, _ = mpi_rgba.shape # 1, 32, 4, 256, 384

                    K_640 = torch.tensor([[577.,   0., 320.], [  0., 577., 240.], [  0.,   0.,   1.]])[None,...].repeat(B,1,1)
                    K_640_inv = torch.inverse(K_640)
                    f = 577 * (640/W)
                    K = torch.tensor([[f,   0., W/2], [  0., f, H/2], [  0.,   0.,   1.]])[None,...].repeat(B,1,1)
                    K_inv = torch.inverse(K)
                    start_disparity = 0.999
                    end_disparity = 0.001
                    disparity_src = torch.linspace(
                        start_disparity, end_disparity, S, dtype=torch.float32,
                        device=pred_device
                    ).unsqueeze(0).repeat(B, 1)

                    xyz_src_BS3HW = mpi_rendering.get_src_xyz_from_plane_disparity(
                        mpi_rendering.HomographySample(H, W, device=pred_device).meshgrid,
                        disparity_src,
                        K_inv
                    )
                    xyz_src_BS3HW_640 = mpi_rendering.get_src_xyz_from_plane_disparity(
                        mpi_rendering.HomographySample(H, W, device=pred_device).meshgrid2,
                        disparity_src,
                        K_640_inv
                    )


                    mpi_all_rgb_src = mpi_rgba[:, :, 0:3, :, :]  # BxSx3xHxW
                    mpi_all_sigma_src = mpi_rgba[:, :, 3:, :, :]  # BxSx1xHxW
                    src_imgs_syn_1, src_depth_syn_1, blend_weights, weights = mpi_rendering.render(
                        mpi_all_rgb_src,
                        mpi_all_sigma_src,
                        xyz_src_BS3HW,
                        use_alpha=True,
                    )
                    pred_nonplane_depth_hw = src_depth_syn_1[0,0]
                    pred_nonplane_depth_hw3 = self.drawDepth(pred_nonplane_depth_hw.cpu().numpy())
                    see_pred_nonplane_depth_hw3 = cv2.resize(pred_nonplane_depth_hw3,(400,300)).transpose(2, 0, 1)

                    try:
                        storage.put_image("results/{}_src_depth_nonplane".format(im_idx), see_pred_nonplane_depth_hw3)
                    except:
                        self.writer.add_image("results/{}_src_depth_nonplane".format(im_idx), see_pred_nonplane_depth_hw3, 0)


                    depth_map0_HW = pred_nonplane_depth_hw.cpu().numpy()
                    depth_from_pred_plane_hw_and_nonplane = self.plane2depth(pred_plane_para_pixel_3HW,pred_plane_mask_PHW,depth_map0_HW,H=H,W=W)
                    depth_from_pred_plane_hw3_and_nonplane = self.drawDepth(depth_from_pred_plane_hw_and_nonplane)
                    see_depth_from_pred_plane_hw3_and_nonplane = cv2.resize(depth_from_pred_plane_hw3_and_nonplane,(400,300)).transpose(2, 0, 1)
                    try: 
                        storage.put_image("results/{}_src_depth_mix".format(im_idx), see_depth_from_pred_plane_hw3_and_nonplane)
                    except:
                        self.writer.add_image("results/{}_src_depth_mix".format(im_idx), see_depth_from_pred_plane_hw3_and_nonplane, 0)


            # SAVE NUMBER EVALUATION

            # EVAL 0 target view
            # pred_tgt_RGB_3HW = results['pred_tgt_RGB_3HW']
            # pred_tgt_vmask_HW = results['pred_tgt_vmask_HW']

            # pred_tgt_RGB_3HW = torch.clamp(pred_tgt_RGB_3HW, 0., 1.)
            # gt_tgt_view_rgb_3HW = torch.clamp(gt_tgt_view_rgb_3HW, 0., 1.)
            # losses_rgb_ssim = 1 - self.ssim(pred_tgt_RGB_3HW[None,...], gt_tgt_view_rgb_3HW[None,...])

            pred_tgt_RGB_3HW = results['pred_tgt_RGB_3HW']
            pred_tgt_vmask_3HW = results['pred_tgt_vmask_HW'].float().repeat(3,1,1)
            zeros_3HW = torch.zeros(pred_tgt_RGB_3HW.shape).to(pred_tgt_RGB_3HW)

            pred_tgt_RGB_3HW = torch.where(pred_tgt_vmask_3HW>0.1,pred_tgt_RGB_3HW,zeros_3HW)            
            gt_tgt_view_rgb_3HW = torch.where(pred_tgt_RGB_3HW>0.1,gt_tgt_view_rgb_3HW,zeros_3HW)

            pred_tgt_RGB_3HW = torch.clamp(pred_tgt_RGB_3HW, 0., 1.)
            gt_tgt_view_rgb_3HW = torch.clamp(gt_tgt_view_rgb_3HW, 0., 1.)
            losses_rgb_ssim = 1 - self.ssim(pred_tgt_RGB_3HW[None,...], gt_tgt_view_rgb_3HW[None,...])
            losses_rgb_psnr = self.psnr(pred_tgt_RGB_3HW[None,...], gt_tgt_view_rgb_3HW[None,...])
            losses_rgb_lpips = self.lpips(pred_tgt_RGB_3HW[None,...], gt_tgt_view_rgb_3HW[None,...])
            
    
            # EVAL SEG
            targets_per_image = input['instances']
            gt_masks_PHW = targets_per_image.gt_masks.cpu().numpy()
            
            # EVAL 1
            plane_info = self.evaluateMasks(panoptic_img_HW, gt_masks_PHW, device=None, pred_non_plane_idx=0,printInfo=False)

            # EVAL 2
            gt_depth_from_plane = input['depth_from_plane'].cpu().numpy()
            gt_masks_HW = self.mask_nhw_to_hw(gt_masks_PHW)
            depth_pixel_recall, depth_plane_recall = self.eval_plane_recall_depth(input['file_name'], panoptic_img_HW.copy(), gt_masks_HW.copy(), pred_plane_depth_HW, gt_depth_from_plane)

            # EVAL 3
            instance_param = pred_src_plane_para_P3.cpu().numpy()
            gt_plane_instance_parameter = input['planes_paras'].cpu().numpy()
            gt_plane_instance_parameter = gt_plane_instance_parameter[:,:3] / gt_plane_instance_parameter[:,3:]
            normal_plane_recall, normal_pixel_recall = self.eval_plane_recall_normal(input['file_name'], panoptic_img_HW.copy(), gt_masks_HW.copy(), instance_param, gt_plane_instance_parameter)
            
            self._predictions.append(
                {
                    "eval_info": plane_info,
                    "depth_pixel_recall": depth_pixel_recall, 
                    "depth_plane_recall": depth_plane_recall,
                    "normal_pixel_recall": normal_pixel_recall, 
                    "normal_plane_recall": normal_plane_recall,
                    "ssim": losses_rgb_ssim.cpu().numpy(),
                    "psnr": losses_rgb_psnr.cpu().numpy(),
                    "lpips": losses_rgb_lpips.cpu().numpy(),
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
            gtSegmentations = (np.expand_dims(gtSegmentations, -1) == np.arange(gtNumPlanes)).astype(np.float32)  # H, W, gtNumPlanes
        if len(predSegmentations.shape) == 2:
            predSegmentations = (np.expand_dims(predSegmentations, -1) == np.arange(predNumPlanes)).astype(np.float32)  # H, W, predNumPlanes

        planeAreas = gtSegmentations.sum(axis=(0, 1))  # gt plane pixel number

        intersectionMask = np.expand_dims(gtSegmentations, -1) * np.expand_dims(predSegmentations, 2) > 0.5  # H, W, gtNumPlanes, predNumPlanes

        depthDiffs = gtDepths - predDepths  # H, W
        depthDiffs = depthDiffs[:, :, np.newaxis, np.newaxis]  # H, W, 1, 1

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
        :param segmentation: label map for plane segmentation [H, W] where 20 indicate non-planar
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

        # gt_param = gt_param.reshape(-1, 3)
        # print('p',param.shape, plane_num) # p (11, 3) 11
        # print('g',gt_param.shape, gt_plane_num) # g (5, 3) 5

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


        # gt_masks_PHW = []
        # if gt_non_plane_idx > 0:
        #     for i in range(gt_non_plane_idx):
        #         mask_i = gtSegmentations == i
        #         mask_i = mask_i.float()
        #         if mask_i.sum() > 0:
        #             gt_masks_PHW.append(mask_i)
        # else:
        #     assert pred_non_plane_idx == -1 or pred_non_plane_idx == 0
        #     for i in range(gt_non_plane_idx+1, 100):
        #         mask_i = gtSegmentations == i
        #         mask_i = mask_i.float()
        #         if mask_i.sum() > 0:
        #             gt_masks_PHW.append(mask_i)
        # gtMasks = torch.stack(gt_masks_PHW, dim=0)
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
        all_ssim = []
        all_psnr = []
        all_lpips = []
        pixelDepth_recall_curve = np.zeros((13))
        planeDepth_recall_curve = np.zeros((13, 3))
        pixelNorm_recall_curve = np.zeros((13))
        planeNorm_recall_curve = np.zeros((13, 3))
        for p in self._predictions:
            eval_info = p['eval_info']
            all_RI.append(eval_info[0])
            all_VOI.append(eval_info[1])
            all_SC.append(eval_info[2])
            all_ssim.append(p['ssim'])
            all_psnr.append(p['psnr'])
            all_lpips.append(p['lpips'])
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
        all_ssim = np.array(all_ssim)
        all_psnr = np.array(all_psnr)
        all_lpips = np.array(all_lpips)
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

        try:
            storage = get_event_storage()
            storage.put_scalar("val: Mean RI on %d samples"%len(self._predictions), all_RI.mean())
            storage.put_scalar("val: Mean VOI on %d samples"%len(self._predictions), all_VOI.mean())
            storage.put_scalar("val: Mean SC on %d samples"%len(self._predictions), all_SC.mean())
            storage.put_scalar("val: Mean ssim on %d samples"%len(self._predictions), all_ssim.mean())
            storage.put_scalar("val: Mean psnr on %d samples"%len(self._predictions), all_psnr.mean())
            storage.put_scalar("val: Mean lpips on %d samples"%len(self._predictions), all_lpips.mean())

            storage.put_image("stat/eval/depth_pixel", see_depth_pixel)
            storage.put_image("stat/eval/depth_plane", see_depth_plane)
            storage.put_image("stat/eval/normal_pixel", see_normal_pixel)
            storage.put_image("stat/eval/normal_plane", see_normal_plane)
        except:
            self.writer.add_scalar("val: Mean RI on %d samples"%len(self._predictions), all_RI.mean(), 0)
            self.writer.add_scalar("val: Mean VOI on %d samples"%len(self._predictions), all_VOI.mean(), 0)
            self.writer.add_scalar("val: Mean SC on %d samples"%len(self._predictions), all_SC.mean(), 0)
            self.writer.add_scalar("val: Mean ssim on %d samples"%len(self._predictions), all_ssim.mean(), 0)
            self.writer.add_scalar("val: Mean psnr on %d samples"%len(self._predictions), all_psnr.mean(), 0)
            self.writer.add_scalar("val: Mean lpips on %d samples"%len(self._predictions), all_lpips.mean(), 0)

            self.writer.add_image("stat/eval/depth_pixel", see_depth_pixel, 0)
            self.writer.add_image("stat/eval/depth_plane", see_depth_plane, 0)
            self.writer.add_image("stat/eval/normal_pixel", see_normal_pixel, 0)
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
