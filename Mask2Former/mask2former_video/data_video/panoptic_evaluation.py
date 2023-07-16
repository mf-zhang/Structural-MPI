# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import io
import itertools
import json, time
from symbol import pass_stmt
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
from typing import Optional
from PIL import Image
from tabulate import tabulate

from visualizer import TrackVisualizer

from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.structures import Instances
import torch, cv2

from detectron2.utils.events import get_event_storage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# from .evaluator import DatasetEvaluator
from detectron2.evaluation import DatasetEvaluator
import mpi_rendering
# from ssim import SSIM2
from piqa import SSIM,PSNR,LPIPS

from moviepy.editor import ImageSequenceClip

logger = logging.getLogger(__name__)

class ScanNetVideoEvaluator(DatasetEvaluator):
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

        self.use_gpu = os.environ['EVAL_USE_GPU'] == 'True'

        try:
            tmp = get_event_storage()
        except:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter('./testing_output')
            

        if self.use_gpu:
            # self.ssim = SSIM(size_average=True).cuda()
            self.ssim = SSIM().cuda()
            self.psnr = PSNR()
            self.lpips = LPIPS().cuda()
        else:
            self.ssim = SSIM()
            self.psnr = PSNR()
            self.lpips = LPIPS()

    def save_im(self, im_addr, im, resize=[384,256], save16=False, abs_addr=False):
        import cv2

        if save16 and self.save_all_im:
            im_save = im * 1000
            im_save = im_save.astype(np.uint16)
            if not abs_addr:
                im_addr = '%s/%s.png'%(self.home_addr,im_addr)
            if not os.path.exists(os.path.dirname(im_addr)):
                os.makedirs(os.path.dirname(im_addr))
            cv2.imwrite(im_addr, im_save)
            return
        
        if self.save_all_im:
            if type(im) == torch.Tensor:
                im_save = im.cpu().numpy()
            else:
                im_save = im
                
            if im.shape[0] in [3,4]:
                im_save = im_save.transpose(1,2,0)

            if not abs_addr:
                im_addr = '%s/%s.png'%(self.home_addr,im_addr)
            
            im_save = cv2.resize(im_save,resize)
            if im_save.shape[2] == 3:
                im_save = im_save[:,:,[2,1,0]]
            else:
                im_save = im_save[:,:,[2,1,0,3]]
            if not os.path.exists(os.path.dirname(im_addr)):
                os.makedirs(os.path.dirname(im_addr))
            cv2.imwrite(im_addr, im_save)

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

    def mask_hw_to_nhw(self, mask_hw, non_plane_idx=0):
        mask_nhw = []
        if non_plane_idx == 0:
            for i in range(1, np.max(mask_hw)+1):
                mask_i = mask_hw == i
                mask_i = mask_i.astype(np.float)
                if mask_i.sum() > 0:
                    mask_nhw.append(mask_i)
        if len(mask_nhw) > 0:
            mask_nhw = np.stack(mask_nhw)
        else:
            mask_nhw = np.stack([mask_hw])
        return mask_nhw
        
    def drawDepthImage(self, depth, maxDepth=5):
        depthImage = np.clip(depth / maxDepth * 255, 0, 255).astype(np.uint8)
        depthImage = cv2.applyColorMap(255 - depthImage, colormap=cv2.COLORMAP_JET)
        return depthImage

    def drawDepth(self, depth, hotmap=False):
        # assert depth.dim() == 3 or depth.dim() == 2
        # if depth.dim() == 3:
        #     depth = depth[0].cpu().numpy()
        # else:
        #     depth = depth.cpu().numpy()

        depth_color = self.drawDepthImage(depth)
        if not hotmap:
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

            # recall_planeRCNN = np.array(
            #     [0., 10.93, 20.17, 25.40, 30.58, 34.75, 36.72, 37.92, 38.84, 39.23, 39.56, 39.76, 39.93])
            # pre_defined_recalls['PlaneRCNN'] = recall_planeRCNN

            # recall_planeNet = np.array(
            #     [0., 5.78, 10.15, 12.28, 13.74, 14.79, 15.91, 16.77, 17.54, 18.02, 18.33, 18.53, 18.59])
            # pre_defined_recalls['PlaneNet'] = recall_planeNet



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

    def compute_depth_errors(self, gt, pred, var=None):
        thresh = np.maximum((gt / pred), (pred / gt))

        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_diff = np.mean(np.abs(gt - pred))

        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        err = np.log(pred) - np.log(gt)
        silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

        log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

        irmse = (1/gt - 1/pred) ** 2
        irmse = np.sqrt(irmse.mean())

        if var is not None:
            var[var < 1e-6] = 1e-6
            nll = 0.5 * (np.log(var) + np.log(2*np.pi) + (np.square(gt - pred) / var))
            nll = np.mean(nll)
        else:
            nll = 0.0

        return dict(a1=a1, a2=a2, a3=a3,
                    abs_diff=abs_diff,
                    abs_rel=abs_rel, sq_rel=sq_rel,
                    rmse=rmse, log_10=log_10, irmse=irmse,
                    rmse_log=rmse_log, silog=silog,
                    nll=nll)

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

    def path_planning(self, num_frames, x, y, z, path_type='', s=0.3):
        from scipy.interpolate import interp1d
        if path_type == 'straight-line':
            corner_points = np.array([[0, 0, 0], [(0 + x) * 0.5, (0 + y) * 0.5, (0 + z) * 0.5], [x, y, z]])
            corner_t = np.linspace(0, 1, len(corner_points))
            t = np.linspace(0, 1, num_frames)
            cs = interp1d(corner_t, corner_points, axis=0, kind='quadratic')
            spline = cs(t)
            xs, ys, zs = [xx.squeeze() for xx in np.split(spline, 3, 1)]
        elif path_type == 'double-straight-line':
            corner_points = np.array([[s*x, s*y, s*z], [-x, -y, -z]])
            corner_t = np.linspace(0, 1, len(corner_points))
            t = np.linspace(0, 1, int(num_frames*0.5))
            cs = interp1d(corner_t, corner_points, axis=0, kind='linear')
            spline = cs(t)
            xs, ys, zs = [xx.squeeze() for xx in np.split(spline, 3, 1)]
            xs = np.concatenate((xs, np.flip(xs)))
            ys = np.concatenate((ys, np.flip(ys)))
            zs = np.concatenate((zs, np.flip(zs)))
        elif path_type == 'circle':
            xs, ys, zs = [], [], []
            for frame_id, bs_shift_val in enumerate(np.arange(-2.0, 2.0, (4./num_frames))):
                xs += [np.cos(bs_shift_val * np.pi) * 1 * x]
                ys += [np.sin(bs_shift_val * np.pi) * 1 * y]
                zs += [np.cos(bs_shift_val * np.pi/2.) * 1 * z - s*z]
            xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

        return xs, ys, zs

    def traj_generation(self):
        traj_config = {}

        traj_config["fps"] = 30
        traj_config["num_frames"] = 90
        traj_config["x_shift_range"] = [0.0, -0.08]
        # traj_config["x_shift_range"] = [-0.16, -0.16]
        traj_config["y_shift_range"] = [0.0, -0.0]
        traj_config["z_shift_range"] = [-0.30, -0.2]
        # traj_config["z_shift_range"] = [-0.30, -0.30]
        traj_config["traj_types"] = ['double-straight-line', 'circle']
        traj_config["name"] = ['zoom-in', 'swing']

        tgts_poses = []
        generic_pose = np.eye(4)
        for traj_idx in range(len(traj_config['traj_types'])):
            tgt_poses = []
            sx, sy, sz = self.path_planning(traj_config['num_frames'],
                                       traj_config['x_shift_range'][traj_idx],
                                       traj_config['y_shift_range'][traj_idx],
                                       traj_config['z_shift_range'][traj_idx],
                                       path_type=traj_config['traj_types'][traj_idx])
            for xx, yy, zz in zip(sx, sy, sz):
                tgt_poses.append(generic_pose * 1.)
                tgt_poses[-1][:3, -1] = np.array([xx, yy, zz])
            tgts_poses += [tgt_poses]
        return tgts_poses, traj_config

    def process(self, inputs, outputs):

        for idx, (input, output) in enumerate(zip(inputs, outputs)):
            
            if os.environ['DEMO'] == 'True':
                if True: # prepare
                    input_scene = input['file_name'][0].split('/')[-4]
                    input_sceneidx = int(input_scene[6:9])
                    input_sceneidx2 = int(input_scene[10:])
                    input_imidx = int(input['file_name'][0].split('/')[-1][:-4])
                    im_idx0 = 1000000000 + input_sceneidx * 1000000 + input_sceneidx2 * 10000 + input_imidx
                    panoptic_img_HWs, segments_infos = output["panoptic_seg"]
                    device_type = segments_infos[0][0]['mask_orig'].float()

                    poses, poses_config = self.traj_generation()
                    for ri in range(2): # zoom and swing
                        print('generating video %d...'%ri)
                        tgt_img_np_list = []
                        tgt_disp_np_list = []
                        num_frame = len(poses[ri])
                        for vfi in range(num_frame):
                            if os.environ['SAMPLING_FRAME_NUM'] == '2':
                                cam_pose0 = input['cam_pose'][0].to(device_type)
                                cam_pose1 = input['cam_pose'][1].to(device_type)
                                gt_G_src_tgt_44 = torch.inverse(torch.tensor(poses[ri][vfi]).to(device_type))

                                new_tgt_cam_pose = torch.linalg.inv(gt_G_src_tgt_44) @ cam_pose0
                                gt_G_src_tgt_44 = cam_pose0 @ torch.linalg.inv(new_tgt_cam_pose.to(device_type))
                                gt_G_src_tgt_44_2 = cam_pose1 @ torch.linalg.inv(new_tgt_cam_pose.to(device_type))
                            else:
                                gt_G_src_tgt_44 = torch.tensor(poses[ri][vfi]).to(device_type)

                            if True: # prepare 1 data
                                fi = 0
                                panoptic_img_HW, segments_info = panoptic_img_HWs[fi], segments_infos[fi]
                                H, W = panoptic_img_HW.shape
                                self.K_inv_dot_xy_1 = input['K_inv_dot_xy_1'].cpu().numpy()
                                pred_src_plane_para_P3 = torch.stack([x['plane_para'] for x in segments_info]).to(device_type)
                                pred_class_P                = torch.stack([torch.tensor(not(x['isplane']))*50 for x in segments_info]).to(device_type)
                                pred_plane_mask_orig_PHW    = torch.stack([x['mask_orig'] for x in segments_info]).to(device_type)
                                pred_src_plane_RGBA_P4hw    = torch.stack([x['plane_RGBA'] for x in segments_info]).sigmoid().to(device_type)
                                gt_src_view_rgb_3HW         = (input['image'][fi]/255.).to(device_type)
                                gt_src_view_depth_map_hw    = input['depth_map'][fi].to(device_type)
                                gt_tgt_view_rgb_3HW         = (input['tgt_image']/255.).to(device_type)
                                gt_tgt_view_depth_map_HW    = input['tgt_depth_map'].to(device_type)
                                # gt_G_src_tgt_44             = input['tgt_G_src_tgt'][fi].to(device_type)
                                K_inv_dot_xy_1_3HxW         = input['K_inv_dot_xy_1'].view(3,-1).to(device_type)
                                K_inv_dot_xy_1              = [K_inv_dot_xy_1_3HxW]
                                use_src_view_rgb = os.environ['USE_ORIG_RGB'] == 'True'     
                            
                            results, vis_results = mpi_rendering.render_everything(pred_plane_mask_orig_PHW, pred_src_plane_RGBA_P4hw, pred_src_plane_para_P3, 
                                                                                        None, gt_G_src_tgt_44, gt_src_view_rgb_3HW, K_inv_dot_xy_1,
                                                                                        device_type, True, use_src_view_rgb=use_src_view_rgb, pred_class_P=pred_class_P)
                            
                            if os.environ['SAMPLING_FRAME_NUM'] == '1':
                                pred_tgt_depth_HW = results['pred_tgt_depth_HW']
                                pred_tgt_depth_HW3 = self.drawDepth(pred_tgt_depth_HW.detach().cpu().numpy())
                                # see_pred_tgt_depth_3HW = pred_tgt_depth_HW3.transpose(2, 0, 1)

                                pred_tgt_RGB_3HW = results['pred_tgt_RGB_3HW'].float()
                                see_pred_tgt_RGB_HW3 = (pred_tgt_RGB_3HW*255).type(torch.uint8).cpu().numpy().transpose(1,2,0) #[:,:,[2,1,0]]

                                tgt_img_np_list.append(see_pred_tgt_RGB_HW3)
                                tgt_disp_np_list.append(pred_tgt_depth_HW3)


                            if os.environ['SAMPLING_FRAME_NUM'] == '2':
                                # # src tgt sm_G -> tm_G
                                # # sm_G^-1 @ src = tgt_m
                                # # tm_G = tgt @ tgt_m^-1

                                gt_G_src_tgt_44 = gt_G_src_tgt_44_2


                                # G = campose0 @ inv(new_tgt_cam)
                                # G @ new_tgt_cam = campose0
                                # new_tgt_cam = inv(G) @ campose0

                                fi = 1
                                panoptic_img_HW, segments_info = panoptic_img_HWs[fi], segments_infos[fi]
                                H, W = panoptic_img_HW.shape
                                self.K_inv_dot_xy_1 = input['K_inv_dot_xy_1'].cpu().numpy()
                                pred_src_plane_para_P3 = torch.stack([x['plane_para'] for x in segments_info]).to(device_type)
                                pred_class_P                = torch.stack([torch.tensor(not(x['isplane']))*50 for x in segments_info]).to(device_type)
                                pred_plane_mask_orig_PHW    = torch.stack([x['mask_orig'] for x in segments_info]).to(device_type)
                                pred_src_plane_RGBA_P4hw    = torch.stack([x['plane_RGBA'] for x in segments_info]).sigmoid().to(device_type)
                                gt_src_view_rgb_3HW         = (input['image'][fi]/255.).to(device_type)
                                gt_src_view_depth_map_hw    = input['depth_map'][fi].to(device_type)
                                gt_tgt_view_rgb_3HW         = (input['tgt_image']/255.).to(device_type)
                                gt_tgt_view_depth_map_HW    = input['tgt_depth_map'].to(device_type)
                                # gt_G_src_tgt_44             = input['tgt_G_src_tgt'][fi].to(device_type)
                                K_inv_dot_xy_1_3HxW         = input['K_inv_dot_xy_1'].view(3,-1).to(device_type)
                                K_inv_dot_xy_1              = [K_inv_dot_xy_1_3HxW]
                                use_src_view_rgb = os.environ['USE_ORIG_RGB'] == 'True'


                                results2, vis_results2 = mpi_rendering.render_everything(pred_plane_mask_orig_PHW, pred_src_plane_RGBA_P4hw, pred_src_plane_para_P3, 
                                                                                        None, gt_G_src_tgt_44, gt_src_view_rgb_3HW, K_inv_dot_xy_1,
                                                                                        device_type, True, use_src_view_rgb=use_src_view_rgb, pred_class_P=pred_class_P)

                                render_tgt_RGBs = []
                                render_tgt_Ds = []
                                render_tgt_As = []

                                render_tgt_RGBs.append(results['pred_tgt_RGB_3HW'])
                                render_tgt_Ds.append(results['pred_tgt_depth_HW'])
                                render_tgt_As.append(results['pred_tgt_alpha_acc_1HW'])
                                render_tgt_RGBs.append(results2['pred_tgt_RGB_3HW'])
                                render_tgt_Ds.append(results2['pred_tgt_depth_HW'])
                                render_tgt_As.append(results2['pred_tgt_alpha_acc_1HW'])

                                render_tgt_RGBs_T3HW = torch.stack(render_tgt_RGBs,dim=0)
                                render_tgt_As_T1HW = torch.stack(render_tgt_As,dim=0)
                                render_tgt_Ds_T1HW = torch.stack(render_tgt_Ds,dim=0)[:,None,:,:]

                                T,_,H,W = render_tgt_RGBs_T3HW.shape
                                weights = torch.ones([T,1,H,W], device=render_tgt_RGBs_T3HW.device)

                                hard_alpha = False
                                if hard_alpha:
                                    for ti in range(T):
                                        render_tgt_As_T1HW[ti,0] = (render_tgt_RGBs_T3HW[ti][1] > 0.2).float()
                                
                                scale_brightness = True
                                if scale_brightness:
                                    overlap_A_1HW = torch.logical_and(render_tgt_As_T1HW[0],render_tgt_As_T1HW[1])
                                    overlap_A_1HW = overlap_A_1HW.repeat(3,1,1).to(torch.bool)
                                    scale0 = torch.sum(render_tgt_RGBs_T3HW[0][overlap_A_1HW])
                                    scale1 = torch.sum(render_tgt_RGBs_T3HW[1][overlap_A_1HW])
                                    render_tgt_RGBs_T3HW[1] *= (scale0/scale1)

                                erode_black_line = True
                                if erode_black_line:
                                    # cv2.imwrite('./zzz_a1.png',render_tgt_As_T1HW[0][0].cpu().numpy()*255)
                                    # cv2.imwrite('./zzz_a2.png',render_tgt_As_T1HW[1][0].cpu().numpy()*255)
                                    ksize = 10
                                    ksize = int(ksize) if int(ksize) % 2 == 1 else int(ksize)+1
                                    max_pool = torch.nn.MaxPool2d(kernel_size=ksize, stride=1, padding=int((ksize-1)/2))
                                    render_tgt_As_T1HW = -max_pool(-render_tgt_As_T1HW[None,...][0])
                                    # cv2.imwrite('./zzz_a11.png',render_tgt_As_T1HW[0][0].cpu().numpy()*255)
                                    # cv2.imwrite('./zzz_a22.png',render_tgt_As_T1HW[1][0].cpu().numpy()*255)
                                    render_tgt_RGBs_T3HW *= render_tgt_As_T1HW.repeat(1,3,1,1)
                                    render_tgt_Ds_T1HW   *= render_tgt_As_T1HW.repeat(1,1,1,1)

                                out_frame = (render_tgt_RGBs_T3HW*weights).sum(0) / (1e-10+(render_tgt_As_T1HW*weights).sum(0))
                                out_depth = (render_tgt_Ds_T1HW  *weights).sum(0) / (1e-10+(render_tgt_As_T1HW*weights).sum(0))

                                out_frame = torch.clip(out_frame,0.,1.)

                                see_out_frame = (out_frame*255).type(torch.uint8).cpu().numpy().transpose(1,2,0) #[:,:,[2,1,0]]

                                out_depth_HW = out_depth[0]
                                out_depth_HW3 = self.drawDepth(out_depth_HW.detach().cpu().numpy())

                                tgt_img_np_list.append(see_out_frame)
                                tgt_disp_np_list.append(out_depth_HW3)

                                continue
                                        

                        rgb_clip = ImageSequenceClip(tgt_img_np_list, fps=poses_config["fps"])
                        video_rgb_addr = './testing_output/f%s_%d_rgb_%d.mp4'%(os.environ['SAMPLING_FRAME_NUM'], im_idx0, ri)
                        video_depth_addr = './testing_output/f%s_%d_depth_%d.mp4'%(os.environ['SAMPLING_FRAME_NUM'], im_idx0, ri)
                        if not os.path.exists(os.path.dirname(video_rgb_addr)):
                            os.makedirs(os.path.dirname(video_rgb_addr))
                        rgb_clip.write_videofile(video_rgb_addr,
                                                fps=poses_config["fps"],
                                                verbose=False,
                                                logger=None)
                        disp_clip = ImageSequenceClip(tgt_disp_np_list, fps=poses_config["fps"])
                        disp_clip.write_videofile(video_depth_addr,
                                                fps=poses_config["fps"],
                                                verbose=False,
                                                logger=None)
                    exit()

            #################################

            if os.environ['SAMPLING_FRAME_NUM'] == '2':
                trans_s1_s2 = input['cam_pose'][0].cpu().numpy() @ np.linalg.inv(input['cam_pose'][1].cpu().numpy())
                dis2 = trans_s1_s2[0,3]*trans_s1_s2[0,3] + trans_s1_s2[1,3]*trans_s1_s2[1,3] + trans_s1_s2[2,3]*trans_s1_s2[2,3]
                two_src_distance = np.sqrt(dis2)
            else:
                two_src_distance = 0
          
            # print(os.path.basename(input['file_name'][0]),os.path.basename(input['file_name'][1]),os.path.basename(input['tgt_filename']))
            input_scene = input['tgt_filename'].split('/')[-4]
            input_sceneidx = int(input_scene[6:9])
            input_sceneidx2 = int(input_scene[10:])
            input_imidx = int(input['tgt_filename'].split('/')[-1][:-4])

            im_idx0 = 1000000000 + input_sceneidx * 1000000 + input_sceneidx2 * 10000 + input_imidx

            if   os.environ['DATA_MAX_GAP'] == '40': # 263
                view_gap = 10
            elif os.environ['DATA_MAX_GAP'] == '30': # 177
                view_gap = 7
            elif os.environ['DATA_MAX_GAP'] == '20': # 177
                view_gap = 5
            elif os.environ['DATA_MAX_GAP'] == '10': # 97
                view_gap = 3
            elif os.environ['DATA_MAX_GAP'] == '5':  # 47
                view_gap = 2
            else:
                assert False

            self.save_all_im = False
            if self.save_all_im:
                self.home_addr = '/home/v-mingfzhang/exp_im/ours-scannet-f%s-gap%s-gap%s/'%(os.environ['SAMPLING_FRAME_NUM'],os.environ['TGT_IM_GAP'],os.environ['DATA_MAX_GAP'])

            if im_idx0 % view_gap == 0 or self.save_all_im:
                save_images = True
            else:
                save_images = False

            # render_wo_nonplane = False
            plane_understand_only = os.environ['TRAIN_PHASE'] in ['0','1']

            panoptic_img_HWs, segments_infos = output["panoptic_seg"]

            if self.use_gpu or self.save_all_im:
                device_type = segments_infos[0][0]['mask_orig'].float()
            else:
                device_type = torch.zeros([1]).float()

            if os.environ['LLFF_BLEND'] == 'True':
                render_tgt_RGBs = []
                render_tgt_As = []
                render_tgt_Ds = []

            for fi in range(len(panoptic_img_HWs)):
                # ttt01 = time.process_time()
                
                # PREPARE EVALUATION ITEMS

                # 1 segmentation
                if True:
                    panoptic_img_HW, segments_info = panoptic_img_HWs[fi], segments_infos[fi]
                    panoptic_img_torch_HW = panoptic_img_HW.cpu()
                    panoptic_img_HW = panoptic_img_torch_HW.numpy()
                    H, W = panoptic_img_HW.shape

                    if len(segments_info) == 0:
                        print('zmf: this eval sample has no predicted plane.')
                        continue
                    if input['planes_paras'][fi].shape[0] < 1:
                        print('zmf: this eval sample has no gt plane.',input['file_name'])
                        continue
                
                # 2 depth from plane
                if not os.environ['ONLY_SEG'] == 'True':
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


                                        
                # 3 target view 'results' 'vis_results'
                if not plane_understand_only:
                    # prepare orig data
                    pred_class_P                = torch.stack([torch.tensor(not(x['isplane']))*50 for x in segments_info]).to(device_type)
                    pred_plane_mask_orig_PHW    = torch.stack([x['mask_orig'] for x in segments_info]).to(device_type)
                    pred_src_plane_RGBA_P4hw          = torch.stack([x['plane_RGBA'] for x in segments_info]).sigmoid().to(device_type)

                    gt_src_view_rgb_3HW         = (input['image'][fi]/255.).to(device_type)
                    gt_src_view_depth_map_hw    = input['depth_map'][fi].to(device_type)
                    gt_tgt_view_rgb_3HW         = (input['tgt_image']/255.).to(device_type)
                    gt_tgt_view_depth_map_HW    = input['tgt_depth_map'].to(device_type)
                    gt_G_src_tgt_44             = input['tgt_G_src_tgt'][fi].to(device_type)
                    K_inv_dot_xy_1_3HxW         = input['K_inv_dot_xy_1'].view(3,-1).to(device_type)
                    K_inv_dot_xy_1              = [K_inv_dot_xy_1_3HxW]


                    use_src_view_rgb = os.environ['USE_ORIG_RGB'] == 'True'
                    results, vis_results = mpi_rendering.render_everything(pred_plane_mask_orig_PHW, pred_src_plane_RGBA_P4hw, pred_src_plane_para_P3, 
                                                                            None, gt_G_src_tgt_44, gt_src_view_rgb_3HW, K_inv_dot_xy_1,
                                                                            device_type, save_images, use_src_view_rgb=use_src_view_rgb, pred_class_P=pred_class_P)

                    if os.environ['LLFF_BLEND'] == 'True':
                        render_tgt_RGBs.append(results['pred_tgt_RGB_3HW'])
                        render_tgt_Ds.append(results['pred_tgt_depth_HW'])
                        render_tgt_As.append(results['pred_tgt_alpha_acc_1HW'])

                # ttt02 = time.process_time()

                # SAVE VARIOUS IMAGES
                if save_images:
                    
                    # 1 src planes_RGB, render_depth, render_rgb
                    #   tgt render_depth, render_rgb
                    if not plane_understand_only:
                        # 0 target view
                        P,_, H, W = pred_src_plane_RGBA_P4hw.shape

                        # # src planes
                        # for pi in range(P):
                        #     this_plane_RGBA_4HW = vis_results['pred_src_plane_RGBA_P4HW'][pi]
                        #     see_this_plane_pred_RGBA_4HW = (this_plane_RGBA_4HW*255).cpu().numpy().astype(np.uint8) # [:,:,[2,1,0,3]]
                        #     try:
                        #         storage = get_event_storage()
                        #         storage.put_image("{}/plane_{}_{}".format(im_idx0,pi,fi), see_this_plane_pred_RGBA_4HW)
                        #     except:
                        #         self.writer.add_image("{}/plane_{}_{}".format(im_idx0,pi,fi), see_this_plane_pred_RGBA_4HW, 0)
                        #         self.save_im("{}/plane_{}_{}".format(im_idx0,pi,fi), see_this_plane_pred_RGBA_4HW)

                        # src depth
                        if True:
                            pred_src_depth_HW = results['pred_src_depth_HW']
                            pred_src_depth_HW3 = self.drawDepth(pred_src_depth_HW.detach().cpu().numpy())
                            see_pred_src_depth_3HW = pred_src_depth_HW3.transpose(2, 0, 1)
                            # gt_src_view_depth_map_HW3 = self.drawDepth(gt_src_view_depth_map_hw.detach().cpu().numpy())
                            # see_gt_src_view_depth_map_3HW = gt_src_view_depth_map_HW3.transpose(2, 0, 1)
                            try:
                                storage.put_image("results/{}_src_depth_{}".format(im_idx0,fi), see_pred_src_depth_3HW)
                                # storage.put_image("results/{}_src_depth_gt_{}".format(im_idx0,fi), see_gt_src_view_depth_map_3HW)
                            except:
                                self.writer.add_image("results/{}_src_depth_{}".format(im_idx0,fi), see_pred_src_depth_3HW, 0)
                                # self.writer.add_image("results/{}_src_depth_gt_{}".format(im_idx0,fi), see_gt_src_view_depth_map_3HW, 0)
                                self.save_im("results/{}_src_depth_{}".format(im_idx0,fi), see_pred_src_depth_3HW)

                        # src rgb
                        if True:
                            pred_src_RGB_3HW = results['pred_src_RGB_3HW']
                            see_pred_src_RGB_3HW = (pred_src_RGB_3HW*255).type(torch.uint8) #.cpu().numpy() #[:,:,[2,1,0]]
                            # see_gt = (gt_src_view_rgb_3HW*255.).type(torch.uint8) #.cpu().numpy() #[:,:,[2,1,0]]
                            try:
                                storage.put_image("results/{}_src_im_pred_{}".format(im_idx0,fi), see_pred_src_RGB_3HW)
                                # storage.put_image("results/{}_src_im_gt_{}".format(im_idx0,fi), see_gt)
                            except:
                                self.writer.add_image("results/{}_src_im_pred_{}".format(im_idx0,fi), see_pred_src_RGB_3HW, 0)
                                # self.writer.add_image("results/{}_src_im_gt_{}".format(im_idx0,fi), see_gt, 0)
                                self.save_im("results/{}_src_im_pred_{}".format(im_idx0,fi), see_pred_src_RGB_3HW)
                
                        # tgt depth
                        if True:
                            pred_tgt_depth_HW = results['pred_tgt_depth_HW']
                            pred_tgt_depth_HW3 = self.drawDepth(pred_tgt_depth_HW.detach().cpu().numpy())
                            see_pred_tgt_depth_3HW = pred_tgt_depth_HW3.transpose(2, 0, 1)
                            gt_tgt_view_depth_map_HW3 = self.drawDepth(gt_tgt_view_depth_map_HW.detach().cpu().numpy())
                            see_gt_tgt_view_depth_map_3HW = gt_tgt_view_depth_map_HW3.transpose(2, 0, 1)
                            try:
                                storage.put_image("results/{}_tgt_depth_{}".format(im_idx0,fi), see_pred_tgt_depth_3HW)
                                storage.put_image("results/{}_tgt_depth_gt_{}".format(im_idx0,fi), see_gt_tgt_view_depth_map_3HW)
                            except:
                                self.writer.add_image("results/{}_tgt_depth_{}".format(im_idx0,fi), see_pred_tgt_depth_3HW, 0)
                                self.writer.add_image("results/{}_tgt_depth_gt_{}".format(im_idx0,fi), see_gt_tgt_view_depth_map_3HW, 0)
                                self.save_im("results/{}_tgt_depth_{}".format(im_idx0,fi), see_pred_tgt_depth_3HW)
                                self.save_im("results/{}_tgt_depth_gt_{}".format(im_idx0,fi), see_gt_tgt_view_depth_map_3HW)

                        # tgt rgb & mask
                        if True:
                            pred_tgt_RGB_3HW = results['pred_tgt_RGB_3HW'].float()
                            pred_tgt_vmask_3HW = results['pred_tgt_vmask_HW'].float()[None,:,:].repeat(3,1,1)
                            zeros_3HW = torch.zeros(pred_tgt_RGB_3HW.shape).to(pred_tgt_RGB_3HW)
                            if not self.save_all_im:
                                pred_tgt_RGB_3HW = torch.where(pred_tgt_vmask_3HW>0.05,pred_tgt_RGB_3HW,zeros_3HW)
                            see_pred_tgt_RGB_3HW = (pred_tgt_RGB_3HW*255).type(torch.uint8) #.cpu().numpy() #[:,:,[2,1,0]]
                            if not self.save_all_im:
                                gt_tgt_view_rgb_3HW = torch.where(pred_tgt_RGB_3HW>0.05,gt_tgt_view_rgb_3HW,zeros_3HW)
                            see_gt = (gt_tgt_view_rgb_3HW*255.).type(torch.uint8) #.cpu().numpy() #[:,:,[2,1,0]]

                            pred_tgt_vmask_3HW = pred_tgt_vmask_3HW / torch.max(pred_tgt_vmask_3HW)
                            see_mask = (pred_tgt_vmask_3HW*255.).type(torch.uint8) #.cpu().numpy()
                            try:
                                storage.put_image("results/{}_tgt_im_pred_{}".format(im_idx0,fi), see_pred_tgt_RGB_3HW)
                                storage.put_image("results/{}_tgt_im_gt_{}".format(im_idx0,fi), see_gt)
                                storage.put_image("results/{}_tgt_mask_{}".format(im_idx0,fi), see_mask)
                            except:
                                self.writer.add_image("results/{}_tgt_im_pred_{}".format(im_idx0,fi), see_pred_tgt_RGB_3HW, 0)
                                self.writer.add_image("results/{}_tgt_im_gt_{}".format(im_idx0,fi), see_gt, 0)
                                self.writer.add_image("results/{}_tgt_mask_{}".format(im_idx0,fi), see_mask, 0)
                                self.save_im("results/{}_tgt_im_pred_{}".format(im_idx0,fi), see_pred_tgt_RGB_3HW)
                                self.save_im("results/{}_tgt_im_gt_{}".format(im_idx0,fi), see_gt)
                                self.save_im("results/{}_tgt_mask_{}".format(im_idx0,fi), see_mask)
                
                    # 2 segmentation
                    if True:
                        src_image_HW3 = input['image'][fi].permute(1,2,0).cpu().numpy()
                        metadata = MetadataCatalog.get("scannet_val_video")
                        show_pred_seg = Visualizer(src_image_HW3, metadata, instance_mode=ColorMode.IMAGE)
                        show_pred_seg = show_pred_seg.draw_panoptic_seg_predictions(panoptic_img_torch_HW, segments_info)
                        
                        see_pred_seg_3HW = show_pred_seg.get_image()
                        see_pred_seg_3HW = cv2.resize(see_pred_seg_3HW,(int(300/H*W),300)).transpose(2, 0, 1)

                        try:
                            storage = get_event_storage()
                            storage.put_image("results/{}_src_seg_{}".format(im_idx0,fi), see_pred_seg_3HW)
                            storage.put_image("results/{}_src_im_gt_{}".format(im_idx0,fi), input['image'][fi])
                        except:
                            self.writer.add_image("results/{}_src_seg_{}".format(im_idx0,fi), see_pred_seg_3HW, 0)
                            self.writer.add_image("results/{}_src_im_gt_{}".format(im_idx0,fi), input['image'][fi], 0)
                            self.save_im("results/{}_src_seg_{}".format(im_idx0,fi), see_pred_seg_3HW)
                            self.save_im("results/{}_src_im_gt_{}".format(im_idx0,fi), input['image'][fi])

                    # 1 save segmentation gt
                    if True:
                        # src_image_HW3 = input['image'][fi].permute(1,2,0).cpu().numpy()
                        # metadata = MetadataCatalog.get("scannet_val_video")
                        show_pred_seg = Visualizer(src_image_HW3, metadata, instance_mode=ColorMode.IMAGE)
                        gt_panoptic_img_torch_HW = torch.from_numpy(self.mask_nhw_to_hw(input['instances'][fi].gt_masks.tensor.cpu().numpy()))
                        gt_segments_info = []
                        for si in range(1,int(torch.max(gt_panoptic_img_torch_HW)+1)):
                            seg_dic = {}
                            seg_dic['id'] = si
                            seg_dic['isthing'] = True
                            # print(input['nonplane_num'][fi])
                            # print(torch.max(gt_panoptic_img_torch_HW) - input['nonplane_num'][fi])
                            if os.environ['SEG_NONPLANE'] == 'True':
                                isplane = si <= (torch.max(gt_panoptic_img_torch_HW) - input['nonplane_num'][fi])
                            else:
                                isplane = True
                            seg_dic['category_id'] = 0 if isplane else 51
                            gt_segments_info.append(seg_dic)
                        show_pred_seg = show_pred_seg.draw_panoptic_seg_predictions(gt_panoptic_img_torch_HW, gt_segments_info)
                        
                        see_pred_seg_3HW = show_pred_seg.get_image()
                        see_pred_seg_3HW = cv2.resize(see_pred_seg_3HW,(int(300/H*W),300)).transpose(2, 0, 1)

                        try:
                            storage = get_event_storage()
                            storage.put_image("results/{}_src_seg_gt_{}".format(im_idx0,fi), see_pred_seg_3HW)
                        except:
                            self.writer.add_image("results/{}_src_seg_gt_{}".format(im_idx0,fi), see_pred_seg_3HW, 0)
                            self.save_im("results/{}_src_seg_gt_{}".format(im_idx0,fi), see_pred_seg_3HW)

                    if not os.environ['ONLY_SEG'] == 'True':
                        # 2 save plane depth
                        if True:
                            see_pred_plane_depth_3HW = cv2.resize(pred_plane_depth_HW3,(int(300/H*W),300)).transpose(2, 0, 1)
                            try:
                                storage.put_image("results/{}_src_depth_from_plane_{}".format(im_idx0,fi), see_pred_plane_depth_3HW)
                            except:
                                self.writer.add_image("results/{}_src_depth_from_plane_{}".format(im_idx0,fi), see_pred_plane_depth_3HW, 0)
                                self.save_im("results/{}_src_depth_from_plane_{}".format(im_idx0,fi), see_pred_plane_depth_3HW)
                                self.save_im("results/{}_src_depth_from_plane_{}_16".format(im_idx0,fi), pred_plane_depth_HW, save16=True)
                        
                        # 3 save gt depth
                        if True:                            
                            gt_depth_map = input['depth_map'][fi].cpu().numpy()
                            gt_depth_HW3 = self.drawDepth(gt_depth_map)
                            see_gt_depth_3HW = cv2.resize(gt_depth_HW3,(int(300/H*W),300)).transpose(2, 0, 1)
                            try:
                                storage.put_image("results/{}_src_depth_gt_{}".format(im_idx0,fi), see_gt_depth_3HW)
                            except:
                                self.writer.add_image("results/{}_src_depth_gt_{}".format(im_idx0,fi), see_gt_depth_3HW, 0)
                                self.save_im("results/{}_src_depth_gt_{}".format(im_idx0,fi), see_gt_depth_3HW)
                                self.save_im("results/{}_src_depth_gt_{}_16".format(im_idx0,fi), gt_depth_map, save16=True)

                # ttt03 = time.process_time()
                if os.environ['ONLY_SEG'] == 'True':
                    continue
                
                # SAVE NUMBER EVALUATION

                # ssim psnr lpips
                if not plane_understand_only:
                    pred_tgt_RGB_3HW = results['pred_tgt_RGB_3HW'].float()
                    pred_tgt_vmask_3HW = results['pred_tgt_vmask_HW'].float()[None,:,:].repeat(3,1,1)
                    zeros_3HW = torch.zeros(pred_tgt_RGB_3HW.shape).to(pred_tgt_RGB_3HW)

                    pred_tgt_RGB_3HW = torch.where(pred_tgt_vmask_3HW>0.05,pred_tgt_RGB_3HW,zeros_3HW)            
                    gt_tgt_view_rgb_3HW = torch.where(pred_tgt_vmask_3HW>0.05,gt_tgt_view_rgb_3HW,zeros_3HW)
                    # gt_tgt_view_rgb_3HW = torch.where(pred_tgt_RGB_3HW>0.05,gt_tgt_view_rgb_3HW,zeros_3HW)

                    pred_tgt_RGB_3HW = torch.clamp(pred_tgt_RGB_3HW, 1e-5, 1.-1e-5)
                    gt_tgt_view_rgb_3HW = torch.clamp(gt_tgt_view_rgb_3HW, 1e-5, 1.-1e-5)
                    losses_rgb_ssim = self.ssim(pred_tgt_RGB_3HW[None,...], gt_tgt_view_rgb_3HW[None,...])
                    losses_rgb_psnr = self.psnr(pred_tgt_RGB_3HW[None,...], gt_tgt_view_rgb_3HW[None,...])
                    losses_rgb_lpips = self.lpips(pred_tgt_RGB_3HW[None,...], gt_tgt_view_rgb_3HW[None,...])
                    if torch.isnan(losses_rgb_lpips):
                        print('nan in lpips',losses_rgb_ssim,losses_rgb_psnr)
                        losses_rgb_ssim = torch.tensor(0.).to(device_type)
                        losses_rgb_psnr = torch.tensor(0.).to(device_type)
                        losses_rgb_lpips = torch.tensor(0.).to(device_type)

                # depth number eval
                if not plane_understand_only:
                    pred_src_depth_HW = results['pred_src_depth_HW'].cpu().numpy()
                    gt_src_depth_HW = gt_src_view_depth_map_hw.cpu().numpy()
                    pred_tgt_depth_HW = results['pred_tgt_depth_HW'].cpu().numpy()
                    gt_tgt_depth_HW = gt_tgt_view_depth_map_HW.cpu().numpy()
                    pred_tgt_vmask_HW = results['pred_tgt_vmask_HW'].cpu().numpy()

                    min_depth = 1e-3
                    max_depth = 10

                    src_valid_mask = np.logical_and(gt_src_depth_HW > min_depth, gt_src_depth_HW < max_depth)
                    tgt_valid_mask = np.logical_and(gt_tgt_depth_HW > min_depth, gt_tgt_depth_HW < max_depth)
                    tgt_valid_mask = np.logical_and(tgt_valid_mask, pred_tgt_vmask_HW)

                    pred_src_depth_HW[pred_src_depth_HW < min_depth] = min_depth
                    pred_src_depth_HW[pred_src_depth_HW > max_depth] = max_depth
                    pred_src_depth_HW[np.isinf(pred_src_depth_HW)] = max_depth
                    pred_src_depth_HW[np.isnan(pred_src_depth_HW)] = min_depth

                    pred_tgt_depth_HW[pred_tgt_depth_HW < min_depth] = min_depth
                    pred_tgt_depth_HW[pred_tgt_depth_HW > max_depth] = max_depth
                    pred_tgt_depth_HW[np.isinf(pred_tgt_depth_HW)] = max_depth
                    pred_tgt_depth_HW[np.isnan(pred_tgt_depth_HW)] = min_depth

                    src_depth_metrics = self.compute_depth_errors(gt_src_depth_HW[src_valid_mask],pred_src_depth_HW[src_valid_mask])
                    tgt_depth_metrics = self.compute_depth_errors(gt_tgt_depth_HW[tgt_valid_mask],pred_tgt_depth_HW[tgt_valid_mask])


                # EVAL SEG
                targets_per_image = input['instances'][fi]
                gt_masks_PHW = targets_per_image.gt_masks.tensor.cpu().numpy()

                # delete nonplane
                if os.environ['SEG_NONPLANE'] == 'True':
                    if input['nonplane_num'][fi] > 0:
                        gt_masks_PHW = gt_masks_PHW[:-input['nonplane_num'][fi]]       

                    zeromap = np.zeros(panoptic_img_HW.shape)
                    for seg in segments_info:
                        pid = seg['id']
                        if not seg['isplane']:
                            panoptic_img_HW[panoptic_img_HW==pid] = zeromap[panoptic_img_HW==pid]
                    panoptic_img_PHW = self.mask_hw_to_nhw(panoptic_img_HW) # to deal with num jumping
                    panoptic_img_HW = self.mask_nhw_to_hw(panoptic_img_PHW)

                    pred_src_plane_para_P3 = torch.stack([x['plane_para'] for x in segments_info]).to(device_type)
                    pred_isplane = [x['isplane'] for x in segments_info]
                    pred_src_plane_para_P3 = pred_src_plane_para_P3[pred_isplane]

                    if pred_src_plane_para_P3.shape[0] == 0 or panoptic_img_PHW.shape[0] == 0:
                        continue
                    pred_plane_para_pixel_3HW = self.get_plane_parameters(pred_src_plane_para_P3.cpu().numpy(),panoptic_img_PHW)
                    depth_map0_HW = np.zeros([H, W])
                    pred_plane_depth_HW = self.plane2depth(pred_plane_para_pixel_3HW,panoptic_img_PHW,depth_map0_HW,H=H,W=W)                    

                # ttt04 = time.process_time()
                # EVAL 1
                plane_info = self.evaluateMasks(panoptic_img_HW, gt_masks_PHW, device=device_type.device, pred_non_plane_idx=0,printInfo=False)
                
                # EVAL 2
                gt_depth_from_plane = input['depth_from_plane'][fi].cpu().numpy()
                gt_masks_HW = self.mask_nhw_to_hw(gt_masks_PHW)
                depth_pixel_recall, depth_plane_recall = self.eval_plane_recall_depth(input['file_name'], panoptic_img_HW.copy(), gt_masks_HW.copy(), pred_plane_depth_HW, gt_depth_from_plane)
                depth_AP = self.eval_plane_ap_depth(input['file_name'], panoptic_img_HW.copy(), gt_masks_HW.copy(), pred_plane_depth_HW, gt_depth_from_plane)

                # EVAL 3
                instance_param = pred_src_plane_para_P3.cpu().numpy()
                gt_plane_instance_parameter = input['planes_paras'][fi].cpu().numpy()
                if os.environ['SEG_NONPLANE'] == 'True' and input['nonplane_num'][fi] > 0:
                    gt_plane_instance_parameter = gt_plane_instance_parameter[:-input['nonplane_num'][fi]]
                gt_plane_instance_parameter = gt_plane_instance_parameter[:,:3] / gt_plane_instance_parameter[:,3:]
                normal_plane_recall, normal_pixel_recall = self.eval_plane_recall_normal(input['file_name'], panoptic_img_HW.copy(), gt_masks_HW.copy(), instance_param, gt_plane_instance_parameter)
                
                if plane_understand_only:
                    losses_rgb_ssim = torch.tensor(0.)
                    losses_rgb_psnr = torch.tensor(0.)
                    losses_rgb_lpips = torch.tensor(0.)
                    src_depth_metrics = {}
                    src_depth_metrics['abs_rel'] = 0.
                    src_depth_metrics['log_10'] = 0.
                    src_depth_metrics['rmse'] = 0.
                    src_depth_metrics['a1'] = 0.
                    src_depth_metrics['a2'] = 0.
                    src_depth_metrics['a3'] = 0.
                    tgt_depth_metrics = {}
                    tgt_depth_metrics['abs_rel'] = 0.
                    tgt_depth_metrics['log_10'] = 0.
                    tgt_depth_metrics['rmse'] = 0.
                    tgt_depth_metrics['a1'] = 0.
                    tgt_depth_metrics['a2'] = 0.
                    tgt_depth_metrics['a3'] = 0.


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
                        "src_abs_rel":src_depth_metrics['abs_rel'],
                        "src_log_10":src_depth_metrics['log_10'],
                        "src_rmse":src_depth_metrics['rmse'],
                        "src_a1":src_depth_metrics['a1'],
                        "src_a2":src_depth_metrics['a2'],
                        "src_a3":src_depth_metrics['a3'],
                        "tgt_abs_rel":tgt_depth_metrics['abs_rel'],
                        "tgt_log_10":tgt_depth_metrics['log_10'],
                        "tgt_rmse":tgt_depth_metrics['rmse'],
                        "tgt_a1":tgt_depth_metrics['a1'],
                        "tgt_a2":tgt_depth_metrics['a2'],
                        "tgt_a3":tgt_depth_metrics['a3'],
                        "ap02": depth_AP[0],
                        "ap04": depth_AP[1],
                        "ap06": depth_AP[2],
                        "ap09": depth_AP[3],
                        "two_src_distance": two_src_distance,
                    }
                )

                # ttt05 = time.process_time()

            if os.environ['LLFF_BLEND'] == 'True':
                render_tgt_RGBs_T3HW = torch.stack(render_tgt_RGBs,dim=0)
                render_tgt_As_T1HW = torch.stack(render_tgt_As,dim=0)
                render_tgt_Ds_T1HW = torch.stack(render_tgt_Ds,dim=0)[:,None,:,:]

                T,_,H,W = render_tgt_RGBs_T3HW.shape
                weights = torch.ones([T,1,H,W], device=render_tgt_RGBs_T3HW.device)


                hard_alpha = False
                if hard_alpha:
                    for ti in range(T):
                        render_tgt_As_T1HW[ti,0] = (render_tgt_RGBs_T3HW[ti][1] >= 0.).float()

                trick = False
                if trick:
                    small_one = 1 if torch.sum(render_tgt_As_T1HW[0]) > torch.sum(render_tgt_As_T1HW[1]) else 0
                    large_one = 1 if small_one == 0 else 0

                    # zmf_map = render_tgt_As_T1HW[0] < render_tgt_As_T1HW[1]
                    # weights[0][zmf_map] = weights[0][zmf_map] * 0.

                    # zmf_map = render_tgt_As_T1HW[0] > render_tgt_As_T1HW[1]
                    # weights[1][zmf_map] = weights[1][zmf_map] * 0.

                    zmf_map = render_tgt_As_T1HW[large_one] > 0.6
                    weights[small_one][zmf_map] = weights[small_one][zmf_map] * 0.

                    # weights[small_one] = weights[small_one] * 0.
                
                scale_brightness = True
                if scale_brightness:
                    overlap_A_1HW = torch.logical_and(render_tgt_As_T1HW[0],render_tgt_As_T1HW[1])
                    overlap_A_1HW = overlap_A_1HW.repeat(3,1,1).to(torch.bool)
                    scale0 = torch.sum(render_tgt_RGBs_T3HW[0][overlap_A_1HW])
                    scale1 = torch.sum(render_tgt_RGBs_T3HW[1][overlap_A_1HW])
                    render_tgt_RGBs_T3HW[1] *= (scale0/scale1)

                erode_black_line = True
                if erode_black_line:
                    ksize = 5
                    ksize = int(ksize) if int(ksize) % 2 == 1 else int(ksize)+1
                    max_pool = torch.nn.MaxPool2d(kernel_size=ksize, stride=1, padding=int((ksize-1)/2))
                    render_tgt_As_T1HW = -max_pool(-render_tgt_As_T1HW[None,...][0])
                    render_tgt_RGBs_T3HW *= render_tgt_As_T1HW.repeat(1,3,1,1)

                out_frame = (render_tgt_RGBs_T3HW*weights).sum(0) / (1e-10+(render_tgt_As_T1HW*weights).sum(0))
                out_depth = (render_tgt_Ds_T1HW  *weights).sum(0) / (1e-10+(render_tgt_As_T1HW*weights).sum(0))
                gt_tgt_view_rgb_3HW = (input['tgt_image']/255.).to(device_type)


                if save_images:

                    # alpha
                    if True:
                        for ti in range(T):
                            tgt_A_HW = render_tgt_As_T1HW[ti,0]
                            see_tgt_A_HW = (tgt_A_HW*255).type(torch.uint8) #.cpu().numpy() #[:,:,[2,1,0]]
                            try:
                                storage.put_image("results/{}_zip_alpha_{}".format(im_idx0,ti), see_tgt_A_HW[None,...])
                            except:
                                self.writer.add_image("results/{}_zip_alpha_{}".format(im_idx0,ti), see_tgt_A_HW[None,...], 0)
                                self.save_im("results/{}_zip_alpha_{}".format(im_idx0,ti), see_tgt_A_HW[None,...])

                    # out rgb
                    if True:
                        see_out_frame = (out_frame*255).type(torch.uint8) #.cpu().numpy() #[:,:,[2,1,0]]
                        try:
                            storage.put_image("results/{}_zip_frame".format(im_idx0), see_out_frame)
                        except:
                            self.writer.add_image("results/{}_zip_frame".format(im_idx0), see_out_frame, 0)
                            self.save_im("results/{}_zip_frame".format(im_idx0), see_out_frame)

                    # out depth
                    if True:
                        out_depth_HW = out_depth[0]
                        out_depth_HW3 = self.drawDepth(out_depth_HW.detach().cpu().numpy())
                        see_out_depth_3HW = out_depth_HW3.transpose(2, 0, 1)
                        try:
                            storage.put_image("results/{}_zip_depth".format(im_idx0), see_out_depth_3HW)
                        except:
                            self.writer.add_image("results/{}_zip_depth".format(im_idx0), see_out_depth_3HW, 0)
                            self.save_im("results/{}_zip_depth".format(im_idx0), see_out_depth_3HW)
                

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

    def eval_plane_ap_depth(self, filename, predSegmentations, gtSegmentations, predDepths, gtDepths, threshold=0.5):
        # predNumPlanes = pred_plane_num  # actually, it is the maximum number of the predicted planes
        predNumPlanes = np.max(predSegmentations)
        gtNumPlanes = len(np.unique(gtSegmentations))-1
        predSegmentations -= 1
        gtSegmentations -= 1
        



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

        if True:
            plane_IOUs = planeIOUs
            depths_diff = planeDiffs
            plane_areas = planeAreas
            APs = []
            for diff_threshold in [0.2, 0.3, 0.6, 0.9]:
                correct_mask = np.minimum((depths_diff < diff_threshold), (plane_IOUs > 0.5))
                match_mask = np.zeros(len(correct_mask), dtype=np.bool)
                recalls = []
                precisions = []
                num_predictions = correct_mask.shape[-1]
                num_targets = (plane_areas > 0).sum()
                for rank in range(num_predictions):
                    match_mask = np.maximum(match_mask, correct_mask[:, rank])
                    num_matches = match_mask.sum()
                    precisions.append(float(num_matches) / (rank + 1))
                    recalls.append(float(num_matches) / num_targets)
                    continue
                max_precision = 0.0
                prev_recall = 1.0
                AP = 0.0
                for recall, precision in zip(recalls[::-1], precisions[::-1]):
                    AP += (prev_recall - recall) * max_precision
                    max_precision = max(max_precision, precision)
                    prev_recall = recall
                    continue
                AP += prev_recall * max_precision
                APs.append(AP)

        return APs

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
        pred_plane_idx_max = int(pred_plane_idxs[-1])
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
        predSegmentations = torch.from_numpy(predSegmentations_np).to(device)
        gtSegmentations = torch.from_numpy(gtSegmentations).to(device)

        # print(gtSegmentations.shape)

        pred_masks = []
        if pred_non_plane_idx == 0:
            # print(np.unique(predSegmentations_np))
            # print(np.max(predSegmentations_np)+1)
            for i in range(1, int(np.max(predSegmentations_np)+1)):
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

        if os.environ['ONLY_SEG'] == 'True':
            return

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # print(self._predictions) # RI, VOI, SC

        all_two_src_distance = []
        all_RI = []
        all_VOI = []
        all_SC = []
        all_ssim = []
        all_psnr = []
        all_lpips = []
        all_src_rel = []
        all_src_log = []
        all_src_rmse = []
        all_src_a1 = []
        all_src_a2 = []
        all_src_a3 = []
        all_tgt_rel = []
        all_tgt_log = []
        all_tgt_rmse = []
        all_tgt_a1 = []
        all_tgt_a2 = []
        all_tgt_a3 = []
        all_ap02 = []
        all_ap04 = []
        all_ap06 = []
        all_ap09 = []
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

            all_src_rel.append(p['src_abs_rel'])
            all_src_log.append(p['src_log_10'])
            all_src_rmse.append(p['src_rmse'])
            all_src_a1.append(p['src_a1'])
            all_src_a2.append(p['src_a2'])
            all_src_a3.append(p['src_a3'])
            all_tgt_rel.append(p['tgt_abs_rel'])
            all_tgt_log.append(p['tgt_log_10'])
            all_tgt_rmse.append(p['tgt_rmse'])
            all_tgt_a1.append(p['tgt_a1'])
            all_tgt_a2.append(p['tgt_a2'])
            all_tgt_a3.append(p['tgt_a3'])

            all_ap02.append(p['ap02'])
            all_ap04.append(p['ap04'])
            all_ap06.append(p['ap06'])
            all_ap09.append(p['ap09'])

            if 'two_src_distance' in p:
                all_two_src_distance.append(p['two_src_distance'])

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

        all_src_rel = np.array(all_src_rel)
        all_src_log = np.array(all_src_log)
        all_src_rmse = np.array(all_src_rmse)
        all_src_a1 = np.array(all_src_a1)
        all_src_a2 = np.array(all_src_a2)
        all_src_a3 = np.array(all_src_a3)
        all_tgt_rel = np.array(all_tgt_rel)
        all_tgt_log = np.array(all_tgt_log)
        all_tgt_rmse = np.array(all_tgt_rmse)
        all_tgt_a1 = np.array(all_tgt_a1)
        all_tgt_a2 = np.array(all_tgt_a2)
        all_tgt_a3 = np.array(all_tgt_a3)
        all_ap02 = np.array(all_ap02)
        all_ap04 = np.array(all_ap04)
        all_ap06 = np.array(all_ap06)
        all_ap09 = np.array(all_ap09)

        if len(all_two_src_distance) > 0:
            all_two_src_distance = np.array(all_two_src_distance)
            print('two src distance:',all_two_src_distance.mean())

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

        print('RI',all_RI.mean(),'VOI',all_VOI.mean(),'SC',all_SC.mean(),'SSIM',all_ssim.mean(),'PSNR',all_psnr.mean(),'LPIPS',all_lpips.mean())
        print('src_Rel',all_src_rel.mean(),'src_log10',all_src_log.mean(),'src_RMSE',all_src_rmse.mean(),'src_a1',all_src_a1.mean(),'src_a2',all_src_a2.mean(),'src_a3',all_src_a3.mean())
        print('tgt_Rel',all_tgt_rel.mean(),'tgt_log10',all_tgt_log.mean(),'tgt_RMSE',all_tgt_rmse.mean(),'tgt_a1',all_tgt_a1.mean(),'tgt_a2',all_tgt_a2.mean(),'tgt_a3',all_tgt_a3.mean())
        print('ap02',all_ap02.mean(),'ap04',all_ap04.mean(),'ap06',all_ap06.mean(),'ap09',all_ap09.mean())
        try:
            storage = get_event_storage()
            storage.put_scalar("val: Mean RI on %d samples"%len(self._predictions), all_RI.mean())
            storage.put_scalar("val: Mean VOI", all_VOI.mean())
            storage.put_scalar("val: Mean SC", all_SC.mean())
            storage.put_scalar("val: Mean ssim", all_ssim.mean())
            storage.put_scalar("val: Mean psnr", all_psnr.mean())
            storage.put_scalar("val: Mean lpips", all_lpips.mean())

            storage.put_scalar("val: Mean src_Rel", all_src_rel.mean())
            storage.put_scalar("val: Mean src_log10", all_src_log.mean())
            storage.put_scalar("val: Mean src_RMSE", all_src_rmse.mean())
            storage.put_scalar("val: Mean src_a1", all_src_a1.mean())
            storage.put_scalar("val: Mean src_a2", all_src_a2.mean())
            storage.put_scalar("val: Mean src_a3", all_src_a3.mean())
            storage.put_scalar("val: Mean tgt_Rel", all_tgt_rel.mean())
            storage.put_scalar("val: Mean tgt_log10", all_tgt_log.mean())
            storage.put_scalar("val: Mean tgt_RMSE", all_tgt_rmse.mean())
            storage.put_scalar("val: Mean tgt_a1", all_tgt_a1.mean())
            storage.put_scalar("val: Mean tgt_a2", all_tgt_a2.mean())
            storage.put_scalar("val: Mean tgt_a3", all_tgt_a3.mean())

            storage.put_image("stat/eval/depth_pixel", see_depth_pixel)
            storage.put_image("stat/eval/depth_plane", see_depth_plane)
            storage.put_image("stat/eval/normal_pixel", see_normal_pixel)
            storage.put_image("stat/eval/normal_plane", see_normal_plane)
        except:
            self.writer.add_scalar("val: Mean RI on %d samples"%len(self._predictions), all_RI.mean(), 0)
            self.writer.add_scalar("val: Mean VOI", all_VOI.mean(), 0)
            self.writer.add_scalar("val: Mean SC", all_SC.mean(), 0)
            self.writer.add_scalar("val: Mean ssim", all_ssim.mean(), 0)
            self.writer.add_scalar("val: Mean psnr", all_psnr.mean(), 0)
            self.writer.add_scalar("val: Mean lpips", all_lpips.mean(), 0)

            self.writer.add_scalar("val: Mean src_Rel", all_src_rel.mean(), 0)
            self.writer.add_scalar("val: Mean src_log10", all_src_log.mean(), 0)
            self.writer.add_scalar("val: Mean src_RMSE", all_src_rmse.mean(), 0)
            self.writer.add_scalar("val: Mean src_a1", all_src_a1.mean(), 0)
            self.writer.add_scalar("val: Mean src_a2", all_src_a2.mean(), 0)
            self.writer.add_scalar("val: Mean src_a3", all_src_a3.mean(), 0)
            self.writer.add_scalar("val: Mean tgt_Rel", all_tgt_rel.mean(), 0)
            self.writer.add_scalar("val: Mean tgt_log10", all_tgt_log.mean(), 0)
            self.writer.add_scalar("val: Mean tgt_RMSE", all_tgt_rmse.mean(), 0)
            self.writer.add_scalar("val: Mean tgt_a1", all_tgt_a1.mean(), 0)
            self.writer.add_scalar("val: Mean tgt_a2", all_tgt_a2.mean(), 0)
            self.writer.add_scalar("val: Mean tgt_a3", all_tgt_a3.mean(), 0)

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
