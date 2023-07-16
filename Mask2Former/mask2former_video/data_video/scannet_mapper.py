# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging
from traceback import print_tb

import numpy as np
import torch, cv2, os, time

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Boxes, Instances

from skimage import draw

__all__ = ['ScanNetMapper']

def drawDepthImage(depth, maxDepth=10):
    depthImage = np.clip(depth / maxDepth * 255, 0, 255).astype(np.uint8)
    depthImage = cv2.applyColorMap(255 - depthImage, colormap=cv2.COLORMAP_JET)
    return depthImage

def drawDepth(depth,addr):
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

def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    # image_size = cfg.INPUT.IMAGE_SIZE
    (h,w) = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    # if cfg.INPUT.RANDOM_FLIP != "none":
    #     augmentation.append(
    #         T.RandomFlip(
    #             horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
    #             vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
    #         )
    #     )

    augmentation.extend([
        # T.ResizeScale(
        #     min_scale=1., max_scale=1., target_height=image_size, target_width=image_size
        # ),
        # T.FixedSizeCrop(crop_size=(image_size, image_size)),
        T.Resize((h,w))
    ])

    return augmentation

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

def get_plane_parameters(plane, segmentation, mp3d=False):
    if mp3d:
        offsets = np.linalg.norm(plane, ord=2, axis=1)
        norms = plane / offsets.reshape(-1, 1)
        plane = norms  / offsets.reshape(-1, 1)
    else:
        plane = plane[:,:3] / plane[:,3:]


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

def precompute_K_inv_dot_xy_1(h=192, w=256):
    focal_length = 517.97
    offset_x = 320
    offset_y = 240

    K = [[focal_length, 0, offset_x],
            [0, focal_length, offset_y],
            [0, 0, 1]]

    K_inv = np.linalg.inv(np.array(K))

    # full
    K_inv_dot_xy_1 = np.zeros((3, h, w))
    # xy_map = np.zeros((2, h, w))
    for y in range(h):
        for x in range(w):
            yy = float(y) / h * 480
            xx = float(x) / w * 640

            ray = np.dot(K_inv, np.array([xx, yy, 1]).reshape(3, 1))
            K_inv_dot_xy_1[:, y, x] = ray[:, 0]
            # xy_map[0, y, x] = float(x) / w
            # xy_map[1, y, x] = float(y) / h

    # half
    h = int(h/2)
    w = int(w/2)
    K_inv_dot_xy_1_2 = np.zeros((3, h, w))
    # xy_map = np.zeros((2, h, w))
    for y in range(h):
        for x in range(w):
            yy = float(y) / h * 480
            xx = float(x) / w * 640

            ray = np.dot(K_inv, np.array([xx, yy, 1]).reshape(3, 1))
            K_inv_dot_xy_1_2[:, y, x] = ray[:, 0]
            # xy_map[0, y, x] = float(x) / w
            # xy_map[1, y, x] = float(y) / h

    # 1/4
    h = int(h/2)
    w = int(w/2)
    K_inv_dot_xy_1_4 = np.zeros((3, h, w))
    # xy_map = np.zeros((2, h, w))
    for y in range(h):
        for x in range(w):
            yy = float(y) / h * 480
            xx = float(x) / w * 640

            ray = np.dot(K_inv, np.array([xx, yy, 1]).reshape(3, 1))
            K_inv_dot_xy_1_4[:, y, x] = ray[:, 0]
            # xy_map[0, y, x] = float(x) / w
            # xy_map[1, y, x] = float(y) / h

    # 1/8
    h = int(h/2)
    w = int(w/2)
    K_inv_dot_xy_1_8 = np.zeros((3, h, w))
    # xy_map = np.zeros((2, h, w))
    for y in range(h):
        for x in range(w):
            yy = float(y) / h * 480
            xx = float(x) / w * 640

            ray = np.dot(K_inv, np.array([xx, yy, 1]).reshape(3, 1))
            K_inv_dot_xy_1_8[:, y, x] = ray[:, 0]
            # xy_map[0, y, x] = float(x) / w
            # xy_map[1, y, x] = float(y) / h

    # precompute to speed up processing
    # self.K_inv_dot_xy_1 = K_inv_dot_xy_1
    # self.xy_map = xy_map



    return K_inv_dot_xy_1, K_inv_dot_xy_1_2, K_inv_dot_xy_1_4, K_inv_dot_xy_1_8

def plane2depth(plane_parameters, segmentation, gt_depth, K_inv_dot_xy_1, h=480, w=640):

    depth_map = 1. / np.sum(K_inv_dot_xy_1.reshape(3, -1) * plane_parameters.reshape(3, -1), axis=0)
    depth_map = depth_map.reshape(h, w)

    planar_area = np.zeros(segmentation.shape[1:]).astype(bool)
    for i in range(segmentation.shape[0]):
        planar_area = np.logical_or(planar_area,segmentation[i])
    planar_area = planar_area.astype(int)

    # replace non planer region depth using sensor depth map
    depth_map[planar_area == 0] = gt_depth[planar_area == 0]
    # depth_map[planar_area == 0] = 0 
    return depth_map

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

# This is specifically designed for the COCO dataset.
class ScanNetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        h,
        w,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[ScanNetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train
        self.cfg_h = h
        self.cfg_w = w

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        (h,w) = cfg.INPUT.IMAGE_SIZE

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "h": h,
            "w": w,
        }
        return ret

    def get_plane_params_in_local(self, planes, camera_info):
        """
        input: 
        @planes: plane params
        @camera_info: plane params from camera info, type = dict, must contain 'position' and 'rotation' as keys
        output:
        plane parameters in global frame.
        """
        import quaternion
        tran = np.array(camera_info['position'])
        rot = quaternion.from_float_array(camera_info["rotation"]) # np.array(camera_info['rotation'])
        b = planes
        a = np.ones((len(planes),3))*tran
        planes_world = a + b - ((a*b).sum(axis=1) / np.linalg.norm(b, axis=1)**2).reshape(-1,1)*b
        end = (quaternion.as_rotation_matrix(rot.inverse())@(planes_world - tran).T).T #world2cam
        planes_local = end*np.array([1, -1, -1])# habitat2suncg
        return planes_local

    def get_plane_params_in_global(self, planes, camera_info):
        """
        input:
        @planes: plane params
        @camera_info: plane params from camera info, type = dict, must contain 'position' and 'rotation' as keys
        output:
        plane parameters in global frame.
        """
        import quaternion
        tran = torch.FloatTensor(camera_info["position"])
        rot = quaternion.from_float_array(camera_info["rotation"])
        start = torch.ones((len(planes), 3)) * tran
        end = torch.FloatTensor(planes) * torch.tensor([1, -1, -1])  # suncg2habitat
        end = (
            torch.mm(
                torch.FloatTensor(quaternion.as_rotation_matrix(rot)),
                (end).T,
            ).T
            + tran
        )  # cam2world
        a = end
        b = end - start
        planes_world = ((a * b).sum(dim=1) / (torch.norm(b, dim=1) + 1e-5) ** 2).view(-1, 1) * b
        return planes_world

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # print(dataset_dict.keys()) # dict_keys(['file_name', 'image_id', 'segments_info'])

        if os.environ['TEST_NVS_ONLY'] == 'True':
            # PREPARE 'image'
            if True:      
                dataset_dict['image'] = [] # src images
                for fi in range(len(dataset_dict['file_name'])):
                    image = utils.read_image(dataset_dict['file_name'][fi], format=self.img_format)
                    image = cv2.resize(image,(640,480)) # to match depth map and segmentation's original size
                    utils.check_image_size(dataset_dict, image)
                    image, transforms = T.apply_transform_gens(self.tfm_gens, image)
                    dataset_dict['image'].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            # PREPARE 'cam_pose'
            if True:
                dataset_dict['cam_pose'] = []
                if os.environ['TEST_SRC_NUM'] == '2':
                    pose0_addr = dataset_dict['file_name'][0].replace('/home/v-mingfzhang/code/for_nerf/scenes/','/data/zmf_data/ScanNet/scans///').replace('/train/rgb/','/unpack/pose/')[:-4]+'.txt'
                    pose1_addr = dataset_dict['file_name'][1].replace('/home/v-mingfzhang/code/for_nerf/scenes/','/data/zmf_data/ScanNet/scans///').replace('/train/rgb/','/unpack/pose/')[:-4]+'.txt'
                    pose0 = np.loadtxt(pose0_addr)
                    pose1 = np.loadtxt(pose1_addr)
                    if not (np.isfinite(pose0).all() and np.isfinite(pose1).all()):
                        assert False
                    pose0 = np.linalg.inv(pose0)
                    pose1 = np.linalg.inv(pose1)
                    dataset_dict['cam_pose'].append(torch.from_numpy(pose0))
                    dataset_dict['cam_pose'].append(torch.from_numpy(pose1))
                else:
                    if os.environ['TEST_NYU'] == 'True' or os.environ['TEST_MP3D'] == 'True':
                        pose0 = np.array([
                            [1., 0., 0., 0.],
                            [0., 1., 0., 0.],
                            [0., 0., 1., 0.],
                            [0., 0., 0., 1.],
                        ])
                        dataset_dict['cam_pose'].append(torch.from_numpy(pose0))
                    else:
                        pose0_addr = dataset_dict['file_name'][0].replace('/home/v-mingfzhang/code/for_nerf/scenes/','/data/zmf_data/ScanNet/scans///').replace('/train/rgb/','/unpack/pose/')[:-4]+'.txt'
                        pose0 = np.loadtxt(pose0_addr)
                        if not np.isfinite(pose0).all():
                            assert False
                        pose0 = np.linalg.inv(pose0)
                        dataset_dict['cam_pose'].append(torch.from_numpy(pose0))

            # PREPARE target view
            if True:
                tgt_im_addr = dataset_dict['tgt_file_name'][0]
                tgt_image = utils.read_image(tgt_im_addr, format=self.img_format)
                tgt_image = cv2.resize(tgt_image,(640,480)) # to match depth map and segmentation's original size
                tgt_image = transforms.apply_segmentation(tgt_image.astype(float))
                dataset_dict['tgt_image'] = torch.as_tensor(np.ascontiguousarray(tgt_image.transpose(2, 0, 1)))
                dataset_dict['tgt_filename'] = tgt_im_addr

                # TGT POSE
                if os.environ['TEST_NYU'] == 'True' or os.environ['TEST_MP3D'] == 'True':
                    tgt_pose = np.array([
                        [1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.],
                    ])
                    dataset_dict['tgt_cam_pose'] = torch.from_numpy(tgt_pose)
                else:
                    tgt_pose_addr = tgt_im_addr.replace('/home/v-mingfzhang/code/for_nerf/scenes/','/data/zmf_data/ScanNet/scans///').replace('/test/rgb/','/unpack/pose/')[:-4]+'.txt'
                    tgt_pose = np.loadtxt(tgt_pose_addr)
                    if not np.isfinite(tgt_pose).all():
                        assert False
                    tgt_pose = np.linalg.inv(tgt_pose)
                    dataset_dict['tgt_cam_pose'] = torch.from_numpy(tgt_pose)

                # G_src_tgt
                dataset_dict['tgt_G_src_tgt'] = []
                for src_pose in dataset_dict['cam_pose']:
                    this_G_src_tgt = src_pose.numpy() @ np.linalg.inv(tgt_pose)
                    if np.isnan(this_G_src_tgt).any():
                        print('zmf: nan detected',dataset_dict['file_name'],tgt_im_addr)
                        exit()
                    dataset_dict['tgt_G_src_tgt'].append(torch.from_numpy(this_G_src_tgt))
                
                # TGT DEPTH
                if os.environ['TEST_NYU'] == 'True':
                    tgt_depth_addr = tgt_im_addr.replace('rgb_','sync_depth_').replace('.jpg','.png')
                    tgt_depth_map = utils.read_image(tgt_depth_addr) / 1000.
                    tgt_depth_map = transforms.apply_segmentation(tgt_depth_map.astype(float))
                    dataset_dict['tgt_depth_map'] = torch.from_numpy(tgt_depth_map)
                elif os.environ['TEST_MP3D'] == 'True':
                    tgt_depth_map = np.zeros([480,640])
                    tgt_depth_map = transforms.apply_segmentation(tgt_depth_map.astype(float))
                    dataset_dict['tgt_depth_map'] = torch.from_numpy(tgt_depth_map)
                else:
                    tgt_depth_addr = tgt_im_addr.replace('/home/v-mingfzhang/code/for_nerf/scenes/','/data/zmf_data/ScanNet/scans///').replace('/test/rgb/','/unpack/depth/')[:-4]+'.png'
                    tgt_depth_map = utils.read_image(tgt_depth_addr) / 1000.
                    tgt_depth_map = transforms.apply_segmentation(tgt_depth_map.astype(float))
                    dataset_dict['tgt_depth_map'] = torch.from_numpy(tgt_depth_map)

            # PREPARE 'depth_map'
            if True:
                if os.environ['TEST_NYU'] == 'True':
                    dataset_dict['depth_map'] = []
                    dataset_dict['depth_from_plane'] = []
                    dataset_dict['depth_from_plane_and_nonplane'] = []
                    for fi in range(len(dataset_dict['file_name'])):
                        depth_addr = dataset_dict['file_name'][fi].replace('rgb_','sync_depth_').replace('.jpg','.png')
                        depth_map = utils.read_image(depth_addr) / 1000.
                        depth_map = transforms.apply_segmentation(depth_map.astype(float))
                        dataset_dict['depth_map'].append(torch.as_tensor(np.ascontiguousarray(depth_map)))
                        dataset_dict['depth_from_plane'].append(torch.as_tensor(np.ascontiguousarray(depth_map)))
                        dataset_dict['depth_from_plane_and_nonplane'].append(torch.as_tensor(np.ascontiguousarray(depth_map)))
                elif os.environ['TEST_MP3D'] == 'True':
                    dataset_dict['depth_map'] = []
                    dataset_dict['depth_from_plane'] = []
                    dataset_dict['depth_from_plane_and_nonplane'] = []
                    for fi in range(len(dataset_dict['file_name'])):
                        depth_map = np.zeros([480,640])
                        depth_map = transforms.apply_segmentation(depth_map.astype(float))
                        dataset_dict['depth_map'].append(torch.as_tensor(np.ascontiguousarray(depth_map)))
                        dataset_dict['depth_from_plane'].append(torch.as_tensor(np.ascontiguousarray(depth_map)))
                        dataset_dict['depth_from_plane_and_nonplane'].append(torch.as_tensor(np.ascontiguousarray(depth_map)))
                else:
                    dataset_dict['depth_map'] = []
                    dataset_dict['depth_from_plane'] = []
                    dataset_dict['depth_from_plane_and_nonplane'] = []
                    for fi in range(len(dataset_dict['file_name'])):
                        depth_addr = dataset_dict['file_name'][fi].replace('/home/v-mingfzhang/code/for_nerf/scenes/','/data/zmf_data/ScanNet/scans///').replace('/train/rgb/','/unpack/depth/')[:-4]+'.png'
                        depth_map = utils.read_image(depth_addr) / 1000.
                        depth_map = transforms.apply_segmentation(depth_map.astype(float))
                        dataset_dict['depth_map'].append(torch.as_tensor(np.ascontiguousarray(depth_map)))
                        dataset_dict['depth_from_plane'].append(torch.as_tensor(np.ascontiguousarray(depth_map)))
                        dataset_dict['depth_from_plane_and_nonplane'].append(torch.as_tensor(np.ascontiguousarray(depth_map)))

            # PREPARE 'K_inv_dot_xy_1'
            if True:
                K_inv_dot_xy_1,K_inv_dot_xy_1_2,K_inv_dot_xy_1_4,K_inv_dot_xy_1_8 = precompute_K_inv_dot_xy_1(h=self.cfg_h, w=self.cfg_w)
                dataset_dict['K_inv_dot_xy_1'] = torch.from_numpy(K_inv_dot_xy_1)
                dataset_dict['K_inv_dot_xy_1_2'] = torch.from_numpy(K_inv_dot_xy_1_2)
                dataset_dict['K_inv_dot_xy_1_4'] = torch.from_numpy(K_inv_dot_xy_1_4)
                dataset_dict['K_inv_dot_xy_1_8'] = torch.from_numpy(K_inv_dot_xy_1_8)

            return dataset_dict

        if os.environ['TRAIN_MP3D'] == 'True':
            # parameters for nonplane
            seg_nonplane = os.environ['SEG_NONPLANE'] == 'True'
            seg_nonplane_num = int(os.environ['SEG_NONPLANE_NUM'])
            erode_kernel = self.cfg_h/float(os.environ['ERODE_FAC'])  # larger -> more erode
            start_disparity = 5.
            end_disparity = 0.01
            debug_nonplane = False
            final_vis = False

            # PREPARE 'image'
            if True:      
                dataset_dict['image'] = [] # src images
                for fi in range(len(dataset_dict['file_name'])):
                    image = utils.read_image(dataset_dict['file_name'][fi], format=self.img_format)
                    image = cv2.resize(image,(640,480)) # to match depth map and segmentation's original size
                    utils.check_image_size(dataset_dict, image)
                    image, transforms = T.apply_transform_gens(self.tfm_gens, image)
                    dataset_dict['image'].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            # PREPARE 'cam_pose'
            if True:
                dataset_dict['cam_pose'] = []
                cam_pose_invalid = False
                if os.environ['SAMPLING_FRAME_NUM'] == '2':
                    
                    dataset_dict['cam_pose'].append(dataset_dict['camera'][0])
                    dataset_dict['cam_pose'].append(dataset_dict['camera'][1])               
                else:

                    dataset_dict['cam_pose'].append(dataset_dict['camera'][0])

                
            # PREPARE 'depth_map'
            if True:
                dataset_dict['depth_map'] = []
                depth_map_np = []
                for fi in range(len(dataset_dict['file_name'])):
                    depth_addr = dataset_dict['file_name'][fi].replace('/rgb/','/observations/')
                    depth_map = utils.read_image(depth_addr) / 1000.
                    depth_map = transforms.apply_segmentation(depth_map.astype(float))
                    dataset_dict['depth_map'].append(torch.as_tensor(np.ascontiguousarray(depth_map)))
                    depth_map_np.append(depth_map)

            # PREPARE 'completed depth_map'
            if True and seg_nonplane:
                dataset_dict['depth_map_complete'] = []
                depth_map_complete_np = []
                for fi in range(len(dataset_dict['file_name'])):
                    depth_addr_complete = dataset_dict['file_name'][fi].replace('/rgb/','/observations/')
                    depth_map_complete = utils.read_image(depth_addr_complete) / 1000.
                    depth_map_complete = transforms.apply_segmentation(depth_map_complete.astype(float))
                    dataset_dict['depth_map_complete'].append(torch.as_tensor(np.ascontiguousarray(depth_map_complete)))
                    depth_map_complete_np.append(depth_map_complete)

            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            # dataset_dict['image'] = [torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            # if not self.is_train:
            #     # USER: Modify this if you want to keep them for some reason.
            #     dataset_dict.pop("annotations", None)
            #     return dataset_dict

            # PREPARE 'K_inv_dot_xy_1'
            if True:
                K_inv_dot_xy_1,K_inv_dot_xy_1_2,K_inv_dot_xy_1_4,K_inv_dot_xy_1_8 = precompute_K_inv_dot_xy_1(h=self.cfg_h, w=self.cfg_w)
                dataset_dict['K_inv_dot_xy_1'] = torch.from_numpy(K_inv_dot_xy_1)
                dataset_dict['K_inv_dot_xy_1_2'] = torch.from_numpy(K_inv_dot_xy_1_2)
                dataset_dict['K_inv_dot_xy_1_4'] = torch.from_numpy(K_inv_dot_xy_1_4)
                dataset_dict['K_inv_dot_xy_1_8'] = torch.from_numpy(K_inv_dot_xy_1_8)

            # PREPARE FROM segments_info
            if True:
                dataset_dict['instances'] = []
                dataset_dict['planes_paras'] = []
                dataset_dict['depth_from_plane'] = []
                dataset_dict['depth_from_plane_and_nonplane'] = []
                dataset_dict['mask_nonplane_valid'] = []
                if seg_nonplane:
                    dataset_dict['mask_otherthan_plane'] = []
                    dataset_dict['nonplane_num'] = []

                for fi in range(len(dataset_dict['file_name'])):
                    segments_info = dataset_dict['segments_info'][fi]
                    planes_masks = []
                    planes_paras = []
                    have_plane = False
                    if seg_nonplane:
                        nonplanes_masks = []
                        nonplanes_paras = []
                        have_nonplane = False
                    
                    
                    # PRE-PREPARE plane masks & para
                    if True:
                        for obj in segments_info:
                            # print(obj.keys()) # dict_keys(['id', 'image_id', 'category_id', 'iscrowd', 'area', 'bbox', 'segmentation', 'width', 'height', 'bbox_mode', 'plane'])
                            plane_paras = obj['plane']
                            planes_paras.append(plane_paras)
                            
                            polygons = obj['segmentation']
                            oneplane_mask = np.zeros((640,480), dtype=np.bool)
                            for poly in polygons:
                                # print(poly) # [580.0, 252.5, 581.0, 243.5, 580.0, 252.5]
                                if len(poly) > 0:
                                    have_plane = True
                                    this_mask = poly2mask(poly[::2],poly[1::2],(640,480))
                                    oneplane_mask = np.logical_or(oneplane_mask, this_mask)
                            planes_masks.append(oneplane_mask)
                            
                        if have_plane:
                            planes_paras = np.stack(np.array(planes_paras),axis=0)
                            planes_masks = np.stack(np.array(planes_masks),axis=0)
                            planes_masks = planes_masks.transpose(2,1,0)
                            planes_masks = transforms.apply_segmentation(planes_masks.astype(float))
                            planes_masks = planes_masks.astype(bool)
                            planes_masks = planes_masks.transpose(2,0,1)
                            # for i in range(planes_masks.shape[0]):
                            #     im = planes_masks[i].astype(float)
                            #     cv2.imwrite('./zmf_show/mask_%d.png'%i,(im*255).astype('uint8'))
                    
                    dataset_dict['planes_paras'].append(torch.from_numpy(planes_paras))

                    # PREPARE 'mask_nonplane_valid'
                    if True:
                        mask_nonplane_valid = depth_map_np[fi] > 1e-3
                        for i in range(planes_masks.shape[0]):
                            mask_nonplane_valid = np.logical_and(mask_nonplane_valid,np.logical_not(planes_masks[i]))
                        # drawDepth(torch.from_numpy(mask_nonplane_valid.astype(np.float)),'./see_mask_nonplane_valid.png')
                        dataset_dict['mask_nonplane_valid'].append(torch.as_tensor(np.ascontiguousarray(mask_nonplane_valid)))

                    # otherthan plane mask
                    if True and seg_nonplane:
                        if debug_nonplane:
                            drawDepth(torch.from_numpy(depth_map_np[fi]),'./zmf_debug/1_depth_map_sensor.png')
                            drawDepth(torch.from_numpy(depth_map_complete_np[fi]),'./zmf_debug/2_depth_map_complete.png')
                        mask_otherthan_plane = depth_map_complete_np[fi] > 1e-3
                        for i in range(planes_masks.shape[0]):
                            mask_otherthan_plane = np.logical_and(mask_otherthan_plane,np.logical_not(planes_masks[i]))
                        dataset_dict['mask_otherthan_plane'].append(torch.as_tensor(np.ascontiguousarray(mask_otherthan_plane)))
                        if debug_nonplane:
                            drawDepth(torch.from_numpy(mask_otherthan_plane.astype(np.float)),'./zmf_debug/3_otherthan_plane_mask.png')
                    
                    # 
                    if True and seg_nonplane:
                        if erode_kernel < 2: erode_kernel =  2
                        erode_kernel = int(erode_kernel) if int(erode_kernel) % 2 == 1 else int(erode_kernel)+1
                        max_pool = torch.nn.MaxPool2d(kernel_size=erode_kernel, stride=1, padding=int((erode_kernel-1)/2))
                        eroded_mask = -max_pool(-dataset_dict['mask_otherthan_plane'][fi][None,None,...].float())
                        mask_otherthan_plane = max_pool(eroded_mask)[0,0]
                        if debug_nonplane:
                            drawDepth(mask_otherthan_plane,'./zmf_debug/4_refine_mask.png')

                        depth_map_complete = dataset_dict['depth_map_complete'][fi]
                        mask_otherthan_plane = mask_otherthan_plane > 1e-3
                        non_plane_depth_map = depth_map_complete * mask_otherthan_plane

                        if debug_nonplane:
                            drawDepth(non_plane_depth_map,'./zmf_debug/5_non_plane_depth.png')
                        
                        
                        disparity_S = torch.linspace(
                            start_disparity, end_disparity, seg_nonplane_num
                        )
                        depth_S = torch.reciprocal(disparity_S)
                        # print(disparity_S)
                        # print(depth_S)

                        for di in range(seg_nonplane_num):
                            this_depth_min = depth_S[di-1] if di != 0 else 0.
                            this_depth_max = depth_S[di]
                            
                            this_nonplane_mask_1 = non_plane_depth_map > this_depth_min
                            this_nonplane_mask_2 = non_plane_depth_map < this_depth_max
                            this_nonplane_mask = torch.logical_and(this_nonplane_mask_1,this_nonplane_mask_2)

                            if dataset_dict['depth_map_complete'][fi][this_nonplane_mask].shape[0] == 0:
                                continue
                            
                            have_nonplane = True
                            this_depth = torch.mean(dataset_dict['depth_map_complete'][fi][this_nonplane_mask])
                            n_3 = torch.tensor([0, 0, 1])
                            # this_nonplane_para = n_3 / this_depth
                            this_nonplane_para = n_3 * this_depth

                            nonplanes_masks.append(this_nonplane_mask)
                            nonplanes_paras.append(this_nonplane_para)
                        
                        if have_nonplane:
                            nonplanes_paras = torch.stack(nonplanes_paras)
                            nonplanes_masks = torch.stack(nonplanes_masks)

                            # offset = torch.reciprocal(torch.norm(nonplanes_paras, dim=1, keepdim=True))
                            # nonplanes_paras = nonplanes_paras * offset
                            # nonplanes_paras = torch.cat([nonplanes_paras,offset],dim=1)

                            dataset_dict['planes_paras'][fi] = torch.cat([dataset_dict['planes_paras'][fi], nonplanes_paras],dim=0)

                        
                    
                    # PREPARE 'depth_from_plane'
                    if True:
                        # print(planes_paras.shape) # P, 4
                        # print(planes_masks.shape) # P, h, w
                        pixel_plane_para = get_plane_parameters(planes_paras.copy(),planes_masks,mp3d=True)
                        depth_from_plane = plane2depth(pixel_plane_para,planes_masks,depth_map_np[fi],K_inv_dot_xy_1,h=self.cfg_h,w=self.cfg_w)
                        if final_vis:
                            depth_zeros = np.zeros(depth_map_np[fi].shape)
                            depth_from_plane = plane2depth(pixel_plane_para,planes_masks,depth_zeros,K_inv_dot_xy_1,h=self.cfg_h,w=self.cfg_w)
                        depth_from_plane[depth_from_plane>20.] = 20.
                        depth_from_plane[depth_from_plane<0.] = 0.
                        # if check_tensor_naninf(torch.from_numpy(depth_from_plane).half(),'load depth'): print(dataset_dict['file_name'])
                        # drawDepth(torch.from_numpy(depth_map_np[fi]),'./zmf_show/depthmap_sensor.png')
                        # drawDepth(torch.from_numpy(depth_from_plane),'./zmf_show/depthmap.png')
                        dataset_dict['depth_from_plane'].append(torch.as_tensor(np.ascontiguousarray(depth_from_plane)))

                    # PREPARE 'instances'
                    if True:
                        instances = Instances([self.cfg_h,self.cfg_w])
                        classes = []
                        masks = []
                        for segment_info in segments_info:
                            class_id = 0 # segment_info['category_id']
                            classes.append(class_id)
                        if seg_nonplane and have_nonplane:
                            for tmp in nonplanes_paras:
                                class_id = 50 # segment_info["category_id"]
                                classes.append(class_id)
                        classes = np.array(classes)
                        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

                        if not have_plane:
                            # Some image does not have annotation (all ignored)
                            instances.gt_masks = torch.zeros((0, self.cfg_h, self.cfg_w))
                            instances.gt_boxes = Boxes(torch.zeros((0, 4)))
                            print('zmf have no plane anno', dataset_dict['file_name'])
                        else:
                            plane_mask = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in planes_masks])
                            if seg_nonplane and have_nonplane:
                                nonplane_mask = torch.stack([x for x in nonplanes_masks])
                                plane_mask = torch.cat((plane_mask,nonplane_mask),dim=0)
                            masks = BitMasks(plane_mask)
                            instances.gt_masks = masks #.tensor 
                            instances.gt_boxes = masks.get_bounding_boxes()

                        dataset_dict['instances'].append(instances)
                        if have_nonplane:
                            dataset_dict['nonplane_num'].append(nonplanes_paras.shape[0])
                        else:
                            dataset_dict['nonplane_num'].append(0)
                        
                    # PREPARE 'depth_from_plane_and_nonplane'
                    if True:
                        nplanes_paras = dataset_dict['planes_paras'][fi].numpy() # P, 4
                        nplanes_masks = dataset_dict['instances'][fi].gt_masks.tensor.numpy() # P,h,w
                        pixel_plane_para = get_plane_parameters(nplanes_paras.copy(),nplanes_masks,mp3d=True)
                        depth_from_plane = plane2depth(pixel_plane_para,nplanes_masks,depth_map_np[fi],K_inv_dot_xy_1,h=self.cfg_h,w=self.cfg_w)
                        if final_vis:
                            depth_zeros = np.zeros(depth_map_np[fi].shape)
                            depth_from_plane = plane2depth(pixel_plane_para,nplanes_masks,depth_zeros,K_inv_dot_xy_1,h=self.cfg_h,w=self.cfg_w)
                        dataset_dict['depth_from_plane_and_nonplane'].append(torch.as_tensor(np.ascontiguousarray(depth_from_plane)))

            del dataset_dict['segments_info']
            del dataset_dict['mask_otherthan_plane']
            # print(dataset_dict.keys()) # dict_keys(['file_name', 'image', 'width', 'height', 'cam_pose', 'depth_map', 'depth_map_complete', 'K_inv_dot_xy_1', 'K_inv_dot_xy_1_2', 'K_inv_dot_xy_1_4', 'K_inv_dot_xy_1_8', 'instances', 'planes_paras', 'depth_from_plane', 'depth_from_plane_and_nonplane', 'mask_nonplane_valid', 'nonplane_num'])


            if final_vis:
                rnum = int(np.random.random()*100)
                for fi in range(int(os.environ['SAMPLING_FRAME_NUM'])):
                    print(fi,dataset_dict['file_name'][fi],dataset_dict['height'],dataset_dict['width'],dataset_dict['cam_pose'][fi])
                    print(fi,dataset_dict['instances'][fi].gt_classes)

                    image = dataset_dict['image'][fi].numpy().transpose(1,2,0)[:,:,[2,1,0]]
                    cv2.imwrite('./zmf_debug2/%d_f%d_1_image.png'%(rnum,fi),image)
                    depth_map = dataset_dict['depth_map'][fi]
                    drawDepth(depth_map, './zmf_debug2/%d_f%d_2_depth_sensor.png'%(rnum,fi))
                    depth_map_complete = dataset_dict['depth_map_complete'][fi]
                    drawDepth(depth_map_complete, './zmf_debug2/%d_f%d_2_depth_complete.png'%(rnum,fi))
                    depth_from_plane = dataset_dict['depth_from_plane'][fi]
                    drawDepth(depth_from_plane, './zmf_debug2/%d_f%d_3_depth_plane.png'%(rnum,fi))
                    depth_from_plane_and_nonplane = dataset_dict['depth_from_plane_and_nonplane'][fi]
                    drawDepth(depth_from_plane_and_nonplane, './zmf_debug2/%d_f%d_3_depth_plane_and_nonplane.png'%(rnum,fi))

                    num_nonplane = dataset_dict['nonplane_num'][fi]
                    planes_masks = dataset_dict['instances'][fi].gt_masks.tensor.numpy()[:-num_nonplane,:,:]
                    nonplanes_masks = dataset_dict['instances'][fi].gt_masks.tensor.numpy()[-num_nonplane:,:,:]
                    for pi in range(planes_masks.shape[0]):
                        mask = planes_masks[pi]
                        image[mask] = image[mask]/5.

                    for pi in range(nonplanes_masks.shape[0]):
                        mask = nonplanes_masks[pi]
                        color = np.ones(image.shape) * 255.
                        color[:,:,0] = color[:,:,0] * np.random.random()
                        color[:,:,1] = color[:,:,1] * np.random.random()
                        color[:,:,2] = color[:,:,2] * np.random.random()
                        image[mask] = color[mask]
                    cv2.imwrite('./zmf_debug2/%d_f%d_5_image.png'%(rnum,fi),image)



            return dataset_dict

        # parameters for nonplane
        seg_nonplane = os.environ['SEG_NONPLANE'] == 'True'
        seg_nonplane_num = int(os.environ['SEG_NONPLANE_NUM'])
        erode_kernel = self.cfg_h/float(os.environ['ERODE_FAC'])  # larger -> more erode
        start_disparity = 5.
        end_disparity = 0.01
        debug_nonplane = False
        final_vis = False

        # PREPARE 'image'
        if True:      
            dataset_dict['image'] = [] # src images
            for fi in range(len(dataset_dict['file_name'])):
                image = utils.read_image(dataset_dict['file_name'][fi], format=self.img_format)
                image = cv2.resize(image,(640,480)) # to match depth map and segmentation's original size
                utils.check_image_size(dataset_dict, image)
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
                dataset_dict['image'].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

        # PREPARE 'cam_pose'
        if True:
            dataset_dict['cam_pose'] = []
            cam_pose_invalid = False
            if os.environ['SAMPLING_FRAME_NUM'] == '2':
                if len(dataset_dict['file_name']) == 1:
                    print('ZMF ERROR',len(dataset_dict['file_name']),dataset_dict['file_name'])
                pose0_addr = dataset_dict['file_name'][0].replace('/color/','/pose/')[:-4]+'.txt'
                pose1_addr = dataset_dict['file_name'][1].replace('/color/','/pose/')[:-4]+'.txt'
                pose0 = np.loadtxt(pose0_addr)
                pose1 = np.loadtxt(pose1_addr)
                if not (np.isfinite(pose0).all() and np.isfinite(pose1).all()):
                    pose0 = np.array([
                        [1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.],
                    ])
                    pose1 = np.array([
                        [1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.],
                    ])
                    cam_pose_invalid = True
                pose0 = np.linalg.inv(pose0)
                pose1 = np.linalg.inv(pose1)
                dataset_dict['cam_pose'].append(torch.from_numpy(pose0))
                dataset_dict['cam_pose'].append(torch.from_numpy(pose1))
            else:
                pose0_addr = dataset_dict['file_name'][0].replace('/color/','/pose/')[:-4]+'.txt'
                pose0 = np.loadtxt(pose0_addr)
                if not np.isfinite(pose0).all():
                    pose0 = np.array([
                        [1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.],
                    ])
                    cam_pose_invalid = True
                pose0 = np.linalg.inv(pose0)
                dataset_dict['cam_pose'].append(torch.from_numpy(pose0))

        # PREPARE target view
        if True and os.environ['TRAIN_PHASE'] == '2':
            # TGT IMAGE
            src_idx_list = []
            for fi in range(len(dataset_dict['file_name'])):
                src_idx_list.append(int(os.path.basename(dataset_dict['file_name'][fi])[:-4]))
            src_idx = int(np.mean(src_idx_list))
            gap_idx = np.random.randint(-30,30)
            tgt_idx_1 = src_idx + gap_idx
            tgt_idx_2 = src_idx - gap_idx

            tgt_im_addr_1 = '%s/%d.jpg'%(os.path.dirname(dataset_dict['file_name'][0]),tgt_idx_1)
            tgt_im_addr_2 = '%s/%d.jpg'%(os.path.dirname(dataset_dict['file_name'][0]),tgt_idx_2)
            tgt_im_addr = tgt_im_addr_1
            if not os.path.exists(tgt_im_addr):
                tgt_im_addr = tgt_im_addr_2
                if not os.path.exists(tgt_im_addr):
                    tgt_im_addr = dataset_dict['file_name'][0]
            tgt_image = utils.read_image(tgt_im_addr, format=self.img_format)
            tgt_image = cv2.resize(tgt_image,(640,480)) # to match depth map and segmentation's original size
            tgt_image = transforms.apply_segmentation(tgt_image.astype(float))
            dataset_dict['tgt_image'] = torch.as_tensor(np.ascontiguousarray(tgt_image.transpose(2, 0, 1)))
            dataset_dict['tgt_filename'] = tgt_im_addr

            # print(3421,dataset_dict['file_name'],dataset_dict['tgt_filename'])

            # # SRC & TGT INTRINSIC
            # scale_fac = self.cfg_w/640.
            # intrinsic_general = np.array([
            #     [577.*scale_fac, 0., self.cfg_w/2.],
            #     [0., 577.*scale_fac, self.cfg_h/2.],
            #     [0.,0.,1.],
            # ])
            # dataset_dict['intrinsic'] = torch.from_numpy(intrinsic_general)

            # TGT POSE
            tgt_pose_addr = tgt_im_addr.replace('/color/','/pose/')[:-4]+'.txt'
            tgt_pose = np.loadtxt(tgt_pose_addr)
            if cam_pose_invalid or (not np.isfinite(tgt_pose).all()):
                tgt_pose = np.array([
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.],
                ])
                cam_pose_invalid = True
            tgt_pose = np.linalg.inv(tgt_pose)
            dataset_dict['tgt_cam_pose'] = torch.from_numpy(tgt_pose)

            # G_src_tgt
            dataset_dict['tgt_G_src_tgt'] = []
            for src_pose in dataset_dict['cam_pose']:
                this_G_src_tgt = src_pose.numpy() @ np.linalg.inv(tgt_pose)
                if np.isnan(this_G_src_tgt).any():
                    print('zmf: nan detected',dataset_dict['file_name'],tgt_im_addr)
                    exit()
                dataset_dict['tgt_G_src_tgt'].append(torch.from_numpy(this_G_src_tgt))
                # print('src',src_pose)
                # print('tgt',tgt_pose)
                # print('G',this_G_src_tgt)
                # print('G-1_at_src = tgt',np.linalg.inv(this_G_src_tgt)@src_pose.numpy())

                # # src tgt sm_G -> tm_G
                # # sm_G^-1 @ src = tgt_m
                # # tm_G = tgt @ tgt_m^-1

                # # G^-1 @ src = tgt
                # exit()
            
            # TGT DEPTH
            tgt_depth_addr = tgt_im_addr.replace('/color/','/depth/')[:-4]+'.png'
            tgt_depth_map = utils.read_image(tgt_depth_addr) / 1000.
            tgt_depth_map = transforms.apply_segmentation(tgt_depth_map.astype(float))
            dataset_dict['tgt_depth_map'] = torch.from_numpy(tgt_depth_map)
            
        # PREPARE 'depth_map'
        if True:
            dataset_dict['depth_map'] = []
            depth_map_np = []
            for fi in range(len(dataset_dict['file_name'])):
                depth_addr = dataset_dict['file_name'][fi].replace('/color/','/depth/')[:-4]+'.png'
                depth_map = utils.read_image(depth_addr) / 1000.
                depth_map = transforms.apply_segmentation(depth_map.astype(float))
                dataset_dict['depth_map'].append(torch.as_tensor(np.ascontiguousarray(depth_map)))
                depth_map_np.append(depth_map)

        # PREPARE 'completed depth_map'
        if True and seg_nonplane:
            dataset_dict['depth_map_complete'] = []
            depth_map_complete_np = []
            for fi in range(len(dataset_dict['file_name'])):
                depth_addr_complete = dataset_dict['file_name'][fi].replace('/color/','/depth2/')[:-4]+'.png'
                depth_map_complete = utils.read_image(depth_addr_complete) / 1000.
                depth_map_complete = transforms.apply_segmentation(depth_map_complete.astype(float))
                dataset_dict['depth_map_complete'].append(torch.as_tensor(np.ascontiguousarray(depth_map_complete)))
                depth_map_complete_np.append(depth_map_complete)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        # dataset_dict['image'] = [torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("annotations", None)
        #     return dataset_dict

        # PREPARE 'K_inv_dot_xy_1'
        if True:
            K_inv_dot_xy_1,K_inv_dot_xy_1_2,K_inv_dot_xy_1_4,K_inv_dot_xy_1_8 = precompute_K_inv_dot_xy_1(h=self.cfg_h, w=self.cfg_w)
            dataset_dict['K_inv_dot_xy_1'] = torch.from_numpy(K_inv_dot_xy_1)
            dataset_dict['K_inv_dot_xy_1_2'] = torch.from_numpy(K_inv_dot_xy_1_2)
            dataset_dict['K_inv_dot_xy_1_4'] = torch.from_numpy(K_inv_dot_xy_1_4)
            dataset_dict['K_inv_dot_xy_1_8'] = torch.from_numpy(K_inv_dot_xy_1_8)

        # PREPARE FROM segments_info
        if True:
            dataset_dict['instances'] = []
            dataset_dict['planes_paras'] = []
            dataset_dict['depth_from_plane'] = []
            dataset_dict['depth_from_plane_and_nonplane'] = []
            dataset_dict['mask_nonplane_valid'] = []
            if seg_nonplane:
                dataset_dict['mask_otherthan_plane'] = []
                dataset_dict['nonplane_num'] = []

            for fi in range(len(dataset_dict['file_name'])):
                segments_info = dataset_dict['segments_info'][fi]
                planes_masks = []
                planes_paras = []
                have_plane = False
                if seg_nonplane:
                    nonplanes_masks = []
                    nonplanes_paras = []
                    have_nonplane = False
                
                
                # PRE-PREPARE plane masks & para
                if True:
                    for obj in segments_info:
                        # print(obj.keys()) # dict_keys(['id', 'image_id', 'category_id', 'iscrowd', 'area', 'bbox', 'segmentation', 'width', 'height', 'plane_paras', 'isthing'])
                        plane_paras = obj['plane_paras']
                        planes_paras.append(plane_paras)
                        
                        polygons = obj['segmentation']
                        oneplane_mask = np.zeros((640,480), dtype=np.bool)
                        for poly in polygons:
                            # print(poly) # [580.0, 252.5, 581.0, 243.5, 580.0, 252.5]
                            if len(poly) > 0:
                                have_plane = True
                                this_mask = poly2mask(poly[::2],poly[1::2],(640,480))
                                oneplane_mask = np.logical_or(oneplane_mask, this_mask)
                        planes_masks.append(oneplane_mask)
                        
                    if have_plane:
                        planes_paras = np.stack(np.array(planes_paras),axis=0)
                        planes_masks = np.stack(np.array(planes_masks),axis=0)
                        planes_masks = planes_masks.transpose(2,1,0)
                        planes_masks = transforms.apply_segmentation(planes_masks.astype(float))
                        planes_masks = planes_masks.astype(bool)
                        planes_masks = planes_masks.transpose(2,0,1)
                        # for i in range(planes_masks.shape[0]):
                        #     im = planes_masks[i].astype(float)
                        #     cv2.imwrite('./zmf_show/mask_%d.png'%i,(im*255).astype('uint8'))
                
                dataset_dict['planes_paras'].append(torch.from_numpy(planes_paras))

                # PREPARE 'mask_nonplane_valid'
                if True:
                    mask_nonplane_valid = depth_map_np[fi] > 1e-3
                    for i in range(planes_masks.shape[0]):
                        mask_nonplane_valid = np.logical_and(mask_nonplane_valid,np.logical_not(planes_masks[i]))
                    # drawDepth(torch.from_numpy(mask_nonplane_valid.astype(np.float)),'./see_mask_nonplane_valid.png')
                    dataset_dict['mask_nonplane_valid'].append(torch.as_tensor(np.ascontiguousarray(mask_nonplane_valid)))

                # otherthan plane mask
                if True and seg_nonplane:
                    if debug_nonplane:
                        drawDepth(torch.from_numpy(depth_map_np[fi]),'./zmf_debug/1_depth_map_sensor.png')
                        drawDepth(torch.from_numpy(depth_map_complete_np[fi]),'./zmf_debug/2_depth_map_complete.png')
                    mask_otherthan_plane = depth_map_complete_np[fi] > 1e-3
                    for i in range(planes_masks.shape[0]):
                        mask_otherthan_plane = np.logical_and(mask_otherthan_plane,np.logical_not(planes_masks[i]))
                    dataset_dict['mask_otherthan_plane'].append(torch.as_tensor(np.ascontiguousarray(mask_otherthan_plane)))
                    if debug_nonplane:
                        drawDepth(torch.from_numpy(mask_otherthan_plane.astype(np.float)),'./zmf_debug/3_otherthan_plane_mask.png')
                
                # 
                if True and seg_nonplane:
                    if erode_kernel < 2: erode_kernel =  2
                    erode_kernel = int(erode_kernel) if int(erode_kernel) % 2 == 1 else int(erode_kernel)+1
                    max_pool = torch.nn.MaxPool2d(kernel_size=erode_kernel, stride=1, padding=int((erode_kernel-1)/2))
                    eroded_mask = -max_pool(-dataset_dict['mask_otherthan_plane'][fi][None,None,...].float())
                    mask_otherthan_plane = max_pool(eroded_mask)[0,0]
                    if debug_nonplane:
                        drawDepth(mask_otherthan_plane,'./zmf_debug/4_refine_mask.png')

                    depth_map_complete = dataset_dict['depth_map_complete'][fi]
                    mask_otherthan_plane = mask_otherthan_plane > 1e-3
                    non_plane_depth_map = depth_map_complete * mask_otherthan_plane

                    if debug_nonplane:
                        drawDepth(non_plane_depth_map,'./zmf_debug/5_non_plane_depth.png')
                    
                    
                    disparity_S = torch.linspace(
                        start_disparity, end_disparity, seg_nonplane_num
                    )
                    depth_S = torch.reciprocal(disparity_S)
                    # print(disparity_S)
                    # print(depth_S)

                    for di in range(seg_nonplane_num):
                        this_depth_min = depth_S[di-1] if di != 0 else 0.
                        this_depth_max = depth_S[di]
                        
                        this_nonplane_mask_1 = non_plane_depth_map > this_depth_min
                        this_nonplane_mask_2 = non_plane_depth_map < this_depth_max
                        this_nonplane_mask = torch.logical_and(this_nonplane_mask_1,this_nonplane_mask_2)

                        if dataset_dict['depth_map_complete'][fi][this_nonplane_mask].shape[0] == 0:
                            continue
                        
                        have_nonplane = True
                        this_depth = torch.mean(dataset_dict['depth_map_complete'][fi][this_nonplane_mask])
                        n_3 = torch.tensor([0, 0, 1])
                        this_nonplane_para = n_3 / this_depth

                        nonplanes_masks.append(this_nonplane_mask)
                        nonplanes_paras.append(this_nonplane_para)
                    
                    if have_nonplane:
                        nonplanes_paras = torch.stack(nonplanes_paras)
                        nonplanes_masks = torch.stack(nonplanes_masks)

                        offset = torch.reciprocal(torch.norm(nonplanes_paras, dim=1, keepdim=True))
                        nonplanes_paras = nonplanes_paras * offset
                        nonplanes_paras = torch.cat([nonplanes_paras,offset],dim=1)
                        dataset_dict['planes_paras'][fi] = torch.cat([dataset_dict['planes_paras'][fi], nonplanes_paras],dim=0)

                    
                
                # PREPARE 'depth_from_plane'
                if True:
                    # print(planes_paras.shape) # P, 4
                    # print(planes_masks.shape) # P, h, w
                    pixel_plane_para = get_plane_parameters(planes_paras.copy(),planes_masks)
                    depth_from_plane = plane2depth(pixel_plane_para,planes_masks,depth_map_np[fi],K_inv_dot_xy_1,h=self.cfg_h,w=self.cfg_w)
                    if final_vis:
                        depth_zeros = np.zeros(depth_map_np[fi].shape)
                        depth_from_plane = plane2depth(pixel_plane_para,planes_masks,depth_zeros,K_inv_dot_xy_1,h=self.cfg_h,w=self.cfg_w)
                    depth_from_plane[depth_from_plane>20.] = 20.
                    depth_from_plane[depth_from_plane<0.] = 0.
                    # if check_tensor_naninf(torch.from_numpy(depth_from_plane).half(),'load depth'): print(dataset_dict['file_name'])
                    # drawDepth(torch.from_numpy(depth_map_np[fi]),'./zmf_show/depthmap_sensor.png')
                    # drawDepth(torch.from_numpy(depth_from_plane),'./zmf_show/depthmap.png')
                    dataset_dict['depth_from_plane'].append(torch.as_tensor(np.ascontiguousarray(depth_from_plane)))

                # PREPARE 'instances'
                if True:
                    instances = Instances([self.cfg_h,self.cfg_w])
                    classes = []
                    masks = []
                    for segment_info in segments_info:
                        class_id = 0 # segment_info['category_id']
                        classes.append(class_id)
                    if seg_nonplane and have_nonplane:
                        for tmp in nonplanes_paras:
                            class_id = 50 # segment_info["category_id"]
                            classes.append(class_id)
                    classes = np.array(classes)
                    instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

                    if not have_plane:
                        # Some image does not have annotation (all ignored)
                        instances.gt_masks = torch.zeros((0, self.cfg_h, self.cfg_w))
                        instances.gt_boxes = Boxes(torch.zeros((0, 4)))
                        print('zmf have no plane anno', dataset_dict['file_name'])
                    else:
                        plane_mask = torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in planes_masks])
                        if seg_nonplane and have_nonplane:
                            nonplane_mask = torch.stack([x for x in nonplanes_masks])
                            plane_mask = torch.cat((plane_mask,nonplane_mask),dim=0)
                        masks = BitMasks(plane_mask)
                        instances.gt_masks = masks #.tensor 
                        instances.gt_boxes = masks.get_bounding_boxes()

                    dataset_dict['instances'].append(instances)
                    if have_nonplane:
                        dataset_dict['nonplane_num'].append(nonplanes_paras.shape[0])
                    else:
                        dataset_dict['nonplane_num'].append(0)
                    
                # PREPARE 'depth_from_plane_and_nonplane'
                if True:
                    nplanes_paras = dataset_dict['planes_paras'][fi].numpy() # P, 4
                    nplanes_masks = dataset_dict['instances'][fi].gt_masks.tensor.numpy() # P,h,w
                    pixel_plane_para = get_plane_parameters(nplanes_paras.copy(),nplanes_masks)
                    depth_from_plane = plane2depth(pixel_plane_para,nplanes_masks,depth_map_np[fi],K_inv_dot_xy_1,h=self.cfg_h,w=self.cfg_w)
                    if final_vis:
                        depth_zeros = np.zeros(depth_map_np[fi].shape)
                        depth_from_plane = plane2depth(pixel_plane_para,nplanes_masks,depth_zeros,K_inv_dot_xy_1,h=self.cfg_h,w=self.cfg_w)
                    dataset_dict['depth_from_plane_and_nonplane'].append(torch.as_tensor(np.ascontiguousarray(depth_from_plane)))

        del dataset_dict['image_id']
        del dataset_dict['segments_info']
        del dataset_dict['mask_otherthan_plane']
        # print(dataset_dict.keys()) # dict_keys(['file_name', 'image', 'width', 'height', 'cam_pose', 'depth_map', 'depth_map_complete', 'K_inv_dot_xy_1', 'K_inv_dot_xy_1_2', 'K_inv_dot_xy_1_4', 'K_inv_dot_xy_1_8', 'instances', 'planes_paras', 'depth_from_plane', 'depth_from_plane_and_nonplane', 'mask_nonplane_valid', 'nonplane_num'])


        if final_vis:
            rnum = int(np.random.random()*100)
            for fi in range(int(os.environ['SAMPLING_FRAME_NUM'])):
                print(fi,dataset_dict['file_name'][fi],dataset_dict['height'],dataset_dict['width'],dataset_dict['cam_pose'][fi])
                print(fi,dataset_dict['instances'][fi].gt_classes)

                image = dataset_dict['image'][fi].numpy().transpose(1,2,0)[:,:,[2,1,0]]
                cv2.imwrite('./zmf_debug2/%d_f%d_1_image.png'%(rnum,fi),image)
                depth_map = dataset_dict['depth_map'][fi]
                drawDepth(depth_map, './zmf_debug2/%d_f%d_2_depth_sensor.png'%(rnum,fi))
                depth_map_complete = dataset_dict['depth_map_complete'][fi]
                drawDepth(depth_map_complete, './zmf_debug2/%d_f%d_2_depth_complete.png'%(rnum,fi))
                depth_from_plane = dataset_dict['depth_from_plane'][fi]
                drawDepth(depth_from_plane, './zmf_debug2/%d_f%d_3_depth_plane.png'%(rnum,fi))
                depth_from_plane_and_nonplane = dataset_dict['depth_from_plane_and_nonplane'][fi]
                drawDepth(depth_from_plane_and_nonplane, './zmf_debug2/%d_f%d_3_depth_plane_and_nonplane.png'%(rnum,fi))

                num_nonplane = dataset_dict['nonplane_num'][fi]
                planes_masks = dataset_dict['instances'][fi].gt_masks.tensor.numpy()[:-num_nonplane,:,:]
                nonplanes_masks = dataset_dict['instances'][fi].gt_masks.tensor.numpy()[-num_nonplane:,:,:]
                for pi in range(planes_masks.shape[0]):
                    mask = planes_masks[pi]
                    image[mask] = image[mask]/5.

                for pi in range(nonplanes_masks.shape[0]):
                    mask = nonplanes_masks[pi]
                    color = np.ones(image.shape) * 255.
                    color[:,:,0] = color[:,:,0] * np.random.random()
                    color[:,:,1] = color[:,:,1] * np.random.random()
                    color[:,:,2] = color[:,:,2] * np.random.random()
                    image[mask] = color[mask]
                cv2.imwrite('./zmf_debug2/%d_f%d_5_image.png'%(rnum,fi),image)

        
        if cam_pose_invalid:
            pose0 = np.array([
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ])
            pose0 = np.linalg.inv(pose0)
            dataset_dict['cam_pose'][0] = torch.from_numpy(pose0)
            for k in dataset_dict:
                if type(dataset_dict[k]) == list and len(dataset_dict[k]) == 2:
                    dataset_dict[k][1] = dataset_dict[k][0]
        return dataset_dict
