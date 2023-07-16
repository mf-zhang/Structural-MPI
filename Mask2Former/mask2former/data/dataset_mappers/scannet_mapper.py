# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging
from traceback import print_tb

import numpy as np
import torch, cv2, os

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Boxes, Instances

from skimage import draw

__all__ = ["ScanNetMapper"]

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

def get_plane_parameters(plane, segmentation):
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
        plane_understand_only,
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
        self.h = h
        self.w = w
        self.plane_understand_only = plane_understand_only

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        (h,w) = cfg.INPUT.IMAGE_SIZE

        plane_understand_only = cfg.PLANE_UNDERSTAND_ONLY
        os.environ['PLANE_UNDERSTAND_ONLY'] = 'True' if plane_understand_only else 'False'

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "h": h,
            "w": w,
            "plane_understand_only": plane_understand_only,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        image = cv2.resize(image,(640,480)) # to match depth map and segmentation's original size
        utils.check_image_size(dataset_dict, image)
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)

        if not self.plane_understand_only:
            # TGT IMAGE
            src_idx = int(os.path.basename(dataset_dict["file_name"])[:-4])
            gap_idx = np.random.randint(30)
            tgt_idx = src_idx + gap_idx
            tgt_im_addr = '%s/%d.jpg'%(os.path.dirname(dataset_dict["file_name"]),tgt_idx)
            if not os.path.exists(tgt_im_addr):
                tgt_im_addr = dataset_dict["file_name"]
            tgt_image = utils.read_image(tgt_im_addr, format=self.img_format)
            tgt_image = cv2.resize(tgt_image,(640,480)) # to match depth map and segmentation's original size
            tgt_image = transforms.apply_segmentation(tgt_image.astype(float))
            dataset_dict["tgt_image"] = torch.as_tensor(np.ascontiguousarray(tgt_image.transpose(2, 0, 1)))
            dataset_dict["tgt_filename"] = tgt_im_addr

            # SRC & TGT INTRINSIC
            scale_fac = self.w/640.
            intrinsic_general = np.array([
                [577.*scale_fac, 0., self.w/2.],
                [0., 577.*scale_fac, self.h/2.],
                [0.,0.,1.],
            ])
            dataset_dict["intrinsic"] = torch.from_numpy(intrinsic_general)

            # SRC & TGT POSE, G
            pose_addr = dataset_dict["file_name"].replace('/color/','/pose/')[:-4]+'.txt'
            tgt_pose_addr = tgt_im_addr.replace('/color/','/pose/')[:-4]+'.txt'
            pose = np.loadtxt(pose_addr)
            tgt_pose = np.loadtxt(tgt_pose_addr)
            if not (np.isfinite(pose).all() and np.isfinite(tgt_pose).all()):
                pose = np.array([
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.],
                ])
                tgt_pose = np.array([
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.],
                ])
            pose = np.linalg.inv(pose)
            tgt_pose = np.linalg.inv(tgt_pose)
            G_src_tgt = pose @ np.linalg.inv(tgt_pose)
            if np.isnan(G_src_tgt).any():
                print('zmf: nan detected')
                print(G_src_tgt,pose,tgt_pose)
                print(dataset_dict["file_name"],tgt_im_addr)
                exit()
            dataset_dict["G_src_tgt"] = torch.from_numpy(G_src_tgt)
            
            # TGT DEPTH
            tgt_depth_addr = tgt_im_addr.replace('/color/','/depth/')[:-4]+'.png'
            tgt_depth_map = utils.read_image(tgt_depth_addr) / 1000.
            tgt_depth_map = transforms.apply_segmentation(tgt_depth_map.astype(float))
            dataset_dict["tgt_depth_map"] = torch.from_numpy(tgt_depth_map)
            


        depth_addr = dataset_dict["file_name"].replace('/color/','/depth/')[:-4]+'.png'
        depth_map = utils.read_image(depth_addr) / 1000.
        depth_map = transforms.apply_segmentation(depth_map.astype(float))


        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        segments_info = dataset_dict["segments_info"]
        nplanes_masks = []
        nplanes_paras = []
        have_plane = False
        
        for obj in segments_info:
            plane_paras = obj['plane_paras']
            nplanes_paras.append(plane_paras)
            
            polygons = obj['segmentation']
            oneplane_mask = np.zeros((640,480), dtype=np.bool)
            for poly in polygons:
                if len(poly) > 0:
                    have_plane = True
                    this_mask = poly2mask(poly[::2],poly[1::2],(640,480))
                    oneplane_mask = np.logical_or(oneplane_mask, this_mask)
            nplanes_masks.append(oneplane_mask)
            
        
        if have_plane:
            nplanes_paras = np.stack(np.array(nplanes_paras),axis=0)

            nplanes_masks = np.stack(np.array(nplanes_masks),axis=0)
            nplanes_masks = np.swapaxes(nplanes_masks,1,2)
            nplanes_masks = np.swapaxes(nplanes_masks,0,1)
            nplanes_masks = np.swapaxes(nplanes_masks,1,2)

            nplanes_masks = transforms.apply_segmentation(nplanes_masks.astype(float))
            nplanes_masks = nplanes_masks.astype(bool)

            nplanes_masks = np.swapaxes(nplanes_masks,2,1)
            nplanes_masks = np.swapaxes(nplanes_masks,1,0)


        mask_nonplane_valid = depth_map > 1e-3
        for i in range(nplanes_masks.shape[0]):
            mask_nonplane_valid = np.logical_and(mask_nonplane_valid,np.logical_not(nplanes_masks[i]))
        

        K_inv_dot_xy_1,K_inv_dot_xy_1_2,K_inv_dot_xy_1_4,K_inv_dot_xy_1_8 = precompute_K_inv_dot_xy_1(h=self.h, w=self.w)


        pixel_plane_para = get_plane_parameters(nplanes_paras.copy(),nplanes_masks)
        depth_from_plane = plane2depth(pixel_plane_para,nplanes_masks,depth_map,K_inv_dot_xy_1,h=self.h,w=self.w)

        depth_from_plane[depth_from_plane>100.] = 100.
        depth_from_plane[depth_from_plane<-100.] = -100.


        instances = Instances(image_shape)
        classes = []
        masks = []
        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            # print(class_id, segment_info["iscrowd"]) # 1 0
            if not segment_info["iscrowd"]:
                classes.append(class_id)
        classes = np.array(classes)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)



        if not have_plane:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            print('zmf have no plane anno', dataset_dict["file_name"])
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in nplanes_masks])
            )
            instances.gt_masks = masks.tensor
            instances.gt_boxes = masks.get_bounding_boxes()

        dataset_dict["instances"] = instances
        dataset_dict["depth_map"] = torch.from_numpy(depth_map)
        dataset_dict["planes_paras"] = torch.from_numpy(nplanes_paras)
        dataset_dict["K_inv_dot_xy_1"] = torch.from_numpy(K_inv_dot_xy_1)
        dataset_dict["K_inv_dot_xy_1_2"] = torch.from_numpy(K_inv_dot_xy_1_2)
        dataset_dict["K_inv_dot_xy_1_4"] = torch.from_numpy(K_inv_dot_xy_1_4)
        dataset_dict["K_inv_dot_xy_1_8"] = torch.from_numpy(K_inv_dot_xy_1_8)
        dataset_dict["pixel_plane_para"] = torch.from_numpy(pixel_plane_para)
        dataset_dict["depth_from_plane"] = torch.from_numpy(depth_from_plane)
        dataset_dict["filename"] = dataset_dict["file_name"]
        dataset_dict["mask_nonplane_valid"] = torch.from_numpy(mask_nonplane_valid)


        return dataset_dict
