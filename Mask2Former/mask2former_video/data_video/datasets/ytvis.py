# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import contextlib
import io
import json
import logging
import numpy as np
import os
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse YTVIS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_ytvis_json", "register_ytvis_instances"]


YTVIS_CATEGORIES_2019 = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [0, 82, 0], "isthing": 1, "id": 2, "name": "giant_panda"},
    {"color": [119, 11, 32], "isthing": 1, "id": 3, "name": "lizard"},
    {"color": [165, 42, 42], "isthing": 1, "id": 4, "name": "parrot"},
    {"color": [134, 134, 103], "isthing": 1, "id": 5, "name": "skateboard"},
    {"color": [0, 0, 142], "isthing": 1, "id": 6, "name": "sedan"},
    {"color": [255, 109, 65], "isthing": 1, "id": 7, "name": "ape"},
    {"color": [0, 226, 252], "isthing": 1, "id": 8, "name": "dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 9, "name": "snake"},
    {"color": [0, 60, 100], "isthing": 1, "id": 10, "name": "monkey"},
    {"color": [250, 170, 30], "isthing": 1, "id": 11, "name": "hand"},
    {"color": [100, 170, 30], "isthing": 1, "id": 12, "name": "rabbit"},
    {"color": [179, 0, 194], "isthing": 1, "id": 13, "name": "duck"},
    {"color": [255, 77, 255], "isthing": 1, "id": 14, "name": "cat"},
    {"color": [120, 166, 157], "isthing": 1, "id": 15, "name": "cow"},
    {"color": [73, 77, 174], "isthing": 1, "id": 16, "name": "fish"},
    {"color": [0, 80, 100], "isthing": 1, "id": 17, "name": "train"},
    {"color": [182, 182, 255], "isthing": 1, "id": 18, "name": "horse"},
    {"color": [0, 143, 149], "isthing": 1, "id": 19, "name": "turtle"},
    {"color": [174, 57, 255], "isthing": 1, "id": 20, "name": "bear"},
    {"color": [0, 0, 230], "isthing": 1, "id": 21, "name": "motorbike"},
    {"color": [72, 0, 118], "isthing": 1, "id": 22, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 23, "name": "leopard"},
    {"color": [0, 125, 92], "isthing": 1, "id": 24, "name": "fox"},
    {"color": [209, 0, 151], "isthing": 1, "id": 25, "name": "deer"},
    {"color": [188, 208, 182], "isthing": 1, "id": 26, "name": "owl"},
    {"color": [145, 148, 174], "isthing": 1, "id": 27, "name": "surfboard"},
    {"color": [106, 0, 228], "isthing": 1, "id": 28, "name": "airplane"},
    {"color": [0, 0, 70], "isthing": 1, "id": 29, "name": "truck"},
    {"color": [199, 100, 0], "isthing": 1, "id": 30, "name": "zebra"},
    {"color": [166, 196, 102], "isthing": 1, "id": 31, "name": "tiger"},
    {"color": [110, 76, 0], "isthing": 1, "id": 32, "name": "elephant"},
    {"color": [133, 129, 255], "isthing": 1, "id": 33, "name": "snowboard"},
    {"color": [0, 0, 192], "isthing": 1, "id": 34, "name": "boat"},
    {"color": [183, 130, 88], "isthing": 1, "id": 35, "name": "shark"},
    {"color": [130, 114, 135], "isthing": 1, "id": 36, "name": "mouse"},
    {"color": [107, 142, 35], "isthing": 1, "id": 37, "name": "frog"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "eagle"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "earless_seal"},
    {"color": [255, 208, 186], "isthing": 1, "id": 40, "name": "tennis_racket"},
]


YTVIS_CATEGORIES_2021 = [
    {"color": [106, 0, 228], "isthing": 1, "id": 1, "name": "airplane"},
    {"color": [174, 57, 255], "isthing": 1, "id": 2, "name": "bear"},
    {"color": [255, 109, 65], "isthing": 1, "id": 3, "name": "bird"},
    {"color": [0, 0, 192], "isthing": 1, "id": 4, "name": "boat"},
    {"color": [0, 0, 142], "isthing": 1, "id": 5, "name": "car"},
    {"color": [255, 77, 255], "isthing": 1, "id": 6, "name": "cat"},
    {"color": [120, 166, 157], "isthing": 1, "id": 7, "name": "cow"},
    {"color": [209, 0, 151], "isthing": 1, "id": 8, "name": "deer"},
    {"color": [0, 226, 252], "isthing": 1, "id": 9, "name": "dog"},
    {"color": [179, 0, 194], "isthing": 1, "id": 10, "name": "duck"},
    {"color": [174, 255, 243], "isthing": 1, "id": 11, "name": "earless_seal"},
    {"color": [110, 76, 0], "isthing": 1, "id": 12, "name": "elephant"},
    {"color": [73, 77, 174], "isthing": 1, "id": 13, "name": "fish"},
    {"color": [250, 170, 30], "isthing": 1, "id": 14, "name": "flying_disc"},
    {"color": [0, 125, 92], "isthing": 1, "id": 15, "name": "fox"},
    {"color": [107, 142, 35], "isthing": 1, "id": 16, "name": "frog"},
    {"color": [0, 82, 0], "isthing": 1, "id": 17, "name": "giant_panda"},
    {"color": [72, 0, 118], "isthing": 1, "id": 18, "name": "giraffe"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [255, 179, 240], "isthing": 1, "id": 20, "name": "leopard"},
    {"color": [119, 11, 32], "isthing": 1, "id": 21, "name": "lizard"},
    {"color": [0, 60, 100], "isthing": 1, "id": 22, "name": "monkey"},
    {"color": [0, 0, 230], "isthing": 1, "id": 23, "name": "motorbike"},
    {"color": [130, 114, 135], "isthing": 1, "id": 24, "name": "mouse"},
    {"color": [165, 42, 42], "isthing": 1, "id": 25, "name": "parrot"},
    {"color": [220, 20, 60], "isthing": 1, "id": 26, "name": "person"},
    {"color": [100, 170, 30], "isthing": 1, "id": 27, "name": "rabbit"},
    {"color": [183, 130, 88], "isthing": 1, "id": 28, "name": "shark"},
    {"color": [134, 134, 103], "isthing": 1, "id": 29, "name": "skateboard"},
    {"color": [5, 121, 0], "isthing": 1, "id": 30, "name": "snake"},
    {"color": [133, 129, 255], "isthing": 1, "id": 31, "name": "snowboard"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "squirrel"},
    {"color": [145, 148, 174], "isthing": 1, "id": 33, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 34, "name": "tennis_racket"},
    {"color": [166, 196, 102], "isthing": 1, "id": 35, "name": "tiger"},
    {"color": [0, 80, 100], "isthing": 1, "id": 36, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 37, "name": "truck"},
    {"color": [0, 143, 149], "isthing": 1, "id": 38, "name": "turtle"},
    {"color": [0, 228, 0], "isthing": 1, "id": 39, "name": "whale"},
    {"color": [199, 100, 0], "isthing": 1, "id": 40, "name": "zebra"},
]


def _get_ytvis_2019_instances_meta():
    thing_ids = [k["id"] for k in YTVIS_CATEGORIES_2019 if k["isthing"] == 1]
    thing_colors = [k["color"] for k in YTVIS_CATEGORIES_2019 if k["isthing"] == 1]
    assert len(thing_ids) == 40, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in YTVIS_CATEGORIES_2019 if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_ytvis_2021_instances_meta():
    thing_ids = [k["id"] for k in YTVIS_CATEGORIES_2021 if k["isthing"] == 1]
    thing_colors = [k["color"] for k in YTVIS_CATEGORIES_2021 if k["isthing"] == 1]
    assert len(thing_ids) == 40, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in YTVIS_CATEGORIES_2021 if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def get_coco_metadata():
    from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in COCO_CATEGORIES]
    thing_colors = [k["color"] for k in COCO_CATEGORIES]
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]

    for i in range(50):    
        thing_classes[i] = 'p%d'%i
        # thing_colors[i] = [0,0,180]
    for i in range(50,100):
        thing_classes[i] = 'n%d'%(i-50)
        # thing_colors[i] = [0,0,180]


    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        else:
            stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # # in order to use sem_seg evaluator
        # stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def load_ytvis_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    from .ytvis_api.ytvos import YTVOS

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        ytvis_api = YTVOS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(ytvis_api.getCatIds())
        cats = ytvis_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
                    Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
                    """
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    vid_ids = sorted(ytvis_api.vids.keys())
    # vids is a list of dicts, each looks something like:
    # {'license': 1,
    #  'flickr_url': ' ',
    #  'file_names': ['ff25f55852/00000.jpg', 'ff25f55852/00005.jpg', ..., 'ff25f55852/00175.jpg'],
    #  'height': 720,
    #  'width': 1280,
    #  'length': 36,
    #  'date_captured': '2019-04-11 00:55:41.903902',
    #  'id': 2232}
    vids = ytvis_api.loadVids(vid_ids)

    anns = [ytvis_api.vidToAnns[vid_id] for vid_id in vid_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(ytvis_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    vids_anns = list(zip(vids, anns))
    logger.info("Loaded {} videos in YTVIS format from {}".format(len(vids_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "category_id", "id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (vid_dict, anno_dict_list) in vids_anns:
        record = {}
        record["file_names"] = [os.path.join(image_root, vid_dict["file_names"][i]) for i in range(vid_dict["length"])]
        record["height"] = vid_dict["height"]
        record["width"] = vid_dict["width"]
        record["length"] = vid_dict["length"]
        video_id = record["video_id"] = vid_dict["id"]

        video_objs = []
        for frame_idx in range(record["length"]):
            frame_objs = []
            for anno in anno_dict_list:
                assert anno["video_id"] == video_id

                obj = {key: anno[key] for key in ann_keys if key in anno}

                _bboxes = anno.get("bboxes", None)
                _segm = anno.get("segmentations", None)

                if not (_bboxes and _segm and _bboxes[frame_idx] and _segm[frame_idx]):
                    continue

                bbox = _bboxes[frame_idx]
                segm = _segm[frame_idx]

                obj["bbox"] = bbox
                obj["bbox_mode"] = BoxMode.XYWH_ABS

                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                elif segm:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

                if id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                frame_objs.append(obj)
            video_objs.append(frame_objs)
        record["annotations"] = video_objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts

def load_scannet_json(json_file_scan, image_dir_scan, pairs_txt):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    if os.environ['TEST_NVS_ONLY'] == 'True':
        f_test = open(os.environ['TEST_NVS_TXT'],'r')
        test_lines = f_test.readlines()
        ret_scan = []
        src1l,src2l,tgtl = [],[],[]
        for test_line in test_lines:
            src1,src2,tgt1 = test_line.split()
            if os.environ['TEST_SRC_NUM'] == '2':
                # print(src1,src2,tgt1)
                # src1idx = int(os.path.basename(src1)[:-4])
                # src2idx = int(os.path.basename(src2)[:-4])
                # tgtidx = int(os.path.basename(tgt1)[:-4])
                
                # if abs(src2idx - tgtidx) > abs(src1idx - tgtidx):
                #     src2idx = tgtidx - (src1idx - tgtidx)
                #     src2 = '%s/%d.jpg'%(os.path.dirname(src2),src2idx)
                # src1l.append(src1idx)
                # src2l.append(src2idx)
                # tgtl.append(tgtidx)

                ret_scan.append(
                    {
                        "file_name": [src1, src2],
                        "tgt_file_name": [tgt1],
                    }
                )
            else:
                ret_scan.append(
                    {
                        "file_name": [src1],
                        "tgt_file_name": [tgt1],
                    }
                )
        
        gap_src1_tgt = []
        for i in range(len(src1l)):
            gap_src1_tgt.append(abs(tgtl[i]-src1l[i]))
        gap_src2_tgt = []
        for i in range(len(src2l)):
            gap_src2_tgt.append(abs(tgtl[i]-src2l[i]))
        gap_src1_src2 = []
        for i in range(len(src2l)):
            gap_src1_src2.append(abs(src1l[i]-src2l[i]))

        gap_allsrc_tgt = gap_src1_tgt + gap_src2_tgt




        
        return ret_scan

    if os.environ['TRAIN_MP3D'] == 'True':
        print(json_file_scan, image_dir_scan, pairs_txt)

        with PathManager.open(json_file_scan) as f_scan:
            json_info_scan = json.load(f_scan)
            print('mp3d json load finished')

        ret_scan = []

        for one_data in json_info_scan['data']:

            file_name0 = one_data['0']['file_name'].replace('/Pool1/users/jinlinyi/dataset/mp3d_rpnet_v4_sep20',image_dir_scan)
            file_name1 = one_data['1']['file_name'].replace('/Pool1/users/jinlinyi/dataset/mp3d_rpnet_v4_sep20',image_dir_scan)

            camera0 = one_data['0']['camera']
            camera1 = one_data['1']['camera']

            annotations0 = one_data['0']['annotations']
            annotations1 = one_data['1']['annotations']

            gt_corrs = one_data['gt_corrs']
            rel_pose = one_data['rel_pose']

            if os.environ['SAMPLING_FRAME_NUM'] == '1':
                ret_scan.append(
                    {
                        "file_name": [file_name0],
                        "camera": [camera0],
                        # "image_id": [img_id0],
                        "segments_info": [annotations0],
                        "gt_corrs": gt_corrs,
                        "rel_pose": rel_pose,
                    }
                )

            if os.environ['SAMPLING_FRAME_NUM'] == '2':
                ret_scan.append(
                    {
                        "file_name": [file_name0, file_name1],
                        "camera": [camera0, camera1],
                        # "image_id": [img_id0,img_id0],
                        "segments_info": [annotations0,annotations1],
                        "gt_corrs": gt_corrs,
                        "rel_pose": rel_pose,
                    }
                )

        return ret_scan
        
    # more evaluation samples
    more_eval_samples = False
    if 'val' in pairs_txt and os.environ['SAMPLING_FRAME_NUM'] == '1':
        pairs_txt = '%s/2frame_val_30.txt'%os.path.dirname(pairs_txt)
        more_eval_samples = True

    pairs = []
    with open(pairs_txt,'r') as fpair:
        lines = fpair.readlines()
    for line in lines:
        img_id0 = int(line.split()[0])
        img_id1 = int(line.split()[1])
        img_addr0 = line.split()[2]
        scene0 = int(img_addr0.split('_')[0][-4:])
        scene1 = int(img_addr0.split('_')[1][:2])
        img_num0 = int(line.split()[3])
        img_num1 = int(line.split()[4])
        pairs.append([img_id0,img_id1,scene0,scene1])
        if more_eval_samples:
            pairs.append([img_id1,img_id1,scene1,scene1])

    with PathManager.open(json_file_scan) as f_scan:
        json_info_scan = json.load(f_scan)
        print('scan json load finished')
    
    imgid_to_anno = {}
    ret_scan = []
    appended_filename = []
    import tqdm
    ann_i_over = 0
    max_plane = 0
    for ann_i in tqdm.tqdm(range(len(json_info_scan["annotations"]))):
        if ann_i < ann_i_over:
            continue
        ann = json_info_scan["annotations"][ann_i]
        image_id = ann['image_id']
        file_name = image_dir_scan + '/' + json_info_scan['images'][image_id-1]['file_name']
        file_name = file_name.replace('/frame/','/unpack/')
        
        segments_info = []
        ann_i_over = ann_i
        while True:
            if ann_i_over >= len(json_info_scan["annotations"]):
                break
            if not image_id == json_info_scan["annotations"][ann_i_over]['image_id']:
                break
            ann_i_over += 1
        for ii in range(ann_i,ann_i_over):
            segments_info.append(json_info_scan["annotations"][ii])

        
        for this_dic in segments_info:
            this_dic['isthing'] = 0

        if len(segments_info) == 0:
            print('zmf: find an image without plane label', file_name)
            continue

        a = int(file_name.split('_')[-2][-4:])
        b = int(file_name.split('_')[-1][:2])
        c = int(file_name.split('_')[-1].split('/')[-1][:-4])


        if 'val' in pairs_txt: pass
        elif a > int(os.environ['DATA_MAX_SCENE']): continue
        # if int(os.getenv("DATASET_CHOICE", "warning")) in [2]:
        #     if a > 27 or b != 0:
        #         if a == 0 and b == 2: pass
        #         else: continue
        #     if not (a == 20 and b == 0 and c == 1261): continue
        #     if not (a == 11 and b == 0 and c == 1454): continue
            
        is_skip = False
        for si in segments_info:
            if si['area'] == 307200:
                print('zmf: skip', file_name)
                is_skip = True
                break
        if is_skip: continue

        imgid_to_anno[image_id] = [file_name, segments_info]
        

    for pair in pairs:
        img_id0, img_id1, a, b = pair
        if 'val' in pairs_txt: pass
        elif a > int(os.environ['DATA_MAX_SCENE']): continue

        if (img_id0 in imgid_to_anno) and (img_id1 in imgid_to_anno):
            filename0 = imgid_to_anno[img_id0][0]
            filename1 = imgid_to_anno[img_id1][0]
            segments_info0 = imgid_to_anno[img_id0][1]
            segments_info1 = imgid_to_anno[img_id1][1]
        else:
            print('img_id %d %d were skipped when loading.'%(img_id0,img_id1))
            continue
        
        if os.environ['SAMPLING_FRAME_NUM'] == '1':
            if filename0 not in appended_filename:
                appended_filename.append(filename0)
                ret_scan.append(
                    {
                        "file_name": [filename0],
                        "image_id": [img_id0],
                        "segments_info": [segments_info0],
                    }
                )
            continue

        if os.environ['SAMPLING_FRAME_NUM'] == '2' and os.environ['PAIR_SAME'] == 'True':
            ret_scan.append(
                {
                    "file_name": [filename0, filename0],
                    "image_id": [img_id0,img_id0],
                    "segments_info": [segments_info0,segments_info0],
                }
            )
            continue

        if os.environ['SAMPLING_FRAME_NUM'] == '2' and os.environ['PAIR_SAME'] != 'True':
            ret_scan.append(
                {
                    "file_name": [filename0, filename1],
                    "image_id": [img_id0,img_id1],
                    "segments_info": [segments_info0,segments_info1],
                }
            )
            continue



    assert len(ret_scan), "No images found!"
    # assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    # assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    

    print('zmf load finished in ytvis.py')
    return ret_scan

def register_ytvis_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in YTVIS's json annotation format for
    instance tracking.

    Args:
        name (str): the name that identifies a dataset, e.g. "ytvis_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_ytvis_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="ytvis", **metadata
    )

def register_scannet(
    # name, metadata, image_root, image_root_scan, panoptic_root, panoptic_json, scan_json, instances_json=None
    name, metadata, image_root_scan, scan_json, scan_pairs
):

    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_scannet_json(scan_json, image_root_scan, scan_pairs),
    )
    MetadataCatalog.get(panoptic_name).set(
        # panoptic_root=panoptic_root,
        # image_root=image_root,
        image_root_scan=image_root_scan,
        # panoptic_json=panoptic_json,
        scan_json=scan_json,
        # json_file=instances_json,
        evaluator_type="scannet_video",
        # ignore_label=255,
        # label_divisor=1000,
        **metadata,
    )


if __name__ == "__main__":
    """
    Test the YTVIS json dataset loader.
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys
    from PIL import Image

    logger = setup_logger(name=__name__)
    #assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get("ytvis_2019_train")

    json_file = "./datasets/ytvis/instances_train_sub.json"
    image_root = "./datasets/ytvis/train/JPEGImages"
    dicts = load_ytvis_json(json_file, image_root, dataset_name="ytvis_2019_train")
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "ytvis-data-vis"
    os.makedirs(dirname, exist_ok=True)

    def extract_frame_dic(dic, frame_idx):
        import copy
        frame_dic = copy.deepcopy(dic)
        annos = frame_dic.get("annotations", None)
        if annos:
            frame_dic["annotations"] = annos[frame_idx]

        return frame_dic

    for d in dicts:
        vid_name = d["file_names"][0].split('/')[-2]
        os.makedirs(os.path.join(dirname, vid_name), exist_ok=True)
        for idx, file_name in enumerate(d["file_names"]):
            img = np.array(Image.open(file_name))
            visualizer = Visualizer(img, metadata=meta)
            vis = visualizer.draw_dataset_dict(extract_frame_dic(d, idx))
            fpath = os.path.join(dirname, vid_name, file_name.split('/')[-1])
            vis.save(fpath)
