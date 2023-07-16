# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import os
from detectron2.data import DatasetCatalog, MetadataCatalog

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
    register_scannet,
    get_coco_metadata,
)

# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
                         "ytvis_2019/train.json"),
    "ytvis_2019_val": ("ytvis_2019/valid/JPEGImages",
                       "ytvis_2019/valid.json"),
    "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
}


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/train.json"),
    "ytvis_2021_val": ("ytvis_2021/valid/JPEGImages",
                       "ytvis_2021/valid.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
}

dataset_choice = int(os.getenv("DATASET_CHOICE", "warning"))
if dataset_choice == 'warning':
    print('please choose dataset in train_net.py')
    exit()

data_max_gap = os.environ['DATA_MAX_GAP']

if dataset_choice == 1:
    _PREDEFINED_SPLITS_SCANNET = {
        "scannet_train_video": (
            "../ScanNet/scans/scan_plane/scannet_val.json",
            "../ScanNet/scans/scan_plane/2frame_val_%s.txt"%data_max_gap,
        ),
        "scannet_val_video": (
            "../ScanNet/scans/scan_plane/scannet_val.json",
            "../ScanNet/scans/scan_plane/2frame_val_%s.txt"%data_max_gap,
        ),
    }
if dataset_choice == 2:
    _PREDEFINED_SPLITS_SCANNET = {
        "scannet_train_video": (
            "../ScanNet/scans/scan_plane/scannet_train.json",
            "../ScanNet/scans/scan_plane/2frame_%s.txt"%data_max_gap,
        ),
        "scannet_val_video": (
            "../ScanNet/scans/scan_plane/scannet_val.json",
            "../ScanNet/scans/scan_plane/2frame_val_%s.txt"%data_max_gap,
        ),
    }
if dataset_choice == 3:
    _PREDEFINED_SPLITS_SCANNET = {
        "scannet_train_video": (
            "scan_plane/scannet_train.json",
            "scan_plane/2frame_%s.txt"%data_max_gap,
        ),
        "scannet_val_video": (
            "scan_plane/scannet_val.json",
            "scan_plane/2frame_val_%s.txt"%data_max_gap,
        ),
    }

if os.environ['TRAIN_MP3D'] == 'True':
    _PREDEFINED_SPLITS_SCANNET = {
        "scannet_train_video": (
            "mp3d_planercnn_json/cached_set_train.json",
            "mp3d_planercnn_json/cached_set_train.json",
            # "mp3d_planercnn_json/cached_set_test.json",
            # "mp3d_planercnn_json/cached_set_test.json",
        ),
        "scannet_val_video": (
            "mp3d_planercnn_json/cached_set_test.json",
            "mp3d_planercnn_json/cached_set_test.json",
        ),
    }

if os.environ['TEST_WITH_TRAIN_DATA'] == 'True':
    _PREDEFINED_SPLITS_SCANNET = {
        "scannet_train_video": (
            "../ScanNet/scans/scan_plane/scannet_train.json",
            "../ScanNet/scans/scan_plane/2frame_%s.txt"%data_max_gap,
        ),
        "scannet_val_video": (
            "../ScanNet/scans/scan_plane/scannet_train.json",
            "../ScanNet/scans/scan_plane/2frame_%s.txt"%data_max_gap,
        ),
    }


def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_scannet(root):
    for prefix,(scannet_root, pair_root) in _PREDEFINED_SPLITS_SCANNET.items():
        # prefix_instances = 'coco_2017_train' # prefix[: -len("_panoptic")] or val
        # instances_meta = MetadataCatalog.get(prefix_instances)
        # image_root, instances_json = instances_meta.image_root, instances_meta.json_file

        if os.environ['TRAIN_MP3D'] != 'True':
            image_root_scan = '%s/scans/'%root if dataset_choice in [1,2] else os.environ['AMLT_DATA_DIR']


            register_scannet(
                prefix,
                get_coco_metadata(),
                # image_root,
                image_root_scan,
                # os.path.join(root, panoptic_root),
                # os.path.join(root, panoptic_json),
                os.path.join(root, scannet_root),
                os.path.join(root, pair_root),
                # instances_json,
            )
        else:
            image_root_scan = '/data/data_plane/planeformer/' if dataset_choice in [1,2] else os.environ['AMLT_DATA_DIR']

            register_scannet(
                prefix,
                get_coco_metadata(),
                # image_root,
                image_root_scan,
                # os.path.join(root, panoptic_root),
                # os.path.join(root, panoptic_json),
                os.path.join(root, scannet_root),
                os.path.join(root, pair_root),
                # instances_json,
            )

def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
    register_all_scannet(_root)
