# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.utils.file_io import PathManager

_root = os.getenv("DETECTRON2_DATASETS", "datasets")

dataset_choice = 2

_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "scannet_train_panoptic": (
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        "coco/panoptic_stuff_train2017",
        # "../scannet/scans/scan_plane/scannet_val.json",
        "../scannet/scans/scan_plane/scannet_train.json",
    ),
    "scannet_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_stuff_train2017",
        "../scannet/scans/scan_plane/scannet_val.json",
    ),
}



def get_metadata():
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


def load_coco_panoptic_json(json_file, json_file_scan, image_dir, image_dir_scan, gt_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info


    with PathManager.open(json_file_scan) as f_scan:
        json_info_scan = json.load(f_scan)
        print('scan json load finished')
    
    ret_scan = []
    import tqdm
    ann_i_over = 0
    max_plane = 0
    for ann_i in tqdm.tqdm(range(len(json_info_scan["annotations"]))):
        if ann_i < ann_i_over:
            continue
        ann = json_info_scan["annotations"][ann_i]
        image_id = ann['image_id']
        # print(image_id, json_info_scan['images'][image_id-1])
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

        if dataset_choice in [1,2]:
            if a > 25 or b != 0: continue
            # if not (a == 20 and b == 0 and c == 1261): continue
            # if not (a == 11 and b == 0 and c == 1454): continue

        is_skip = False
        for si in segments_info:
            if si['area'] == 307200:
                print('zmf: skip', file_name)
                is_skip = True
                break
        if is_skip: continue
        
        ret_scan.append(
            {
                "file_name": file_name,
                "image_id": image_id,
                # "pan_seg_file_name": label_file,
                "segments_info": segments_info,
            }
        )

    return ret_scan


def register_coco_panoptic_annos_sem_seg(
    name, metadata, image_root, image_root_scan, panoptic_root, panoptic_json, scan_json, instances_json=None
):

    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_coco_panoptic_json(panoptic_json, scan_json, image_root, image_root_scan, panoptic_root, metadata),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        image_root_scan=image_root_scan,
        panoptic_json=panoptic_json,
        scan_json=scan_json,
        json_file=instances_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


def register_all_coco_panoptic_annos_sem_seg(root):
    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root, scannet_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = 'coco_2017_train' # prefix[: -len("_panoptic")] or val
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        image_root_scan = '/data/zmf/scannet/scans/' if dataset_choice in [1,2] else os.environ['AMLT_DATA_DIR'] 

        register_coco_panoptic_annos_sem_seg(
            prefix,
            get_metadata(),
            image_root,
            image_root_scan,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, scannet_root),
            instances_json,
        )



register_all_coco_panoptic_annos_sem_seg(_root)
