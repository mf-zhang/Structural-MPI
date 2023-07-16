export DETECTRON2_DATASETS='/large/mfzhang/for_SMPI/zip/ScanNet/' # change to new data address
export SAMPLING_FRAME_NUM='2' # number of input views (1 or 2)
export LLFF_BLEND='True'      # enable it when SAMPLING_FRAME_NUM = 2
export SEG_NONPLANE_NUM='32'  # for non-planar area, number of depth gaps (64, 32, 24, 16, ...) like standard MPI
export TRAIN_PHASE='2'        # training phase (1: bootstrap with alpha=mask and no rgba 2: main training)
export EVAL_USE_GPU='True'    # use gpu in eval
export DATA_MAX_GAP='20'      # the frame number gap between 2 source views. (20 for paper evaluation, 40 for more cases)
export DEMO='False'           # False: no demo

export DATASET_CHOICE='2'    
export BLACK_BORDER='True'   # fill the image border with near pixel or black. near-pixel may cause problems in multi-view
export DATA_MAX_SCENE='800'  # 800 means using all data, 20 means only using first 20 scenes of ScanNet for debug
export USE_ORIG_RGB='False'  # use input rgb instead of predicted rgb for each layer for debug
export ERODE_FAC='15'        # erode the plane segmentation borders because plane segmentation labels are not accurate
export SINGLE_MATCH='True'   # in multi-view mode, there is no plane correspondence label, Ture means only relying on auto matching
export PAIR_SAME='False'     # in multi-view mode, Ture means using 2 identical images as input for debug
export ONLY_SEG='False'      # only conduct planar segmentation for debug
export PKL_CKPT='False'      # pytorch load ckpt version problem
export OLD_MPI='False'       # using sparate model to generate standard MPI for non-planar regions, not used in paper
export RUN_ON_SCANNET='True' # run on ScanNet
export SEG_NONPLANE='True'   # generating standard MPI on non-planar regions in a segmentation style
export TEST_WITH_TRAIN_DATA='False' # eval on all training data for debug
export TEST_NVS_ONLY='False' # evaluate only novel view synthesis
export ALL_COVER='False'     # if true, segm confidence threshold = 0, segmentation will cover all image for debug
export TRAIN_MP3D='False'    # may have bug when training on MP3D with multi-gpu

python train_net_video.py --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8ep.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS ../ckpt/model_input2_query100_phase2_step99999_256x384.pth DATALOADER.NUM_WORKERS 8 INPUT.IMAGE_SIZE "[256,384]"

# python train_net_video.py --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8ep.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS ../ckpt/model_input1_query100_phase2_step99999_256x384.pth DATALOADER.NUM_WORKERS 8 INPUT.IMAGE_SIZE "[256,384]" DEMO.GENERATE_VIDEO True