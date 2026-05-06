# ./dist_train.sh ./configs/retinanet_microvitv2_s3_fpn_1x_coco.py 4 --resume /home/dsdl-4090/Documents/dsdl_ssd/MMDET_workspace/MMDET3_workspace/detection/work_dirs/retinanet_microvitv2_s3_fpn_1x_coco/epoch_8.pth

CONFIG=./configs/mask_rcnn_parformer_s1_fpn_1x_coco.py
GPUS=4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# torch.distributed.launch

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch \
    # --resume /home/dsdl-4090/Documents/dsdl_ssd/MMDET_workspace/MMDET3_workspace/detection/work_dirs/retinanet_microvitv2_s3_fpn_1x_coco/epoch_8.pth 