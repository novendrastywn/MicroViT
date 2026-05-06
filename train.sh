DATA_PATH='/home/dsdl-4090/Documents/dsdl_ssd/ImageDataset/ImageNet-1K'
CODE_PATH='/home/dsdl/Documents/ndr_workspace/MicroViTv2' # modify code path here


ALL_BATCH_SIZE=2048
NUM_GPU=4
NUM_WORKERS=16
# GRAD_ACCUM_STEPS=4 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU


# NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python -m torch.distributed.launch 
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nproc_per_node=$NUM_GPU --master_port 12345 main.py --data-path $DATA_PATH --batch-size $BATCH_SIZE \
--model microvitv2_2_mdta --lr 1e-3 --dist-eval --num_workers $NUM_WORKERS --input-size 224 --output_dir ./output/microvitv2_2_mdta \
--weight-decay 0.032 --aa rand-m9-mstd0.5-inc1 --enable_wandb --project=microvitv2
