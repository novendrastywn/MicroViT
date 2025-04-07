export NCCL_LL_THRESHOLD=0
dataset_path='/home/dsdl/Documents/ImageDataset/ImageNet-1K'
code_path='/home/dsdl/Documents/BackboneWorkspace'

NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NUM_GPU=3
epochs=300
dataset=ImageNet
BATCH_SIZE=682

MODEL=microvit_1  OUTPUT_DIR=./checkpoints/$dataset/$MODEL

cd $code_path && python -m torch.distributed.launch --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$NUM_GPU --master_port=$PORT train_imagenet.py \
--model $MODEL  --model-ema --auto_resume --data=$dataset_path --batch-size $BATCH_SIZE --epochs=$epochs --img-size 224 --input-size 3 224 224 --workers=64 \
--lr 4e-3 --warmup-lr 1e-6 --warmup-epochs 5 --min-lr 1e-5 --drop-path 0. --amp --native-amp --crop-pct 0.85 \
--project=revit --experiment=$MODEL \
--output $OUTPUT_DIR --enable_wandb #--distillation-type hard

