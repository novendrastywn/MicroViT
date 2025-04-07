_base_ = [
    './_base_/models/mask-rcnn_r50_fpn.py',
    './_base_/datasets/coco_instance.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        type='ravit_sa2',
        style='pytorch',
        ),
    neck=dict(
        type='FPN',
        in_channels=[ 48, 96, 192, 384],
        out_channels=256,
        num_outs=5,
        ))

#optimizer
optim_wrapper = dict(
    _delete_=True, 
    type='OptimWrapper',
    optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.05) # weight_decay=0.0001
    )
optimizer_config = dict(grad_clip=None)
train_dataloader = dict(batch_size=4, num_workers=4,)
# data = dict(samples_per_gpu=6, workers_per_gpu=2)

