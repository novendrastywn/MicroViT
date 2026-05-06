_base_ = [
    './_base_/models/retinanet_r50_fpn.py',
    './_base_/datasets/coco_detection.py',
    './_base_/schedules/schedule_1x.py', 
    './_base_/default_runtime.py'
]
pretrained = None
model = dict(
    # pretrained=None,
    backbone=dict(        
        type='microvitv2_3',
        style='pytorch',
        # pretrained=pretrained,
        # frozen_stages=-1,
        ),
    neck=dict(
        type='EfficientFPN',
        in_channels=[ 192, 384, 448],
        out_channels=256,
        start_level=0,
        num_outs=5,
        num_extra_trans_convs=1,
        ))

# optimizer
optim_wrapper = dict(
    _delete_=True, 
    type='OptimWrapper',
    optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.025) # weight_decay=0.0001
    )
optimizer_config = dict(grad_clip=None)
train_dataloader = dict(batch_size=4, num_workers=4,)
default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=3))  # only keep latest 3 checkpoints
