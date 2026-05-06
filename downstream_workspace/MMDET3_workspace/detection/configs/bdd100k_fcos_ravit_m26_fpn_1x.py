_base_ = [
    './_base_/models/fcos_r50_fpn_gn_head.py',
    './_base_/datasets/bdd100k_detection.py',
    './_base_/schedules/schedule_1x_bdd100k.py',
    './_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        type='ravit_m26',
        style='pytorch',
        ),
    neck=dict(
        type='FPN',
        in_channels=[ 64, 128, 256, 512],
        out_channels=256,
        num_outs=5,
        ),
    bbox_head=dict(num_classes=10),
)

#optimizer
optim_wrapper = dict(_delete_=True, 
    type='OptimWrapper',
    optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.001) # weight_decay=0.0001
    )
# optim_wrapper = dict(_delete_=True, 
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
#     )

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=16, norm_type=2))
train_dataloader = dict(batch_size=8, num_workers=2,)
default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=3))  # only keep latest 3 checkpoints
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)