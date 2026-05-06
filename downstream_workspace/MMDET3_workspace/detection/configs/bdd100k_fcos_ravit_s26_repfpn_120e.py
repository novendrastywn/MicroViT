_base_ = [
    './_base_/models/fcos_r50_fpn_gn_head.py',
    './_base_/datasets/bdd100k_detection_512x512.py',
    './_base_/schedules/schedule_120e_bdd100k.py',
    './_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        type='ravit_s26',
        style='pytorch',
        ),
    neck=dict(
        _delete_=True,
        type='RepFPN',
        start_level = 1,
        in_channels=[48, 96, 192, 384],
        out_channels=256,
        num_outs=3,),
    bbox_head=dict(
        _delete_=True, 
        type='FCOSHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32,],
        regress_ranges=((-1,128), (128, 256), (256, 512)), #(-1,128), (128, 256), (256, 512)
        center_sampling = True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
)

#optimizer
optim_wrapper = dict(_delete_=True, 
    type='OptimWrapper',
    optimizer = dict(type='AdamW', lr=4e-5, weight_decay=0.001) # weight_decay=0.0001
    )
# optim_wrapper = dict(_delete_=True, 
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
#     )

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=16, norm_type=2))
train_dataloader = dict(batch_size=8, num_workers=2,)
default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=3))  # only keep latest 3 checkpoints
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)