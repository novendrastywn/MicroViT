_base_ = [
    './_base_/models/yolof_r50_dc.py',
    './_base_/datasets/bdd100k_detection.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(num_classes=10),
)

#optimizer
optim_wrapper = dict(_delete_=True, 
    type='OptimWrapper',
    optimizer = dict(type='AdamW', lr=5e-5, weight_decay=0.05) # weight_decay=0.0001
    )
# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
train_dataloader = dict(batch_size=8, num_workers=2,)
default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=3))  # only keep latest 3 checkpoints
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)