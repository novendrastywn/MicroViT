_base_ = [
    './_base_/models/centernet_r50_fpn.py',
    './_base_/datasets/bdd100k_detection.py',
    './_base_/schedules/schedule_60e.py',
    './_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        type='ravit_s26',
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
    optimizer = dict(type='AdamW', lr=3e-5, weight_decay=0.05) # weight_decay=0.0001
    )
optimizer_config = dict(grad_clip=None)
train_dataloader = dict(batch_size=12, num_workers=2,)
# data = dict(samples_per_gpu=6, workers_per_gpu=2)