_base_ = [
    './_base_/models/retinanet_r50_fpn.py',
    './_base_/datasets/bdd100k_detection.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
]


#optimizer
optim_wrapper = dict(
    _delete_=True, 
    type='OptimWrapper',
    optimizer = dict(type='AdamW', lr=3e-5, weight_decay=0.05) # weight_decay=0.0001
    )
optimizer_config = dict(grad_clip=None)
train_dataloader = dict(batch_size=12, num_workers=24,)
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=1)