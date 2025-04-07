# training schedule for 20e
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=60, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.00025, 
        by_epoch=False, begin=0, end=4000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=60,
        by_epoch=True,
        milestones=[40, 55],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
