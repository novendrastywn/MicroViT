dataset_type = 'BGVPDataset'
data_root = '/home/ndr/Container/ImageDataset/bgvp/'

# CLASSES=("pedestrians", "children_wo_disability", "elderly_wo_disability", "non_vulnerable", "with_disability")

# PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228)]

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/segmentation/cityscapes/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/segmentation/',
#          'data/': 's3://openmmlab/datasets/segmentation/'
#      }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(2048, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='train/train_annotations.coco.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/val_annotations.coco.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val/val_annotations.coco.json',
    metric='bbox',
    backend_args=backend_args)

test_evaluator = val_evaluator

#=============================================================================================================

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type='RepeatDataset',
#         times=8,
#         dataset=dict(
#             type=dataset_type,
#             classes=CLASSES,
#             ann_file=data_root + 'train/train_annotations.coco.json',
#             img_prefix=data_root +'train/' ,
#             pipeline=train_pipeline)),
#     val=dict(
#         type=dataset_type,
#         classes=CLASSES,
#         ann_file=data_root + 'val/val_annotations.coco.json',
#         img_prefix=data_root + 'val/' ,
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         classes=CLASSES,
#         ann_file=data_root +'test/test_annotations.coco.json',
#         img_prefix=data_root + 'test/' ,
#         pipeline=test_pipeline))
# evaluation = dict(interval=1, metric='bbox')
#evaluation = dict(interval=1, classwise=True, metric='bbox', save_best='auto')