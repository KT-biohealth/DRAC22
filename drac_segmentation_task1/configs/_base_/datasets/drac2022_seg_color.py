# dataset settings
dataset_type = 'DRAC2022SegDataset'
data_root = 'data/DRAC2022_Seg_Color'
img_norm_cfg = dict(
    mean=[98.1, 76.9, 21.2], std=[69.1, 85.0, 28.4])
#img_scale = (1024, 1024)
img_scale = (256, 256)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    #dict(type='Resize', img_scale=img_scale),
    dict(type='CLAHE', clip_limit=2.0, tile_grid_size=(8, 8)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', degree=30, prob=0.5, seg_pad_val=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='CLAHE', clip_limit=2.0, tile_grid_size=(8, 8)),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images/train',
            ann_dir='masks/train',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='masks/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='masks/test',
        pipeline=test_pipeline))
