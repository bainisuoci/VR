dataset_type = 'DOTADataset'
data_root = '/mnt/zhangyupeng/SAR_datasets/SSDD/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadOBBAnnotations',
        with_bbox=True,
        with_label=True,
        with_poly_as_mask=True),
    dict(type='LoadDOTASpecialInfo'),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='OBBRandomFlip', h_flip_ratio=0.5, v_flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='RandomOBBRotate',
        rotate_after_flip=True,
        angles=(0, 0),
        vert_rate=0.5,
        vert_cls=['roundabout', 'storage-tank']),
    dict(type='Pad', size_divisor=32),
    dict(type='DOTASpecialIgnore', ignore_size=2),
    dict(type='FliterEmpty'),
    dict(type='Mask2OBB', obb_type='obb'),
    dict(type='OBBDefaultFormatBundle'),
    dict(
        type='OBBCollect',
        keys=['img', 'gt_bboxes', 'gt_obboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipRotateAug',
        img_scale=[(640, 640)],
        h_flip=False,
        v_flip=False,
        rotate=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='OBBRandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='RandomOBBRotate', rotate_after_flip=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='OBBCollect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='DOTADataset',
        task='Task1',
        ann_file='/mnt/zhangyupeng/SAR_datasets/SSDD/trainval/annfiles/',
        img_prefix='/mnt/zhangyupeng/SAR_datasets/SSDD/trainval/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadOBBAnnotations',
                with_bbox=True,
                with_label=True,
                with_poly_as_mask=True),
            dict(type='LoadDOTASpecialInfo'),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
            dict(type='OBBRandomFlip', h_flip_ratio=0.5, v_flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='RandomOBBRotate',
                rotate_after_flip=True,
                angles=(0, 0),
                vert_rate=0.5,
                vert_cls=['roundabout', 'storage-tank']),
            dict(type='Pad', size_divisor=32),
            dict(type='DOTASpecialIgnore', ignore_size=2),
            dict(type='FliterEmpty'),
            dict(type='Mask2OBB', obb_type='obb'),
            dict(type='OBBDefaultFormatBundle'),
            dict(
                type='OBBCollect',
                keys=['img', 'gt_bboxes', 'gt_obboxes', 'gt_labels'])
        ]),
    val=dict(
        type='DOTADataset',
        task='Task1',
        ann_file='/mnt/zhangyupeng/SAR_datasets/SSDD/val/annfiles/',
        img_prefix='/mnt/zhangyupeng/SAR_datasets/SSDD/val/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipRotateAug',
                img_scale=[(640, 640)],
                h_flip=False,
                v_flip=False,
                rotate=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='OBBRandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='RandomOBBRotate', rotate_after_flip=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='OBBCollect', keys=['img'])
                ])
        ]),
    test=dict(
        type='DOTADataset',
        task='Task1',
        ann_file='/mnt/zhangyupeng/SAR_datasets/SSDD/test/annfiles/',
        img_prefix='/mnt/zhangyupeng/SAR_datasets/SSDD/test/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipRotateAug',
                img_scale=[(640, 640)],
                h_flip=False,
                v_flip=False,
                rotate=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='OBBRandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='RandomOBBRotate', rotate_after_flip=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='OBBCollect', keys=['img'])
                ])
        ]))
evaluation = dict(metric='mAP')
optimizer = dict(type='AdamW', lr=2.5e-05, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[55, 88, 121])
total_epochs = 200
checkpoint_config = dict(interval=10)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='ATSSOBB',
    pretrained=
    'https://download.openmmlab.com/pretrain/third_party/resnet50_caffe-788b5fa3.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='ScaleAdaptiveHead',
        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        scale_theta=True,
        reg_loss_wh_thre=5,
        bbox_coder=dict(
            type='OBB2DistCoder',
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='CIoULoss', loss_weight=1.0),
        loss_trig=dict(type='L1Loss', loss_weight=0.2),
        anchor_generator=dict(
            type='Theta0AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            center_offset=0.5,
            strides=[8, 16, 32, 64, 128]),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
train_cfg = dict(
    assigner=dict(type='OBBATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=2000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='obb_nms', iou_thr=0.1),
    max_per_img=2000)
work_dir = '/mnt/zhangyupeng/SAR_target_detection/DFDet-master/workdir/test/'
gpu_ids = range(0, 1)
