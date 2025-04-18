# model settings
norm_cfg = dict(type='BN', requires_grad=True)
checkpoint_file = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth"
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MiTB0',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        ),
    decode_head=dict(
        type='identityHead',
        norm_cfg=norm_cfg,
        in_channels=64,
        in_index=4,
        channels=16,
        num_classes=2,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg = dict(mode='slide', crop_size=(448, 448), stride=(170, 170)))
