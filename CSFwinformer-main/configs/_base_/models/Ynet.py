# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='YNet_general'
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
    test_cfg=dict(crop_size=(512,512),stride=(170,170)))
