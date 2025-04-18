_base_ = ['/home/louis/dys/u-mixformer-main/configs/umixformer/A_myhead_umixformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py']

# model settings
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
    
    decode_head=dict(
        type='MYDWCAPFormerHeadCity',
        in_channels=[64, 128, 320, 512]
        )
)