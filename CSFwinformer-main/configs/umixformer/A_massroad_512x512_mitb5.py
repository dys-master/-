_base_ = ['/home/louis/dys/u-mixformer-main/configs/umixformer/A_massroad_512x512_mitb0.py']

# model settings
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64, num_heads=[1, 2, 5, 8], 
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        decoder_params=dict(embed_dim=768,
                            num_heads=[8, 5, 2, 1],
                            pool_ratio=[1, 2, 4, 8]),
        )
)
train_dataloader = dict(batch_size=8, num_workers=4)