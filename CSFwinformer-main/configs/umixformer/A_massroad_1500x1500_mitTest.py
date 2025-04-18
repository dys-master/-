_base_ = ['/home/louis/dys/u-mixformer-main/configs/umixformer/A_massroad_1500x1500.py']


checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa
train_dataloader = dict(batch_size=1, num_workers=4)
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 6, 12, 3]),#自定义层数
    decode_head=dict(
        in_channels=[64, 128, 320, 512]
        )
)
