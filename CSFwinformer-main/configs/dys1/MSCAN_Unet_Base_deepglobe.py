_base_ = [
    '../_base_/models/UNet_mscan_base.py', '../_base_/datasets/deepglobe.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
checkpoint_config = dict(by_epoch=False, interval=16000, max_keep_ckpts=2)
evaluation = dict(interval=16000, metric=['mDice','mIoU','mFscore'], pre_eval=True)
model=dict(test_cfg = dict(mode='slide', crop_size=(512, 512), stride=(170, 170)))
data=dict(samples_per_gpu=1, workers_per_gpu=1,)


optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
