# optimizer
optimizer = dict(
    
    type='AdamW',
    lr=3e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,)
# optimizer_config = dict()
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(_delete_=True,
                 policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000, max_keep_ckpts=2)
evaluation = dict(interval=2000, metric=['mDice','mIoU','mFscore'], pre_eval=True)
