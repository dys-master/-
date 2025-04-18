_base_ = [
    '../_base_/models/UNetDynamicsnake_conv.py', '../_base_/datasets/msroad1.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

evaluation = dict(interval=2000, metric=['mDice','mIoU','mFscore'], pre_eval=True)
model=dict(test_cfg = dict(mode='slide', crop_size=(448, 448), stride=(170, 170)))
data=dict(samples_per_gpu=2, workers_per_gpu=1,)
