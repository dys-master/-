_base_ = [
    '../_base_/models/UNet_mscan.py', '../_base_/datasets/msroad_del.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

evaluation = dict(interval=8000,metric=['mDice','mIoU','mFscore'])
model=dict(test_cfg = dict(mode='slide', crop_size=(512, 512), stride=(170, 170)))
data=dict(samples_per_gpu=4, workers_per_gpu=1,)