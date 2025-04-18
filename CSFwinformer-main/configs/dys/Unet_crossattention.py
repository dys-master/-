_base_ = [
    '../_base_/models/UNet_crossattention.py', '../_base_/datasets/msroad_del.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

evaluation = dict(interval=16000,metric=['mDice','mIoU','mFscore'])
