_base_ = [
    '../_base_/models/zjd_unet_2plus.py', '../_base_/datasets/deepglobe_zz.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_320k.py'
]
#model=dict(test_cfg=dict(mode='whole'))
#model=dict(test_cfg=dict(mode='slide', crop_size=(448,448), stride=(144,144)),
model=dict(test_cfg=dict(mode='slide', crop_size=(512,512), stride=(170,170)),      
        decode_head=dict(
         loss_decode=
        [
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0,class_weight=[0.1, 1]),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0,class_weight=[0.1, 1]),
             ]),
)




optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
#optimizer = dict(type='SGD', lr=0.00012, momentum=0.9, weight_decay=0.0001)
runner = dict(type='IterBasedRunner', max_iters=320000)
evaluation = dict(interval=10000,metric=['mDice','mIoU','mFscore'])
checkpoint_config = dict(by_epoch=False, interval=10000)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    
    val=dict(

    
        img_dir=['images/test'],
        ann_dir=['annotations/test'],
        #img_dir=['images/my/test/difficult','images/my/test/zhong','images/my/test/easy'],
        #ann_dir=['annotations/my/test/difficult','annotations/my/test/zhong','annotations/my/test/easy'],
    ),
   )
