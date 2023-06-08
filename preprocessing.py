import os
import sys
import time 

from utils import utils, paths, helper, hparams

paths.init_dirs('preproc')

ops = [op for op in helper.listdir(paths.dataset_root) if os.path.isdir(op)]

for op in ops:
    start_time = time.time()
    print('Working on video: ', op)
    prefix = helper.prefix(name = 'OP_', id = op.split('OP')[-1])
    target_path = os.path.join(paths.data_root, prefix)
    
    op_files = helper.listdir(op)
    video = [f for f in op_files if f.endswith('.mp4')][0]
    axcs = [f for f in op_files if f.endswith('.txt')][0]
    gopro_videos = helper.listdir(os.path.join(op, 'gopro'))
    
    os.mkdir(target_path)
    utils.vid2audio(video, target_path, samplerate = hparams.samplerate, use_vad = hparams.use_vad)
    utils.GH2audio(gopro_videos, target_path)
    utils.align_audio(target_path, hparams.resolution)
    utils.vid2frame(video, target_path, out_shape = hparams.image_shape) 
    helper.check_align(target_path)
    print('{} is done!\nElapsed time is: {}(hh:mm:ss)'.format(prefix, 
        helper.sec2time(time.time() - start_time))) 

utils.physician_first()

if hparams.data_augmentation:
    utils.data_augmentation(paths.data_root, aug_ratio = hparams.aug_ratio)

paths.init_dirs('features')
utils.xray_features()
utils.audio_features()
utils.check_feats()

print('Pre-processing done!')
