import torchxrayvision as xrv
import moviepy.editor as mp
import soundfile as sf
import numpy as np
import torch as t
import torchaudio
import librosa
import random
import pydub
import cv2 
import os

from utils import helper, paths, hparams
from scipy import signal, stats
from skimage.io import imread
from PIL import Image



def vid2audio(video_path, target_path, samplerate = 16000, use_vad = False):
    '''
    Converts input video with stero audio (mp4) to two wav files
    and applies voice activity detection.
        
    video_path      : string
                        Path to source video.
                    
    target_path     : string
                        Path to save outputs

    samplerate      : int
                        Sampling rate for writing wav. Default 16kHz
    
    use_vad         : bool
                        Use of VAD
    '''
    print('\tVideo to audio...')

    nbytes = 2
    buffersize = 2000

    video = mp.VideoFileClip(video_path)
    sound = video.audio   
    sound.write_audiofile("temp.wav", samplerate, nbytes, buffersize, 
        "pcm_s32le", logger = None)

    data, samplerate = sf.read('temp.wav')

    sf.write(target_path  + '/Channel_1.wav', data[:,0], samplerate)
    sf.write(target_path + '/Channel_2.wav', data[:,1], samplerate)

    os.remove('temp.wav')

    ## First Channel
    # Normalize
    wav = pydub.AudioSegment.from_file(target_path + '/Channel_1.wav', "wav")  
    wav = pydub.effects.normalize(wav)
    wav = wav.get_array_of_samples()
    wav = np.array(wav)

    # Remove white channel-noise (when mic is off)
    _, _, Zxx = signal.stft(wav, samplerate, nperseg = 1024, noverlap = 0)
    S = np.abs(Zxx)

    S_sum = np.zeros((S.shape[1],1))
     
    for i in range(S.shape[1]):
        S_sum[i] = np.sum(S[:,i])
         
    mic_noise = np.where(S_sum > 1E5)[0]

    for win in mic_noise:
        win_start = (win - 1) * 1024
        win_end = win_start + 2048 
        wav[win_start:win_end] = 0

        
    # VAD
    if use_vad:
        sf.write(target_path  + '/Channel_1.wav', wav, samplerate)
        wav, samplerate = sf.read(target_path  + '/Channel_1.wav')
        wav = helper.VAD_Win(wav)

    # Save
    sf.write(target_path  + '/Channel_1.wav', wav, samplerate)
    
    ## Second Channel 
    # Normalize
    wav = pydub.AudioSegment.from_file(target_path + '/Channel_2.wav', "wav")  
    wav = pydub.effects.normalize(wav)
    wav = wav.get_array_of_samples()
    wav = np.array(wav)

    # Remove white channel-noise (when mic is off)
    _, _, Zxx = signal.stft(wav, samplerate, nperseg = 1024, noverlap = 0)
    S = np.abs(Zxx)

    S_sum = np.zeros((S.shape[1],1))

    for i in range(S.shape[1]):
        S_sum[i] = np.sum(S[:,i])

    mic_noise = np.where(S_sum > 1E5)[0]

    for win in mic_noise:
        win_start = (win - 1) * 1024
        win_end = win_start + 2048 
        wav[win_start:win_end] = 0

    # VAD
    if use_vad:
        sf.write(target_path  + '/Channel_2.wav', wav, samplerate)
        wav, samplerate = sf.read(target_path  + '/Channel_2.wav')
        wav = helper.VAD_Win(wav)
    
    # Save
    sf.write(target_path  + '/Channel_2.wav', wav, samplerate)



def GH2audio(video_path, target_path):
    '''
    Saves audio from GoPro videos as the third channel, 
    with the same sample rate as previous channels
        
    video_path      : string
                        Path to source videos.
                    
    target_path     : string
                        Path to save outputs
    '''
    print('\tGoPro to audio...')
    _, samplerate = sf.read(os.path.join(target_path, 'Channel_1.wav'))

    single_wav = np.zeros(1)

    nbytes = 2
    buffersize = 2000

    for v in video_path:
        video = mp.VideoFileClip(v)
        sound = video.audio   
        sound.write_audiofile("temp.wav", samplerate, nbytes, buffersize, 
            "pcm_s32le", logger = None)

        data, _ = sf.read('temp.wav')
        single_wav = np.concatenate((single_wav, data[:, 0]))  #both channels are same in gopro, picking first

        os.remove('temp.wav')

    sf.write(target_path + '/Channel_3.wav', single_wav[1:], samplerate)

    wav = pydub.AudioSegment.from_file(target_path + '/Channel_3.wav', "wav")  
    wav = pydub.effects.normalize(wav)
    wav = wav.get_array_of_samples()
    wav = np.array(wav)

    sf.write(target_path  + '/Channel_3.wav', wav, samplerate)



def align_audio(audio_path, resolution):
    '''
    Aligns three audio channels.

    audio_path      : string
                        Path to source wav files.

    resolution      : int
                        Time resolution
    '''
    print('\tAligning audio...')
    
    ch1_wav, samplerate = sf.read(os.path.join(audio_path, 'Channel_1.wav'))
    ch2_wav, _ = sf.read(os.path.join(audio_path, 'Channel_2.wav'))
    gh_wav, _ = sf.read(os.path.join(audio_path, 'Channel_3.wav'))

    sampleunit = resolution * samplerate

    # find lag (probably in first 30 mins)
    ref_wav = ch1_wav[:samplerate*60*30]
    target_wav = gh_wav[:samplerate*60*30]

    corr = signal.correlate(ref_wav, target_wav)
    lag = np.argmax(corr)

    # cross-check with other channel
    ref_wav = ch2_wav[:samplerate*60*30] 
    corr_control = signal.correlate(ref_wav, target_wav)
    lag_control = np.argmax(corr_control)

    if lag != lag_control:
        if corr_control[lag_control] > corr[lag]:
            lag = lag_control


    # shift lagging channels with zero-padding
    # to ensure perfect alignment with 1 fps frames, padding is done with samplerate scale
    if lag < len(target_wav):
        pad_width = len(target_wav) - lag
        pad_width = (pad_width // samplerate) * samplerate
        ch1_wav = np.pad(ch1_wav, (pad_width, 0), constant_values = 0)
        ch2_wav = np.pad(ch2_wav, (pad_width, 0), constant_values = 0)
        pad_diff = len(target_wav) - lag - pad_width
        gh_wav = gh_wav[pad_diff:]
        np.save('pre_pad.npy', pad_width // samplerate)

    if lag > len(target_wav):
        pad_width = lag - len(target_wav) 
        gh_wav = np.pad(gh_wav, (pad_width, 0), constant_values = 0)

    # align also tails with zero-padding
    # to ensure perfect alignment, using resolution scale
    if len(ch1_wav) > len(gh_wav):
        # might loose some data with np floor here
        target_length = int(np.floor(len(gh_wav) / sampleunit) * sampleunit)
        ch1_wav = ch1_wav[:target_length]
        ch2_wav = ch2_wav[:target_length]
        pad_width = target_length - len(gh_wav)
        if pad_width <= 0:
            gh_wav = gh_wav[:target_length]
        else:
            gh_wav = np.pad(gh_wav, (0, pad_width), constant_values = 0)

    if len(gh_wav) > len(ch1_wav):
        target_length = int(np.floor(len(gh_wav) / sampleunit) * sampleunit)
        gh_wav = gh_wav[:target_length]
        pad_width = target_length - len(ch1_wav)
        if pad_width <= 0:
            ch1_wav = ch1_wav[:target_length]
            ch2_wav = ch2_wav[:target_length]
        else:
            ch1_wav = np.pad(ch1_wav, (0, pad_width), constant_values = 0)
            ch2_wav = np.pad(ch2_wav, (0, pad_width), constant_values = 0)
    
    np.save('target_frame.npy', target_length // samplerate)

    sf.write(audio_path + '/' + 'Channel_3.wav', gh_wav, samplerate)
    sf.write(audio_path + '/' + 'Channel_1.wav', ch1_wav, samplerate)
    sf.write(audio_path + '/' + 'Channel_2.wav', ch2_wav, samplerate)



def vid2frame(video_path, target_path, out_fps = 1, out_shape = (256,256)):
    '''
    Converts input video (mp4) to frames.
        
    video_path      : string
                        Path to source video.
                    
    target_path     : string
                        Path to save outputs

    out_fps         : int
                        Output fps

    out_shape       : tuple
                        Output shape of frames, (Height , Width)
    '''
    print('\tVideo to frame...')  

    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS) 
    assert fps > out_fps, "Output FPS can not be larger than original FPS!"
    
    fps_scale = fps / out_fps
    assert fps_scale.is_integer(), "Ratio of original FPS and output FPS is " \
        "{:.2f}, it must be an integer!".format(fps_scale)

    idx = 1 # frame counter

    # add alignment frames at the beginning
    if os.path.exists('pre_pad.npy'):
        pre_pad = np.load('pre_pad.npy')

        for _ in range(pre_pad):
            im = np.zeros((out_shape[0], out_shape[1], 3))
            frame_prefix = helper.prefix(idx, name = 'frame_', buffer = 5)
            image_name = target_path + '/' + frame_prefix + '.jpg'
            cv2.imwrite(image_name, im)
            idx += 1
    
        os.remove('pre_pad.npy')

    # get target frame count 
    target_frame = np.load('target_frame.npy')

    # video to frame
    success, image = video.read()
    count = 0
    while success:
        
        if count % fps_scale == 0 and idx <= target_frame:
        
            frame_prefix = helper.prefix(idx, name = 'frame_', buffer = 5)
            image_name = target_path + '/' + frame_prefix + '.jpg'
            
            image = image[:, :1024] # crop patient info side
            
            ## find bounding boxes of x-rays/windows
            im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(im_gray, 25, 255, 0)
            contours, _ = cv2.findContours(thresh, 1, 2)
                    
            cnt_areas = np.zeros((len(contours),1))

            for i, c in enumerate(contours):
                cnt_areas[i] = cv2.contourArea(c)
            
            cnt_norm = np.abs(stats.zscore(cnt_areas))
            cnt_idx = np.where(cnt_norm > 3)[0]
            
            if len(cnt_idx) > 1: # merge, if there is multiple large contours 
                cnt = [contours[c] for c in cnt_idx]
                boundaries = np.zeros((len(cnt),4))

                for i, c in enumerate(cnt):
                    x,y,w,h = cv2.boundingRect(c)
                    boundaries[i] = [x, y, x + w, y + h]
                    
                x = int(np.min(boundaries[:, 0]))
                y = int(np.min(boundaries[:, 1]))
                w = int(np.max(boundaries[:, 2])) 
                h = int(np.max(boundaries[:, 3]))

            else: # a single large contour exists
                cnt = contours[np.argmax(cnt_areas)]
                x,y,w,h = cv2.boundingRect(cnt)
                w += x
                h += y 

            ## crop box and zero-pad to square (optionally, images could be square-cropped)
            image = image[y:h, x:w, :]

            ## resize and save
            image = cv2.resize(image, dsize = out_shape, interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(image_name, image)
            idx += 1
        
        success, image = video.read()        
        count += 1

    # add alignment frames at the end
    if target_frame - idx > 0:
        for _ in range(target_frame - idx + 1):
            im = np.zeros((out_shape[0], out_shape[1], 3))
            frame_prefix = helper.prefix(idx, name = 'frame_', buffer = 5)
            image_name = target_path + '/' + frame_prefix + '.jpg'
            cv2.imwrite(image_name, im)
            idx += 1

    os.remove('target_frame.npy')

    # remove non-xray windows
    helper.remove_windows(target_path)



def data_augmentation(data_path, aug_ops = None, aug_ratio = 0.25):
    '''
    Augments audio and images of an OP
        
    data_path       : string
                        Path to source operation.

    aug_ops         : list of operations
                        Choosen operation to augment. If it is not given, 
                        operations will be choosen randomly from good_ops.

    aug_ratio       : float
                        A number between [0,1], represents portion of operations
                        randomly choosen from good_ops.
    '''

    print('\tData augmentation...')

    if aug_ops is None:
        # select from good ops, 
        # ops are same as helper.get_datasets_partitions
        good_ops = ['OP_001', 'OP_002', 'OP_004', 'OP_006', 'OP_008', 'OP_010', 'OP_011', 
            'OP_012', 'OP_013', 'OP_014', 'OP_016', 'OP_019', 'OP_023', 'OP_025', 'OP_027',
            'OP_029', 'OP_030', 'OP_031', 'OP_032', 'OP_033', 'OP_034', 'OP_035', 'OP_036', 
            'OP_038', 'OP_039']

        aug_ops = random.sample(good_ops, k = int(len(good_ops) * aug_ratio))

    ## Audio
    # iter
    for op in sorted(aug_ops):
        target_path = os.path.join(data_path, op + 'A')
        os.mkdir(target_path)
        print(target_path)
        # Channel 1
        ch1_wav, samplerate = sf.read(os.path.join(data_path, op, 'Channel_1.wav'))
        
        aug_wav = np.zeros_like(ch1_wav)


        # Cutting windows parts and augment
        win_len = 20 * samplerate
        start_idx = 0
        end_idx = win_len

        # windowing func
        hann_win = signal.windows.hann(win_len)
        
        while end_idx < len(ch1_wav):
            if np.sum(ch1_wav[start_idx:end_idx]) != 0:
                wav_chunk = ch1_wav[start_idx:end_idx]
                wav_chunk = wav_chunk * hann_win
                aug_wav[start_idx:end_idx] = helper.audio_augment(wav_chunk, 
                    samplerate)
            
            start_idx = end_idx
            end_idx = min(end_idx + win_len, len(ch1_wav))

        sf.write(os.path.join(target_path, 'Channel_1.wav'), aug_wav, samplerate)

        # Channel 2
        ch2_wav, _ = sf.read(os.path.join(data_path, op, 'Channel_2.wav'))

        aug_wav = np.zeros_like(ch2_wav)

        # Cutting windows parts and augment
        start_idx = 0
        end_idx = win_len
        
        while end_idx < len(ch2_wav):
            if np.sum(ch2_wav[start_idx:end_idx]) != 0:
                wav_chunk = ch2_wav[start_idx:end_idx]
                wav_chunk = wav_chunk * hann_win
                aug_wav[start_idx:end_idx] = helper.audio_augment(wav_chunk, 
                    samplerate)
            
            start_idx = end_idx
            end_idx = min(end_idx + win_len, len(ch2_wav))
                
        sf.write(os.path.join(target_path, 'Channel_2.wav'), aug_wav, samplerate)

        # Channel 3
        ch3_wav, _ = sf.read(os.path.join(data_path, op, 'Channel_3.wav'))

        aug_wav = np.zeros_like(ch3_wav)

        # Cutting windows parts and augment
        start_idx = 0
        end_idx = win_len
        
        while end_idx < len(ch3_wav):
            if np.sum(ch3_wav[start_idx:end_idx]) != 0:
                wav_chunk = ch3_wav[start_idx:end_idx]
                wav_chunk = wav_chunk * hann_win
                aug_wav[start_idx:end_idx] = helper.audio_augment(wav_chunk, 
                    samplerate)
            
            start_idx = end_idx
            end_idx = min(end_idx + win_len, len(ch3_wav))
                
        sf.write(os.path.join(target_path, 'Channel_3.wav'), aug_wav, samplerate)

        ## X-Ray
        frames = [f for f in os.listdir(os.path.join(data_path, op)) if f.endswith('.jpg')]
        for f in frames:
            im = Image.open(os.path.join(data_path, op, f))
            aug_im = helper.image_augment(im)
            aug_im.save(os.path.join(target_path, f))



def physician_first():
    ''' Read annotation.json and changes order of channels
    to ensure that first channel is the physician mic.
    '''
    input('It will change order of wav files. Press a key to continue')
    
    for op in helper.listdir(paths.data_root):
        print(op)
        op_name = op.split('/')[-1]
        ch1, ch2 = helper.whose_mic(op_name)
        if ch1 != 'physician':
            ch1 = paths.data_root + '/' + op_name + '/Channel_1.wav'
            ch2 = paths.data_root + '/' + op_name + '/Channel_2.wav'
            temp = paths.data_root + '/' + op_name + '/temp.wav'
            os.rename(ch1, temp)
            os.rename(ch2, ch1)
            os.rename(temp, ch2)



def audio_features():
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
    wav2vec = bundle.get_model().to(device)

    # Iter
    for op in sorted(os.listdir(paths.data_root)):
        print('Extracting Audio Features...', op)
        # target path
        target_path = os.path.join(paths.feat_root, op)
        ch_files = [f for f in os.listdir(target_path) if f.startswith('ch')]
        if len(ch_files) == 0:

            # read wavs
            ch1_wav, samplerate = sf.read(os.path.join(paths.data_root, 
                op, 'Channel_1.wav'))
            ch2_wav, _ = sf.read(os.path.join(paths.data_root, 
                op, 'Channel_2.wav'))
            gh_wav, _ = sf.read(os.path.join(paths.data_root, 
                op, 'Channel_3.wav'))

            # pre-emphasis filter
            ch1_wav = librosa.effects.preemphasis(ch1_wav, coef = hparams.coef)
            ch2_wav = librosa.effects.preemphasis(ch2_wav, coef = hparams.coef)
            gh_wav = librosa.effects.preemphasis(gh_wav, coef = hparams.coef)

            # duration in seconds
            dur = len(ch1_wav) // samplerate
            win = hparams.resolution * samplerate
            hann_win = signal.windows.hann(win)

            # pad
            if hparams.casual:
                pad_width = (hparams.resolution - 1 ) * samplerate
                ch1_wav = np.pad(ch1_wav, (pad_width, 0), 
                    'constant', constant_values = 0)
                ch2_wav = np.pad(ch2_wav, (pad_width, 0), 
                    'constant', constant_values = 0)
                gh_wav = np.pad(gh_wav, (pad_width, 0), 
                    'constant', constant_values = 0)
            else:
                pad_width = (hparams.resolution - 1 ) / 2 * samplerate
                ch1_wav = np.pad(ch1_wav, (pad_width, pad_width), 
                    'constant', constant_values = 0)
                ch2_wav = np.pad(ch2_wav, (pad_width, pad_width), 
                    'constant', constant_values = 0)
                gh_wav = np.pad(gh_wav, (pad_width, pad_width), 
                    'constant', constant_values = 0)

            # init idx
            start_idx = 0

            # iter
            for idx in range(dur):
                end_idx = start_idx + win
                
                ch1_chunk = np.copy(ch1_wav[start_idx:end_idx])
                ch1_chunk = ch1_chunk * hann_win
                ch1_chunk_= t.from_numpy(ch1_chunk).float().to(device)
                with t.inference_mode():
                    ch1_feats, _ = wav2vec.extract_features(ch1_chunk_.unsqueeze(0))
                
                ch2_chunk = np.copy(ch2_wav[start_idx:end_idx])
                ch2_chunk = ch2_chunk * hann_win
                ch2_chunk_= t.from_numpy(ch2_chunk).float().to(device)
                with t.inference_mode():
                    ch2_feats, _ = wav2vec.extract_features(ch2_chunk_.unsqueeze(0))

                gh_chunk = np.copy(gh_wav[start_idx:end_idx])
                gh_chunk = gh_chunk * hann_win
                gh_chunk_= t.from_numpy(gh_chunk).float().to(device)
                with t.inference_mode():
                    gh_feats, _ = wav2vec.extract_features(gh_chunk_.unsqueeze(0))

                # save 
                t.save(ch1_feats[-1], target_path + '/' + 
                    helper.prefix(idx + 1, 'ch1_', buffer = 5))
                t.save(ch2_feats[-1], target_path + '/' + 
                    helper.prefix(idx + 1, 'ch2_', buffer = 5))
                t.save(gh_feats[-1], target_path + '/' + 
                    helper.prefix(idx + 1, 'gh_', buffer = 5))

                start_idx += samplerate
        else:
            print('{} number of features already exists'.format(len(ch_files)))



def xray_features():
    # Model
    xrayvision = xrv.models.DenseNet(weights="densenet121-res224-all")

    # Iter
    for op in sorted(os.listdir(paths.data_root)):
        print('\tExtracting X-Ray Features...', op)
        # target path
        target_path = os.path.join(paths.feat_root, op)
        if not os.path.exists(target_path):
            os.mkdir(target_path)

            # read frames
            frames = helper.listdir(os.path.join(paths.data_root, op))
            frames = [ f for f in frames if f.endswith('.jpg')]

            # iter
            for idx, f in enumerate(frames): 
                feat_prefix = helper.prefix(idx + 1, 'frame_', buffer = 5)
                feat_name = target_path + '/' + feat_prefix

                im = imread(f)
                im = xrv.datasets.normalize(im, 255)
                im = im.mean(2)[None, ...]
                im = t.from_numpy(im)
                out = xrayvision.features2(im[None,...])

                t.save(out, feat_name)
        else:
            print('Folder already exists...')



def check_feats():
    for op in sorted(os.listdir(paths.feat_root)):
        ch1_count, ch2_count, frame_count, gh_count = 0, 0, 0, 0        
        for f in sorted(os.listdir(paths.feat_root + '/' + op)):
            if f.startswith('ch1'):
                ch1_count += 1
            if f.startswith('ch2'):
                ch2_count += 1
            if f.startswith('frame'):
                frame_count += 1
            if f.startswith('gh'):
                gh_count += 1
        print(op, ch1_count, ch2_count, frame_count, gh_count)
        if not (ch1_count == ch2_count == frame_count == gh_count):
            print('Count mismatch!!')