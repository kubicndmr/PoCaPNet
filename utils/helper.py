import matplotlib.pyplot as plt
import audiomentations as au
import torchvision as tv
import soundfile as sf
import numpy as np

import matplotlib
import random
import torch
import json
import time
import cv2
import os

from utils import paths, hparams
from scipy import signal


def prefix(id, name = '', buffer = 3):
    '''
    Creates prefix padded with zeros, e.g., name001
    
    id              : int
                        Number to be padded

    name            : string
                        Prefix

    buffer          : int
                        Length of numbers including zeros
    '''
    return name + str(id).zfill(buffer)



def sec2time(sec):
    '''
    Converts seconds to hh:mm:ss

    sec             : int
                        Seconds
    '''
    h = int(sec // 3600)
    m = int(sec % 3600 // 60)
    s = int(sec % 3600 % 60)
    return "{:02d}:{:02d}:{:02d}".format(h, m, s)



def time2sec(time_str):
    '''
    Converts hh:mm:ss to seconds

    time_str        : str
                        Time
    '''

    time_str = time_str.split(':')

    if len(time_str) == 3:
        sec = int(time_str[0])*3600 + \
                int(time_str[1])*60 + int(time_str[2])
    elif len(time_str) == 2:
        sec = int(time_str[0])*60 + int(time_str[1])

    return sec



def print_log(text, file_name = 'Log.txt', ends_with = '\n', display = True):
    '''
    Prints output to the log file.
    
    text        : string or List               
                        Output text

    file_name   : string
                        Target log file

    ends_with   : string
                        Ending condition for print func.

    display     : Bool
                        Wheter print to screen or not.
    '''
    
    if display:
        print(text, end = ends_with)

    with open(file_name, "a") as text_file:
        print(text, end = ends_with, file = text_file)
    


def listdir(path):
    '''Returns dir with full path'''
    return sorted([os.path.join(path, f) for f in os.listdir(path)])



def line_list(line, n = None):
    ''' Returns line as list without spaces'''
    line = line.replace('\n', '')
    line = line.split(' ')
    line = list(filter(lambda val: val !=  '', line))
    if n == None:
        return line
    else:
        return line[n]



def log_paths():
    print_log('\n******* PATHS ********', paths.log_txt)
    print_log('\tDataset Path\t:{}'.format(paths.dataset_root), paths.log_txt)
    print_log('\tTarget Path\t:{}'.format(paths.target_root), paths.log_txt)
    print_log('\tPreprocessed Data\t:{}'.format(paths.data_root), paths.log_txt)
    print_log('\tFeatures\t:{}'.format(paths.feat_root), paths.log_txt)
    print_log('\tOutput\t:{}'.format(paths.out_root), paths.log_txt)


def remove_windows(frame_path):
    ''' Replaces frames having (blue) windows screens
        with black images'''
    
    frames = listdir(frame_path)
    frames = [ f for f in frames if f.endswith('.jpg')]

    for f in frames:
        image = cv2.imread(f) #reads brg

        channels = np.sum(image, axis=(0,1))

        if channels[0] > channels[1]:
            image = np.zeros_like(image)
            cv2.imwrite(f, image)
    
    
    
def VAD(x, threshold = [-5, 1]):
    '''
    Applies Voice Activity Detection algorithm [1] 
    to input signal and returns voiced signal.

    x        : numpy array
                    input wav

    threshold       : list, tuple
                        Slience threshold of energy and ACF

    [1] https://wiki.aalto.fi/pages/viewpage.action?pageId=151500905
    '''
    # pad 
    x_pad = np.pad(x, (0, hparams.win_length), 
        'constant', constant_values = 0)
    
    # compute energy and corr
    start_idx = 0
    end_idx = hparams.win_length
    num_win = int(np.ceil((len(x_pad) - 
        hparams.win_length) / hparams.hop_length))
    
    win_hann = signal.windows.hann(hparams.win_length)

    energy = np.empty((num_win,))
    corr = np.empty((num_win,)) 

    idx = 0
    while end_idx < len(x_pad):
        win = x_pad[start_idx:end_idx]
        win  = win * win_hann

        energy[idx] = 10 * np.log10(np.sum(win**2) + np.finfo(float).eps)
        corr[idx] = np.sum(win[:-1] * win[1:])

        start_idx += hparams.hop_length
        end_idx += hparams.hop_length
        idx += 1

    # thresholding
    energy_t = energy > threshold[0]
    corr_t = corr > threshold[1]

    # align with wav time scale
    energy_a = np.repeat(energy_t, hparams.hop_length)
    corr_a = np.repeat(corr_t, hparams.hop_length)

    # trim padding
    energy_a = energy_a[:len(x)]
    corr_a = corr_a[:len(x)]

    # combine decisions with or
    voiced = np.maximum(energy_a, corr_a)

    # add tolerance
    voiced_copy = np.copy(voiced)

    tol = int(0.85 * hparams.samplerate)
    for idx in range(1, len(voiced)):
        if voiced[idx] == 1 and voiced[idx - 1] == 0:
            if idx <= tol:
                voiced_copy[:idx] = 1
            else:
                voiced_copy[idx-tol:idx] = 1
        if voiced[idx] == 0 and voiced[idx - 1] == 1:
            if idx + tol >= len(voiced):
                voiced_copy[idx:] = 1
            else:
                voiced_copy[idx:idx+tol] = 1

    return x * voiced_copy
    

    
def VAD_Win(x, threshold = [-5, 1], win_len_sec = 10):
    '''
    Applies VAD algorith via windows.
    Performs better then VAD over very long sequences.

    x               : numpy array
                        Input wav

    threshold       : list, tuple
                        Slience threshold of energy and ACF
    
    win_len_sec     : int
                        Window lengt in seconds

    '''
    # vad window
    win_len = win_len_sec * hparams.samplerate
    
    # init vad version
    x_vad = np.zeros_like(x)

    # iter
    start_idx = 0
    end_idx = win_len

    while end_idx < len(x):
        x_vad[start_idx:end_idx] = VAD(x[start_idx:end_idx], threshold)
        
        start_idx = end_idx
        end_idx += win_len

    return x_vad



def audio_augment(x, samplerate, transform = None):
    ''' Given single-channel audio data, applies 
    random transformation 
    
    x               : np array               
                        input audio
    
    samplerate      : int               
                        Sampling rate

    transform       : audiomentations Compose class            
                        Augmentation functions
    
    '''
    if transform is None:
        transform = au.Compose([
            au.Gain(min_gain_in_db = -12, max_gain_in_db = 12, p = 0.15),
            au.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.15),
            au.ApplyImpulseResponse(ir_path = paths.rir_path, p = 0.15),
            au.TanhDistortion(min_distortion = 0.01, max_distortion = 0.25, p = 0.15),
            au.LowPassFilter(min_cutoff_freq = 150, max_cutoff_freq = 7500, p = 0.2),
            au.HighPassFilter(min_cutoff_freq = 20, max_cutoff_freq = 2400, p = 0.2)
            ])

    return transform(samples = x, sample_rate = samplerate)
    
    

def image_augment(x, transform = None):
    ''' Given an x-ray image, applies 
    random transformation 
    
    x               : Torch Tensor or PIL Image
                        input x-ray

    transform       : Torchvision transform           
                        Augmentation functions
    
    '''
    if transform is None:
        transform = random.choice([
            tv.transforms.ColorJitter(brightness = 0, contrast = 0, saturation = 0, hue = 0),
            tv.transforms.GaussianBlur(kernel_size = (5, 5), sigma = (0.1, 2.0)),
            tv.transforms.RandomRotation(degrees = (0, 180)),
            tv.transforms.RandomAdjustSharpness(sharpness_factor = 2, p = 1),
            tv.transforms.RandomAutocontrast(p = 1),
            tv.transforms.RandomEqualize(p = 1)
            ])

    return transform(x)



def check_align(target_path):
    '''Checks lengths of audio channels and 
    compares duration with number of frames'''

    ch1_wav, samplerate = sf.read(os.path.join(target_path, 'Channel_1.wav'))
    ch2_wav, _ = sf.read(os.path.join(target_path, 'Channel_2.wav'))
    gh_wav, _ = sf.read(os.path.join(target_path, 'Channel_3.wav'))
    
    assert len(ch1_wav) == len(ch2_wav), "Channel 1 and 2 don\'t have same len"
    assert len(ch1_wav) == len(gh_wav), "Mics and GoPro don\'t have same len"

    frames = os.listdir(target_path)
    frames = [ f for f in frames if f.endswith('.jpg')]

    assert len(frames) == len(ch1_wav) // samplerate, 'Number of frames ({}) ' \
        'and audio duration ({}) don\'t match'.format(len(frames), len(ch1_wav) // samplerate)



def whose_mic(op_name):
    ''' Reads annotations and returns info about
    who (phyisician or assistant) wears which mic
    '''
    # read
    with open(paths.annot_file) as f:
        annots = json.load(f)

        # iter
        for annot in annots:
            if annot[0] == op_name:
                ch1 = annot[1]
                ch2 = annot[2]
                return ch1, ch2
                
                

def one_hot(size, idx):
    '''Return sizex1 shaped one hot torch tensor'''
    t = torch.zeros((size,))
    t[idx] = 1
    return t



def plot_ribbon(data, title, out_path = None, repeat = 512):
    ''' Plots color ribbon with legend
    
    data        : np.array [1xN]
                    Data to plot

    title       : str
                    Title and save name of the figure, e.g. OP name

    out_path    : str
                    path to save. If None, saves into hparams.out_root

    repeat      : int
                    Vertical width of the ribbon 
    '''
    if out_path is None:
        save_path = paths.out_root + '/' + title
    else:
        save_path = out_path +  '/' + title

    ## Hard Coded Variable ###
    # Labels: should be same as dict in data.py get_label function
    phases = ['Preperation', 'Puncture', 'GuideWire', 'CathPlacement', 
        'CathPositioning', 'CathAdjustment', 'CathControl', 'Closing', 'Transition']

    # ensure data type
    assert type(data) == type(np.zeros([1, 1])), "Input data should be a numpy array"

    # ensure horizontal
    if data.shape[1] == 1:
        data = np.transpose(data)
    
    data = np.repeat(data, repeats = repeat, axis = 0)
    formatter = matplotlib.ticker.FuncFormatter(lambda s, 
        x: time.strftime('%M:%S', time.gmtime(s // 60)))
    xtick_pos = np.linspace(0, data.shape[1], data.shape[1] // 350)

    # discrete cmap
    def_cmap = plt.cm.get_cmap('tab10')
    color_list = def_cmap(np.linspace(0, 1, 9))
    disc_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('DMap', color_list, 9)

    # plot
    plt.figure(dpi = 600)
    plt.matshow(data, cmap = disc_cmap, vmin = 0, vmax = 8)
    plt.grid(False)
    plt.yticks([])
    plt.clim(-0.5, 8.5)
    cbar = plt.colorbar(ticks = range(len(phases)))
    cbar.ax.set_yticks(np.arange(len(phases)), labels = phases)
    plt.xticks(xtick_pos, fontsize = 18)
    plt.gca().xaxis.tick_bottom()
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlabel('Time (HH:MM)')
    plt.title(title, fontsize = 20, pad = 10)
    plt.savefig(save_path, bbox_inches = 'tight')
    plt.close('all')



def plot_parameters():
    ''' Plots loss functions and evaluation metrics
    
    '''
    params = np.load(os.path.join(paths.out_root, 
        'epoch_params.npy'), allow_pickle = True)
    
    train_loss = params.item().get('train_loss')
    val_loss = params.item().get('val_loss')
    val_scores = params.item().get('val_scores')
    metrics = params.item().get('metrics')
    modality_scales = params.item().get('modality_scales')

    # Losses
    plt.figure(dpi = 600, constrained_layout = True) 
    plt.style.use('fivethirtyeight')
    plt.plot(train_loss,  linewidth = 2, label = 'Train Loss')
    plt.plot(val_loss,  linewidth = 2, label = 'Valid Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Loss Functions')
    plt.xlabel('Epochs')
    plt.savefig(paths.out_root + '/losses.png')
    plt.close('all')
    
    # Metrics
    unused_metrics = np.where(val_scores[0, :] == 0)[0]

    plt.figure(dpi = 600, constrained_layout = True) #figsize = (16, 9),
    plt.style.use('fivethirtyeight')
    
    for i, metric in enumerate(metrics):
        if metric == 'acc':
            m = val_scores[:, 0]
        if metric == 'recall':
            m = val_scores[:, 1]
        if metric == 'precision':
            m = val_scores[:, 2]
        if metric == 'f1':
            m = val_scores[:, 3]
        if metric == 'jaccard':
            m = val_scores[:, 4]
    
        ax = plt.subplot(5 - len(unused_metrics), 1, i + 1)
        ax.plot(m, linewidth = 2, label = metric)
        ax.set_title(metric)
        ax.set_xlabel('Epochs')
        ax.grid(True)
        ax.legend()
    
    plt.savefig(paths.out_root + '/metrics.png')
    plt.close('all')



def get_datasets_partitions():
    '''divides operstions into training, validation and test sets
    
    Note: 
        Data in some operations are not perfect. For example, operation could be 
        taken over by another surgent at some point but would not wear a mic, 
        causing a missing audio channel. Therefore, we grouped all ops into three 
        categories according to severity of data loss and randomly diveded afterwards.
    '''
    train_set, valid_set, test_set = list(), list(), list()

    good_ops = ['OP_001', 'OP_004', 'OP_010', 'OP_012', 'OP_014', 'OP_025',
        'OP_027', 'OP_028', 'OP_029', 'OP_030', 'OP_031', 'OP_033', 'OP_034',
        'OP_035', 'OP_036', 'OP_038', 'OP_039', 'OP_040']
    random.Random(1881).shuffle(good_ops)

    mid_ops = [ 'OP_005', 'OP_006', 'OP_007', 'OP_015', 'OP_016', 'OP_017', 
        'OP_019', 'OP_026','OP_032']
    random.Random(1881).shuffle(mid_ops)

    low_ops = [ 'OP_002', 'OP_003', 'OP_009', 'OP_011', 'OP_013', 'OP_018', 
        'OP_019', 'OP_020', 'OP_021', 'OP_022', 'OP_023', 'OP_024', 'OP_037']
    random.Random(1881).shuffle(mid_ops)

    for i, op in enumerate(good_ops):
        if i < int(hparams.data_partition[0]* len(good_ops)):
            train_set.append(op)
        elif i < int((hparams.data_partition[0] + 
            hparams.data_partition[1]) * len(good_ops)):
            valid_set.append(op)
        else:
            test_set.append(op)

    for i, op in enumerate(mid_ops):
        if i <= int(hparams.data_partition[0] * len(mid_ops)):
            train_set.append(op)
        elif i <= int((hparams.data_partition[0] + 
            hparams.data_partition[1]) * len(mid_ops)):
            valid_set.append(op)
        else:
            test_set.append(op)

    if hparams.include_low_ops:
        for i, op in enumerate(low_ops):
            if i <= int(hparams.data_partition[0] * len(mid_ops)):
                train_set.append(op)
            elif i <= int((hparams.data_partition[0] + 
                hparams.data_partition[1]) * len(mid_ops)):
                valid_set.append(op)

    # aug ops
    if hparams.data_aug:
        aug_ops = [f for f in os.listdir(paths.feat_root) if f.endswith('A')]
        for op in aug_ops:
            train_set.append(op)

    return sorted(train_set), sorted(valid_set), sorted(test_set)