from utils import paths, helper

# Training
resolution = 7                      # temporal resolution, seconds. Must be an odd number
casual = True                       # time window casuality
data_partition = [0.6, 0.2, 0.2]    # dataset partition ratio, training:validation:test
batch_size = 500
epoch  = 100
learning_rate = 9e-6                # learning rate
patience = 5                        # patience for early stopper
weight_decay = 1e-6                 # l2 reg

# Data
use_vad = False                     # VAD flag
dset_shuffle = False                # shuffle pytorch dataloader
aug_ratio = 0.25                    # portion of data augmentation to include in training set
data_aug = False                    # Wheter include augmented data
include_low_ops = False             # Whether to include severely corrupted recordings

# Multi Stage Model
num_stages = 2
num_layers = 10
num_f_maps = 64
dim = 1024
num_classes = 9
causal_conv = False

# DSP
samplerate = 16000
n_fft = 2048
n_mels = 40                         # same as avlnet, if changed, can't use pretrained weights
hop_length = 200                    # 12.5ms
win_length = 800                    # 50ms
image_shape = (224, 224)            # same as resnet
coef = 0.97                         # pre-emphasis filter coefficient
sec_win = int(samplerate / hop_length)
res_win = resolution * sec_win

# Visualization 
plot_gtribbons = True              # plot ribbon graphs for true labels 

def log_variables():
    helper.print_log('\n******* HPARAMS ********', paths.log_txt)
    helper.print_log('\tTime resolution\t:{}'.format(resolution), paths.log_txt)
    helper.print_log('\tCasuality\t:{}'.format(str(casual)), paths.log_txt)
    helper.print_log('\tData partition\t:{}'.format(data_partition), paths.log_txt)
    helper.print_log('\tBacth size\t:{}'.format(batch_size), paths.log_txt)
    helper.print_log('\tEpochs\t\t:{}'.format(epoch), paths.log_txt)
    helper.print_log('\tLearning rate\t:{}'.format(learning_rate), paths.log_txt)
    helper.print_log('\tWeight decay\t:{}'.format(weight_decay), paths.log_txt)
    helper.print_log('\tEarly stopper patience\t:{}'.format(patience), paths.log_txt)
    helper.print_log('\tShuffle\t:{}'.format(dset_shuffle), paths.log_txt)
    helper.print_log('\tData Augmentation\t:{}'.format(data_aug), paths.log_txt)
    helper.print_log('\n******* MS-TCN ********', paths.log_txt)
    helper.print_log('\tNumber of Stages\t:{}'.format(num_stages), paths.log_txt)
    helper.print_log('\tNumber of Layers in a Stage\t:{}'.format(num_layers), paths.log_txt)
    helper.print_log('\tNumber of Kernels\t:{}'.format(num_f_maps), paths.log_txt)
    helper.print_log('\tDimension\t:{}'.format(dim), paths.log_txt)
    helper.print_log('\tCasual Convolution\t:{}'.format(causal_conv), paths.log_txt)
    helper.print_log('\n******* DSP ********', paths.log_txt)
    helper.print_log('\tSampling rate\t:{}'.format(samplerate), paths.log_txt)
    helper.print_log('\tFFT window\t:{}'.format(n_fft), paths.log_txt)
    helper.print_log('\tWindow Length\t:{} (ms)'.format(win_length * 1000 / samplerate), paths.log_txt)
    helper.print_log('\tHop Length\t:{} (ms)'.format(hop_length * 1000 / samplerate), paths.log_txt)
    helper.print_log('\tX-ray input image size\t:{}\n'.format(image_shape), paths.log_txt)