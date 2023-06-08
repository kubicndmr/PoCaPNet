import os

# Paths for dataset
dataset_root = '/DATA/PCP_Dataset' 
target_root = './'
annot_file = 'annotations.json'

# Paths for pre-processed data
data_root = '/DATA/data' 
pre_proc = [data_root] 

# Paths for extracted features
feat_root = '/DATA/features'
features = [feat_root]

# Paths for output
out_root = os.path.join(target_root, 'ldam_loss')
log_txt = os.path.join(out_root, 'log.txt')
outputs = [out_root]

# Other
rir_path = '/DATA/RIR'      # room impulse responses

# Function for initializing folders
def init_dirs(dir_mode):
    '''Initalize target directories.
    '''
    if dir_mode == 'preproc':
        path_set = pre_proc
    if dir_mode == 'features':
        path_set = features
    if dir_mode == 'train':
        path_set = outputs

    for p in path_set:
        if not os.path.exists(p):
            os.makedirs(p)