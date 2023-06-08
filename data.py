import numpy as np
import torch
import json
import os

from torch.utils.data import Dataset, DataLoader
from utils import helper, paths, hparams

class OPDataset(Dataset):
    def __init__(self, op):
        self.data = self.get_data(op)

    def __getitem__(self,index):
        # image
        im = torch.empty((hparams.resolution, 1024))
        for i, d in enumerate(self.data[0][index]):
            im[i, :] = torch.load(d)

        # audio channels
        ch1 = torch.load(self.data[1][index]).squeeze()
        ch2 = torch.load(self.data[2][index]).squeeze()
        ch3 = torch.load(self.data[3][index]).squeeze()
        
        # labels
        y = torch.tensor(self.data[4][index])

        # positional encoding
        pe = self.positional_encoding(ndim = 1024, idx = index)
        
        return (im, ch1, ch2, ch3, pe, y)  
    

    def __len__(self):
        return len(self.data[0])


    def positional_encoding(self, ndim, idx):
        ''' from "Attention is all you need"
        '''
        pe = np.zeros((ndim, 1))
        
        for t in range(ndim):
            if t % 2 == 0:
                pe[t] = np.sin(idx / np.power(10000, t / ndim))
            elif t % 2 == 1:
                pe[t] = np.cos(idx / np.power(10000, t / ndim))

        return torch.from_numpy(pe.T)


    def get_label(self, data_path, data_length):
        '''Reads annotations file and 
        creates label vector including phase 
        labels for each second'''

        # Label dictionary
        label_dic = {'Preparation' : 0, 'Puncture' : 1, 'GuideWire' : 2, 
            'CathPlacement' : 3, 'CathPositioning' : 4, 'CathAdjustment' : 5, 
            'CathControl' : 6, 'Closing' : 7, 'Transition' : 8}
        
        # init according to preparation key
        labels = np.ones((data_length, 1)) * 99

        # get name
        op_name = data_path.split('/')[-1]
        if op_name.endswith('A'):
            op_name = op_name[:-1]
        
        # read
        with open(paths.annot_file) as f:
            annots = json.load(f)

            # iter
            for annot in annots:
                if annot[0] == op_name:
                    time_stamps = annot[3]
                    phases = annot[4]
                    for p, t in zip(phases, time_stamps):
                        t = t.split('-')
                        t_start = helper.time2sec(t[0])
                        t_end = helper.time2sec(t[1]) + 1
                        if p == 'Closing':
                            labels[t_start:] = label_dic[p]
                        else: 
                            labels[t_start:t_end] = label_dic[p]
        
        assert len(np.where(labels == 99)[0]) == 0

        # plot ribbon
        if hparams.plot_gtribbons:
            if not os.path.exists(os.path.join(paths.out_root, 'label_ribbons')):
                os.mkdir(os.path.join(paths.out_root, 'label_ribbons'))    
            helper.plot_ribbon(data = labels, title = op_name, 
                out_path = os.path.join(paths.out_root, 'label_ribbons'))

        return labels


    def receptive_windowing(self, data_list):
        '''
        Creates list of lists with length of res considering casuality parameter.

        data_list   : list
                        data list to window

        TODO this function could have written much easier with padding, but works
        '''
        res = hparams.resolution
        out_list = list()

        if hparams.casual:
            for i in range(1, len(data_list) + 1):
                if i < res:
                    window_list = [data_list[0]] * res
                    window_list[-i:] = data_list[:i]
                else:
                    window_list = data_list[i - res : i]
                
                out_list.append(window_list)

        else:
            for i in range(1, len(data_list) + 1):
                win = (res - 1) // 2 
                if i < win:
                    window_list = [data_list[0]] * res
                    window_list[-(i + win):] = data_list[:(i + win)]
                elif i >= len(data_list) - win:
                    window_list = [data_list[-1]] * res
                    window_list[:(len(data_list) - i + win)] = data_list[i - win:]
                else:
                    window_list = data_list[i - win : i + win + 1]

                out_list.append(window_list)

        return out_list

        
    def get_data(self, op):
        '''
        reads op data from pointed dataset

        op          : str
                        target op path
        '''
        op_data = helper.listdir(os.path.join(paths.feat_root, op))

        # read data
        image_data = list()
        channel_1 = list()
        channel_2 = list()
        channel_3 = list()

        feat = [ d for d in op_data if 'frame' in d]
        feat = self.receptive_windowing(feat)
        for f in feat:
            image_data.append(f)

        feat = [ d for d in op_data if 'ch1' in d]
        for f in feat:
            channel_1.append(f)
            
        feat = [ d for d in op_data if 'ch2' in d]
        for f in feat:
            channel_2.append(f)

        feat = [ d for d in op_data if 'gh' in d]
        for f in feat:
            channel_3.append(f)

        label = self.get_label(op, len(image_data))

        return [image_data, channel_1, channel_2, channel_3, label]


    def pos_weight(self):
        phases = np.zeros((9,))
        
        for l in self.data[4]:
            phases[int(l)] += 1

        weights = len(self.data[4]) / phases

        return weights


def get_train_dataset(op_list):
    helper.print_log('Data partition mode: train', 
        paths.log_txt, display= False)
    helper.print_log('\tDataset: {}'.format(op_list), 
        paths.log_txt, display = False)
    helper.print_log('\tNumber of Operations: {}'.format(len(op_list)), 
        paths.log_txt, display = False)

    dataset = list()
    phases = np.zeros((9,))
    ds_len = 0

    for op in op_list:
        ds = OPDataset(op)

        dl = DataLoader(dataset = ds, 
                batch_size = hparams.batch_size, 
                shuffle = hparams.dset_shuffle)
        ds_len += ds.__len__()
            
        dur = len(os.listdir(os.path.join(paths.data_root, op))) - 3
        labels = ds.get_label(op, dur)
        for l in labels:
            phases[int(l)] += 1
        
        dataset.append([op, dl])

    phases_ = np.copy(phases)  # manipulate transition phase
    phases_[-1] = np.max(phases_) * 2
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, phases_)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(phases_)

    helper.print_log('\tPhase weights: {}'.format(list(per_cls_weights)), 
        paths.log_txt, display= False)

    helper.print_log('\tTraining dataset size: {:,}\n\n'.format(ds_len), 
        paths.log_txt, display= False)

    per_cls_weights = torch.from_numpy(per_cls_weights)

    return [dataset, ds_len], [phases_, per_cls_weights]


def get_valid_dataset(op_list):
    helper.print_log('Data partition mode: valid', 
        paths.log_txt, display= False)
    helper.print_log('\tDataset: {}'.format(op_list), 
        paths.log_txt, display = False)
    helper.print_log('\tNumber of Operations: {}'.format(len(op_list)), 
        paths.log_txt, display = False)

    dataset = list()
    ds_len = 0

    for op in op_list:
        ds = OPDataset(op)
        dl = DataLoader(dataset = ds, 
                batch_size = hparams.batch_size, 
                shuffle = hparams.dset_shuffle)
        ds_len += ds.__len__()
        dataset.append([op, dl])
        dur = len(os.listdir(os.path.join(paths.data_root, op))) - 3
        _ = ds.get_label(op, dur)

    helper.print_log('\tValidation dataset size: {:,}\n\n'.format(ds_len), 
        paths.log_txt, display= False)

    return [dataset, ds_len]