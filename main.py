import torch 
import data

from utils.helper import print_log, plot_parameters, get_datasets_partitions, log_paths
from model.pocapnet import PoCaPNet
from model.losses import LDAMLoss
from utils import paths, hparams
from datetime import datetime
from trainer import Trainer
from tester import Tester

# initalize paths
paths.init_dirs('train')


# log
print_log('\n\tTraining "{}" started at: {} \n'.format(paths.out_root[2:],
        datetime.now().strftime('%d-%m-%Y %H:%M:%S')), paths.log_txt)
print_log('GPU: {}'.format(torch.cuda.get_device_name()),  paths.log_txt)
print_log('Properties: {}'.format(torch.cuda.get_device_properties("cuda")), 
    paths.log_txt)
log_paths()

hparams.log_variables()


# get datasets
train_ops, valid_ops, test_ops = get_datasets_partitions()
trainset, pos_weights = data.get_train_dataset(train_ops)
validset = data.get_valid_dataset(valid_ops)
testset = data.get_valid_dataset(test_ops)


# set up model
model = PoCaPNet()


# print model
print_log('Number of Parameters: {:,}'.format(sum(p.numel() 
    for p in model.parameters() if p.requires_grad)), paths.log_txt)


# loss function
criteria = LDAMLoss( cls_num_list = pos_weights[0], weight = pos_weights[1])


# set up optimizer 
optimizer = torch.optim.Adam(model.parameters(), 
    lr = hparams.learning_rate, weight_decay = hparams.weight_decay)


# set up lr_scheduler
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.5)
scheduler = None
if scheduler is not None:
    print_log('\nScheduler: {}\n'.format(scheduler.state_dict()), 
        paths.log_txt, display = False)


# initialize Trainer object 
trainer = Trainer(model, criteria, optimizer, scheduler, trainset, 
                  validset, pos_weights, hparams.patience)


# train 
model = trainer.fit(hparams.epoch)


# plot results
plot_parameters()


# test
tester = Tester(model, criteria, testset)
tester.test()


# Log finish time
print_log('\n\tTraining finished at: {} \n'.format(
        datetime.now().strftime('%d-%m-%Y %H:%M:%S')), paths.log_txt)