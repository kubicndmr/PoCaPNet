import os

import numpy as np
import torch as t

from utils.helper import listdir, print_log, plot_ribbon
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from utils import paths, hparams
import matplotlib.pyplot as plt

class Trainer:
    
    def __init__(self,               
                 model,                     # Model to be trained.
                 crit = None,               # Loss function
                 optim = None,              # Optimiser
                 sched = None,              # Scheduler
                 trainset = None,           # Training data set
                 valset = None,             # Validation data set
                 pos_weights = None,        # Phase weights
                 patience = None,           # Early stopping patience
                 delta = 0,                 # Early stopping delta
                 eval_metrics = ['acc']     # Metrics for evaluation
                 ):  

        self._model = model
        self._crit = crit
        self._optim = optim
        self._sched = sched
        self._trainset = trainset[0]
        self._trainset_len = trainset[1]
        self._valset = valset[0]
        self._valset_len = valset[1]
        self._pos_weights = pos_weights
        self._patience = patience
        self._delta = delta
        self.eval_metrics = eval_metrics

        if t.cuda.is_available():
            self._model = model.cuda()
            self.device = t.device("cuda")
            if self._crit is not None:
                self._crit = crit.cuda()
        else:
            self.device = t.device("cpu")
    
    
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 
            paths.out_root + '/checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    

    def restore_checkpoint(self, ckp):
        print_log('\nBest pretrained model is: {}\n'.format(ckp), paths.log_txt)
        ckp = t.load(ckp, 'cuda' if t.cuda.is_available() else None)
        self._model.load_state_dict(ckp['state_dict'])


    def save_params(self, train_loss, val_loss, val_scores, lr):
        np.save(paths.out_root + '/epoch_params', {"train_loss": train_loss, 
            "val_loss": val_loss, "val_scores": val_scores, "lr": lr, 
            "metrics": self.eval_metrics})


    def evaluate(self, y, y_hat, op_name = None):
        # init metrics
        acc_, recall_, precision_, = None, None, None
        f1_, jaccard_ = None, None

        # flatten lists
        y = [item for sublist in y for item in sublist]
        y_hat = [item for sublist in y_hat for item in sublist]

        # convert np array
        y_hat = np.expand_dims(y_hat, axis = 1)
        y = np.array(y)

        # exclude transition periods
        tr_idx = np.where(y == 8)[0]
        y_hat = np.delete(y_hat, tr_idx)
        y = np.delete(y, tr_idx)

        if 'acc' in self.eval_metrics:
            acc_ = accuracy_score(y, y_hat)
        if 'recall' in self.eval_metrics:
            recall_ = recall_score(y, y_hat, average = 'weighted')
        if 'precision' in self.eval_metrics:
            precision_ = precision_score(y, y_hat, average = 'weighted')
        if 'f1' in self.eval_metrics:
            f1_ = f1_score(y, y_hat, average = 'weighted')
        if 'jaccard' in self.eval_metrics:
            jaccard_ = jaccard_score(y, y_hat, average = 'weighted')
        if 'conf_mat' in self.eval_metrics:
            conf_mat = confusion_matrix(y, y_hat)
            disp = ConfusionMatrixDisplay(conf_mat)
            disp.plot()
            plt.title(op_name)
            plt.savefig(paths.out_root + '/CM_' + op_name)

        # plot ribbon
        if 'Test' in op_name:
            plot_ribbon(np.expand_dims(y_hat, 1), op_name)

        return {'acc': acc_, 'recall': recall_, 'precision': precision_, 
            'f1': f1_, 'jaccard': jaccard_}


    def train_step(self, x, c1, c2, c3, pe, y_prev, y):
        self._optim.zero_grad()
        y_hat = self._model(x, c1, c2, c3, pe, y_prev)
        y = y.squeeze()
        e = 0
        for s in range(hparams.num_stages):
            e_s = self._crit(y_hat[:,:,s].squeeze().double(), y)
            e += e_s
        e.backward()
        self._optim.step()
        return (e, y_hat[:,:,-1])


    def val_test_step(self, x, c1, c2, c3, pe, y_prev, y):
        with t.no_grad():
            y_hat = self._model(x, c1, c2, c3, pe, y_prev)
            y = y. squeeze()
            e = 0
            for s in range(hparams.num_stages):
                e_s = self._crit(y_hat[:,:,s].squeeze().double(), y)
                e += e_s
            return (e, y_hat[:,:,-1])
    

    def train_epoch(self):
        print_log('\t__training__:', paths.log_txt)
        self._model.train()

        op_count = 0
        iter_count = 0 
        epoch_loss = 0
        
        for op_name, op_dataloader in self._trainset:

            op_loss = 0
            y_prev = t.zeros((1, hparams.dim), device = self.device)

            for x_, c1_, c2_, c3_, pe_, y_ in op_dataloader:
                print('\t\tprogress: {:.2f} %'.format(iter_count / 
                    self._trainset_len * 100), end = '\r')
                
                x = x_.squeeze(dim = 0).clone().detach().float().to(self.device)
                c1 = c1_.squeeze(dim = 0).clone().detach().float().to(self.device)
                c2 = c2_.squeeze(dim = 0).clone().detach().float().to(self.device)
                c3 = c3_.squeeze(dim = 0).clone().detach().float().to(self.device)
                pe = pe_.squeeze(dim = 0).clone().detach().float().to(self.device)
                y = y_.squeeze(dim = 0).clone().detach().long().to(self.device)
                y_prev_ = y_prev.clone().detach().float().to(self.device)
                (e, y_hat) = self.train_step(x, c1, c2, c3, pe, y_prev_, y)
                
                y_prev[0, -x.shape[0]:] = t.argmax(y_hat, dim = 1)
                y_prev = t.roll(y_prev, x.shape[0])

                op_loss += e

                iter_count += x.shape[0]
            
            op_count += 1
            epoch_loss += op_loss

        print('\t\t', end = '\r')

        return epoch_loss.item() / op_count
    

    def val_test(self, out_prefix):
        print_log('\t__validation_test__:', paths.log_txt)
        self._model.eval()
     
        op_count = 0
        epoch_loss = 0
        eval_scores = np.zeros((1, 5))

        for op_name, op_dataloader in self._valset:
            print('\t\tworking on: {}'.format(op_name), end = '\r')
            
            op_loss = 0
            y_estim = list()
            y_ground = list()
            y_prev = t.zeros((1, hparams.dim), device = self.device)

            for x_, c1_, c2_, c3_, pe_, y_ in op_dataloader:
                x = x_.squeeze(dim = 0).clone().detach().float().to(self.device)
                c1 = c1_.squeeze(dim = 0).clone().detach().float().to(self.device)
                c2 = c2_.squeeze(dim = 0).clone().detach().float().to(self.device)
                c3 = c3_.squeeze(dim = 0).clone().detach().float().to(self.device)
                pe = pe_.squeeze(dim = 0).clone().detach().float().to(self.device)
                y = y_.squeeze(dim = 0).clone().detach().long().to(self.device)
                y_prev_ = y_prev.clone().detach().float().to(self.device)
                (e, y_hat) = self.val_test_step(x, c1, c2, c3, pe, y_prev_, y)

                y_prev[0, -x.shape[0]:] = t.argmax(y_hat, dim = 1)
                y_prev = t.roll(y_prev, x.shape[0])

                op_loss += e

                y_estim.append(t.argmax(y_hat, 
                    dim = 1).cpu().detach().numpy())
                y_ground.append(y.cpu().detach().numpy())

            op_count += 1
            epoch_loss += op_loss

            op_results = self.evaluate(y_ground, y_estim, 
                out_prefix + '_' + op_name)
            
            print('\t\t                    ', end = '\r')
            
            for metric in self.eval_metrics:
                if metric != 'conf_mat':
                    res = op_results[metric]
                    if res is not None:
                        print_log('\t\t{} \t {}\t:{:.5f}'.format(op_name, metric, res), 
                            paths.log_txt)
                        if metric == 'acc':
                            eval_scores[0, 0] += res
                        elif metric == 'recall':
                            eval_scores[0, 1] += res
                        elif metric == 'precision':
                            eval_scores[0, 2] += res
                        elif metric == 'f1':
                            eval_scores[0, 3] += res
                        elif metric == 'jaccard':
                            eval_scores[0, 4] += res

        # scale scores
        epoch_loss = epoch_loss.item()
        epoch_loss /= op_count
        eval_scores /= op_count

        return epoch_loss, eval_scores
    

    def restore_last_session(self):
        if os.path.exists(paths.out_root + '/checkpoints'):
            print('Restoring Last Session!!!')
            ckps = listdir(paths.out_root + '/checkpoints')
            if len(ckps) != 0:
                self.restore_checkpoint(ckps[-1])
            else:
                print('\nCan\'t find any ckp file.')
        else:
            os.mkdir(paths.out_root + '/checkpoints')
            print('Starting without a pre-trained model!!')


    def fit(self, epochs=-1):
        early_stop = False
        patience_counter = 0
    
        self.restore_last_session()
                 
        train_loss = np.zeros((epochs, 1))
        val_loss = np.zeros((epochs, 1))
        val_scores = np.zeros((epochs, 5))
        lr = np.zeros((epochs, 1))

        best_loss = 1E9 # just a large number

        print_log('\nEvaliation Metircs:\t{}'.format(self.eval_metrics), 
                paths.log_txt)

        for e in range(epochs):
            if early_stop == False:
                # print epoch number
                print_log('\nEpoch : {}/{}'.format(e + 1, epochs), paths.log_txt)

                # train
                train_loss[e] = self.train_epoch()
                print_log('\t\tloss\t:{:.5f}'.format(train_loss[e][0]), paths.log_txt)

                # validate
                val_loss[e], val_scores[e, :] = self.val_test(out_prefix = 'Epoch_' + str(e + 1))
                print_log('\t\tloss\t:{:.5f}'.format(val_loss[e][0]), paths.log_txt)            

                # scheduler step, get, display and step 
                if self._sched is not None:            
                    lr[e] = self._sched.get_last_lr()
                    print_log('\tl_rate:\t{}'.format(lr[e]), paths.log_txt)
                    self._sched.step()

                # save parameters
                self.save_params(train_loss, val_loss, val_scores, lr)
            
                if val_loss[e] < best_loss:
                    # save checkpoint and params
                    self.save_checkpoint(e)
                
                    # store best loss
                    best_loss = val_loss[e]

                    # reset counter
                    patience_counter = 0

                if val_loss[e] > best_loss + self._delta:
                    # increment
                    patience_counter += 1

                if self._patience is not None and self._patience < patience_counter:
                    early_stop = True

        return self._model