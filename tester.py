import torch as t

from utils.helper import print_log
from model.pocapnet import PoCaPNet
from trainer import Trainer
from utils import paths

class Tester(Trainer):

    def __init__(self,
                 model,                     # Model to be trained.
                 crit,                      # Loss function              
                 testset,                   # Testset
                 eval_metrics = ['acc', 'recall', 'precision', 'f1', 'jaccard']         
                 ):

        self._model = model
        self._crit = crit
        self._valset = testset[0]
        self._valset_len = testset[1]
        self.eval_metrics = eval_metrics

        if t.cuda.is_available():
            self._model = model.cuda()
            self._crit = crit.cuda()
            self.device = t.device("cuda")
        else:
            self.device = t.device("cpu")


    def test(self):
        print_log('\n\tTesting:\n', paths.log_txt)

        self.restore_last_session()

        test_loss, eval_scores = self.val_test(out_prefix = 'Test')
        
        print_log('\t\tloss\t:{:.5f}'.format(test_loss), paths.log_txt)