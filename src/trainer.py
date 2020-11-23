import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from src.utils.dataset import GraphData

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp


class Trainer:
    def __init__(self, args, net, G_data):
        self.args = args
        self.net = net
        self.feat_dim = G_data.feat_dim
        # self.fold_idx = G_data.fold_idx
        self.fold_idx = 0
        self.init(args, G_data.train_gs, G_data.valid_gs, G_data.test_gs)
        if torch.cuda.is_available():
            self.net.cuda()

    def init(self, args, train_gs, valid_gs, test_gs):
        print('#train: %d, valid: %d, #test: %d' % (len(train_gs), len(valid_gs), len(test_gs)))
        train_data = GraphData(train_gs, self.feat_dim)
        valid_data = GraphData(valid_gs, self.feat_dim)
        test_data = GraphData(test_gs, self.feat_dim)
        self.train_d = train_data.loader(self.args.batch, True)
        self.valid_d = valid_data.loader(self.args.batch, False)
        self.test_d = test_data.loader(self.args.batch, False)
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.args.lr, amsgrad=True,
            weight_decay=0.0008)

    def to_cuda(self, gs):
        if torch.cuda.is_available():
            if type(gs) == list:
                return [g.cuda() for g in gs]
            return gs.cuda()
        return gs

    def run_epoch(self, epoch, data, model, optimizer):
        losses, accs, n_samples = [], [], 0
        for batch in tqdm(data, desc=str(epoch), unit='b'):
            cur_len, gs, hs, ys = batch
            gs, hs, ys = map(self.to_cuda, [gs, hs, ys])
            loss, acc, _ = model(gs, hs, ys)
            losses.append(loss*cur_len)
            accs.append(acc*cur_len)
            n_samples += cur_len
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        avg_loss, avg_acc = sum(losses) / n_samples, sum(accs) / n_samples
        return avg_loss.item(), avg_acc.item()

    def evaluate(self, epoch, data, model, thr=None, return_best_thr=False):
        model.eval()
        total = 0.
        prec, rec, f1 = 0., 0., 0.
        y_true, y_pred, y_score = [], [], []
        losses, accs, n_samples = [], [], 0
        for batch in tqdm(data, desc=str(epoch), unit='b'):
            cur_len, gs, hs, ys = batch
            gs, hs, ys = map(self.to_cuda, [gs, hs, ys])
            loss, acc, out = model(gs, hs, ys)
            losses.append(loss*cur_len)
            accs.append(acc*cur_len)
            n_samples += cur_len

            y_true += ys.data.tolist()
            y_pred += out.max(1)[1].data.tolist()
            y_score += out[:, 1].data.tolist()
            total += cur_len

        if thr is not None:
            logger.info("using threshold %.4f", thr)
            y_score = np.array(y_score)
            y_pred = np.zeros_like(y_score)
            y_pred[y_score > thr] = 1

        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
        auc = roc_auc_score(y_true, y_score)
        logger.info("loss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
                    sum(losses) / n_samples, auc, prec, rec, f1)

        # avg_loss, avg_acc = sum(losses) / n_samples, sum(accs) / n_samples
        # return avg_loss.item(), avg_acc.item()

        if return_best_thr:
            precs, recs, thrs = precision_recall_curve(y_true, y_score)
            f1s = 2 * precs * recs / (precs + recs)
            f1s = f1s[:-1]
            thrs = thrs[~np.isnan(f1s)]
            f1s = f1s[~np.isnan(f1s)]
            best_thr = thrs[np.argmax(f1s)]
            logger.info("best threshold=%4f, f1=%.4f", best_thr, np.max(f1s))
            return sum(losses) / n_samples, [prec, rec, f1, auc], best_thr
        else:
            return sum(losses) / n_samples, [prec, rec, f1, auc], None

    def train(self):
        max_acc = 0.0
        train_str = 'Train epoch %d: loss %.5f acc %.5f'
        test_str = 'Test epoch %d: loss %.5f acc %.5f max %.5f'
        line_str = '%d:\t%.5f\n'
        best_thr = None
        best_test_metrics = None
        best_valid_metrics = None

        # Test
        val_loss, val_metrics, thr = self.evaluate(-1, self.valid_d, self.net, return_best_thr=True)
        print("validation loss:", val_loss, "metrics", val_metrics, "thr:", thr)
        test_loss, test_metrics, _ = self.evaluate(-1, self.test_d, self.net, thr=thr)
        print("test loss:", test_loss, "metrics", test_metrics)

        for e_id in range(self.args.num_epochs):
            self.net.train()
            loss, acc = self.run_epoch(
                e_id, self.train_d, self.net, self.optimizer)
            print(train_str % (e_id, loss, acc))

            with torch.no_grad():
                self.net.eval()
                # loss, acc = self.run_epoch(e_id, self.test_d, self.net, None)
                val_loss, val_metrics, thr = self.evaluate(e_id, self.valid_d, self.net, return_best_thr=True)
                print("validation loss:", val_loss, "metrics", val_metrics, "thr:", thr)
                test_loss, test_metrics, _ = self.evaluate(e_id, self.test_d, self.net, thr=thr)
                print("test loss:", test_loss, "metrics", test_metrics)
            if val_metrics[-1] > max_acc:
                max_acc = val_metrics[-1]
                best_thr = thr
                best_valid_metrics = val_metrics
                best_test_metrics = test_metrics
            # max_acc = max(max_acc, acc)
            # print(test_str % (e_id, loss, acc, max_acc))

        # with open(self.args.acc_file, 'a+') as f:
        #     f.write(line_str % (self.fold_idx, max_acc))

        print("best validation metrics", best_valid_metrics, "thr:", best_thr)
        print("best test metrics", best_test_metrics)
