from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import copy
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


from model import *
from data_process import *
from args import *
from TFT import *
from loss import *
from myModel import *

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        return self.best_model

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.best_model = copy.deepcopy(model)
        self.val_loss_min = val_loss

if __name__ == '__main__':

    Inum = 100
    smooth = True
    train_epoch = 0

    fold_data = "..\\..\\results"
    fold_model = "model\\ER"

    net = ["ER"]
    d = [10]
    R = [2]

    def pretrain(info):

        args = CtTransformer_args_parser()
        path = []
        Dtr = []
        Dva = []
        count = 0

        # max_list,min_list,max_rt,min_rt = getBound(args,fold_data,d,R,Inum,1)

        path = fold_model + "\\myModel_pretrain_" + info + ".pkl"
        Dtr,Dva,Dtr_all,Dva_all = get_Train_range(args,fold_data,net,d,R,Inum,smooth,[],[],[],[])
        model = CtTransformer(args,head_type="pretrain").to(device)
        pretrainModel(args,model,Dtr,Dva,path)

    def pretrainModel(args,model,Dtr,Val,path):
        optimizer = args.optimizer
        weight_decay = args.weight_decay
        lr = args.lr
        step_size = args.step_size
        gamma = args.gamma
        epochs = args.epochs
        loss_type = args.loss_type
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        loss_function = PretrainLoss().to(device)

        if optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=0.9, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        # training
        # min_epochs = 1
        best_model = None
        model.train()
        train_loss_all = []
        train_loss_mean = []
        val_loss_all = []
        val_loss_mean = []

        for epoch in tqdm(range(epochs)):
            train_loss = []
            for (seq, label) in Dtr:
                out,out_mask,mask,length = patch_masking(seq,args.patchLen,0.4)
                pred = model(out_mask,length)
                loss = loss_function(pred, out,mask)
                train_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss_all.extend(train_loss)
            train_loss_mean.append(np.mean(train_loss))
            scheduler.step()

            # validation
            val_loss_m,val_loss = get_val_loss(args,model, Val)
            val_loss_all.extend(val_loss)
            val_loss_mean.append(val_loss_m)
            # if epoch > min_epochs and val_loss < min_val_loss:
            #     # min_val_loss = val_loss
            #     best_model = copy.deepcopy(model)
            best_model = early_stopping(val_loss_m, model)
            # model.train()
            if early_stopping.early_stop:
                print("Early stopping")
                break

            print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss_m))
            model.train()
        train_epoch = epoch
        new_path = path.rstrip(".pkl")
        draw_train(new_path,train_epoch,train_loss_all,train_loss_mean,val_loss_all,val_loss_mean)
        state = {'models': best_model.state_dict()}
        torch.save(state, path)

    def draw_train(path,epoch,train_loss_all,train_loss_mean,val_loss_all,val_loss_mean):
        img_path = path + "_train.png"
        train_len = len(train_loss_all)
        delta = int(train_len / (epoch + 1))
        median_delta = int(delta/2)
        x = [i for i in range(train_len)]
        plt.plot(x,train_loss_all,'darkgrey')
        plt.xticks(x, ['']*len(x))
        x1 = [i for i in range(median_delta,(epoch + 1)*delta,delta)]
        label = [i for i in range(1,epoch + 2)]
        plt.plot(x1,train_loss_mean,'r')
        plt.xticks(x1, label)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(img_path)
        plt.close()

        img_path = path + "_val.png"
        val_len = len(val_loss_all)
        delta = int(val_len / (epoch + 1))
        median_delta = int(delta/2)
        x = [i for i in range(val_len)]
        plt.plot(x,val_loss_all,'darkgrey')
        plt.xticks(x, ['']*len(x))
        x1 = [i for i in range(median_delta,(epoch + 1)*delta,delta)]
        label = [i for i in range(1,epoch + 2)]
        plt.plot(x1,val_loss_mean,'r')
        plt.xticks(x1, label)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(img_path)
        plt.close()

    def get_val_loss(args,model, Val):
        model.eval()
        loss_function = PretrainLoss().to(device)
        val_loss = []
        for (seq, label) in Val:
            with torch.no_grad():
                out,out_mask,mask,length = patch_masking(seq,args.patchLen,0.4)
                pred = model(out_mask,length)
                loss = loss_function(pred, out,mask)
                val_loss.append(loss.item())
        return np.mean(val_loss),val_loss
    
    info = " "
    pretrain(info)


