from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import copy


from model import *
from data_process import *
from args import *
from TFT import *
from loss import *
from myModel import *
from modelAblation import *

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

    max_list = []
    min_list = []

    fold_data = "../data"
    fold_model = "model/ER"
    # fold_model = "model/SF"

    net = ["ER"]
    d = [10]
    # R = [1.5,2,2.5,3]
    R = [1.5,2,2.5,3]

    # net = ["SF"]
    # d = [10]
    # R = [2]

    def train(model_type):

        args = CtTransformer_args_parser()
        path = []

        max_list,min_list,max_rt,min_rt = getBound(args,fold_data,d,R,Inum,1)
        Dtr,Dva,Dtr_all,Dva_all = get_Train_range(args,fold_data,net,d,R,Inum,smooth,max_list,min_list,max_rt,min_rt)

        if model_type == "NoPatch":
            path = fold_model + "/NoPatch.pkl"
            model = NoPatch(args).to(device)
        
        elif model_type == "NoGRU":
            path = fold_model + "/NoGRU.pkl"
            model = NoGRU(args).to(device)
        
        elif model_type == "NoattenForward":
            path = fold_model + "/NoattenForward.pkl"
            model = NoattenForward(args).to(device)
            
        trainModel(args,model,Dtr,Dva,path,model_type)



    def trainModel(args,model,Dtr,Val,path,model_type):
        # model_path = path + "net=" + str(net) + "d=" + str(d) + "R=" + str(R) + ".pkl"
        optimizer = args.optimizer
        weight_decay = args.weight_decay
        lr = args.lr
        step_size = args.step_size
        gamma = args.gamma
        epochs = args.epochs
        loss_type = args.loss_type
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        if loss_type == 2:
            quantiles = [0.1, 0.5, 0.9]
            loss_function = QuantileLoss(quantiles).to(device)
        elif loss_type == 1:
            # loss_function = nn.MSELoss().to(device)
            loss_function = nn.SmoothL1Loss().to(device)
            # loss_function = HighdemensionLoss(args.output_size,0).to(device)
        elif loss_type == 3:
            loss_function = nn.MSELoss().to(device)

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

        # min_val_loss = 1000000
        for epoch in tqdm(range(epochs)):
            train_loss = []
            for (seq, label) in Dtr:
                # seq = seq.to(device)
                label = label.to(device)
                
                if model_type == "NoPatch":
                    y_pred = model(seq)
                else:
                    patch_len = args.patchLen
                    patch_input,length = create_patch(seq,patch_len)
                    y_pred = model(patch_input,length)
                    
                # print(y_pred.shape)
                if loss_type == 2:
                    y_pred = y_pred[:,:,:].contiguous().view(-1,3)
                    label = label.flatten()
                loss = loss_function(y_pred, label)
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
        loss_type = args.loss_type
        model.eval()
        if loss_type == 2:
            quantiles = [0.1, 0.5, 0.9]
            loss_function = QuantileLoss(quantiles).to(device)
        elif loss_type == 1:
            # loss_function = nn.MSELoss().to(device)
            loss_function = nn.SmoothL1Loss().to(device)
            # loss_function = HighdemensionLoss(args.output_size,0).to(device)
        elif loss_type == 3:
            loss_function = nn.MSELoss().to(device)
        val_loss = []
        for (seq, label) in Val:
            with torch.no_grad():
                # seq = seq.to(device)
                label = label.to(device)
                if args.Dataset_type == 2:
                    enc_inputs = seq[0]
                    dec_inputs = seq[1]
                    dec_inputs = dec_inputs.unsqueeze(2)
                    
                    y_pred = model(enc_inputs,dec_inputs)
                    y_pred = model.fc(y_pred)
                    y_pred = y_pred.squeeze(2)
                    y_pred = y_pred[:,:-1]
                else:
                    y_pred = model(seq)
                if loss_type == 2:
                    y_pred = y_pred[:,:,:].contiguous().view(-1,3)
                    label = label.flatten()
                loss = loss_function(y_pred, label)
                val_loss.append(loss.item())

        return np.mean(val_loss),val_loss

    # info = "R=2"
    # R = [2]
    # train("myModel",info)

    # info = "R=1.5,2,2.5"
    # R = [1.5,2,2.5]
    # train("myModel",info)


    # train("TFT",info)
    train("noPatch")