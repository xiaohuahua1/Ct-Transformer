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
from myModels import *

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

def transfer_weights(weights_path, model, exclude_head=True):
    # state_dict = model.state_dict()
    new_state_dict = torch.load(weights_path, map_location=device)['models']
    #print('new_state_dict',new_state_dict)
    matched_layers = 0
    unmatched_layers = []

    for name, param in model.state_dict().items():      
        if exclude_head and 'head' in name: continue
        if name in new_state_dict:            
            matched_layers += 1
            input_param = new_state_dict[name]
            if input_param.shape == param.shape: param.copy_(input_param)
            else: unmatched_layers.append(name)
        else:
            unmatched_layers.append(name)
            pass # these are weights that weren't in the original model, such as a new head
    if matched_layers == 0: raise Exception("No shared weight names were found between the models")
    else:
        if len(unmatched_layers) > 0:
            print(f'check unmatched_layers: {unmatched_layers}')
        else:
            print(f"weights from {weights_path} successfully transferred!\n")
    model = model.to(device)
    return model

def freeze(model):
    for param in model.parameters(): 
        param.requires_grad = False 
        
    for param in model.head.parameters(): 
        param.requires_grad = True

def unfreeze(model):
    for param in model.parameters(): 
        param.requires_grad = True 
        

if __name__ == '__main__':

    Inum = 100
    smooth = True
    train_epoch = 0

    fold_data = "../data"
    fold_model = "model/ER"
    fold_model_out = "model/SF"

    net = ["SF"]
    d = [10]
    R = [1.5,2,2.5,3]


    def finetune(fine_type,info):
        args = CtTransformer_pretrain_args_parser()
        path = []
        Dtr = []
        Dva = []

        max_list,min_list,max_rt,min_rt = getBound(args,fold_data,net,d,R,Inum,1)
        path = fold_model + "/myModel_pretrain_" + info + ".pkl"
        Dtr,Dva,Dtr_all,Dva_all = get_Train_range(args,fold_data,net,d,R,Inum,smooth,max_list,min_list,max_rt,min_rt)
        model = CtTransformer(args,head_type="prediction").to(device)
        model = transfer_weights(path,model)
        freeze(model)
        finetuneModel(args,model,Dtr,Dva,fold_model_out,fine_type,info)

    def finetuneModel(args,model,Dtr,Val,fold_model,fine_type,info):
        path = fold_model + "/myModel_" + fine_type + "_" + info + ".pkl"
        optimizer = args.optimizer
        gamma = args.gamma
        patch_len = args.patchLen
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


        if fine_type == "end-to-end":
            weight_decay = args.endtoend_weight_decay
            lr = args.endtoend_lr
            step_size = args.endtoend_step_size
            epochs = args.endtoend_epochs
            freeze_epoch = 5
        elif fine_type == "Linear":
            weight_decay = args.Linear_weight_decay
            lr = args.Linear_lr
            step_size = args.Linear_step_size
            epochs = args.Linear_epochs

        if optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=0.9, weight_decay=weight_decay)
        
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        best_model = None
        model.train()
        train_loss_all = []
        train_loss_mean = []
        val_loss_all = []
        val_loss_mean = []

        if fine_type == "end-to-end":
            for epoch in tqdm(range(freeze_epoch)):
                train_loss = []
                for (seq, label) in Dtr:
                    label = label.to(device)
                    patch_input,length = create_patch(seq,patch_len)
                    y_pred = model(patch_input,length)
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
                best_model = early_stopping(val_loss_m, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                print("end-to-end:linear")
                print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss_m))
                model.train()

            scheduler1 = StepLR(optimizer, step_size=step_size, gamma=gamma)
            for epoch in tqdm(range(epochs)):
                unfreeze(model)
                train_loss = []
                for (seq, label) in Dtr:
                    label = label.to(device)
                    patch_input,length = create_patch(seq,patch_len)
                    y_pred = model(patch_input,length)
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
                scheduler1.step()

                # validation
                val_loss_m,val_loss = get_val_loss(args,model, Val)
                val_loss_all.extend(val_loss)
                val_loss_mean.append(val_loss_m)
                best_model = early_stopping(val_loss_m, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                print("end-to-end:finetune")
                print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss_m))
                model.train()
            train_epoch = epoch + freeze_epoch
            

        elif fine_type == "Linear":
            for epoch in tqdm(range(epochs)):
                train_loss = []
                for (seq, label) in Dtr:
                    label = label.to(device)
                    patch_input,length = create_patch(seq,patch_len)
                    y_pred = model(patch_input,length)
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
                best_model = early_stopping(val_loss_m, model)
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
        patch_len = args.patchLen
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
                patch_input,length = create_patch(seq,patch_len)
                y_pred = model(patch_input,length)
                if loss_type == 2:
                    y_pred = y_pred[:,:,:].contiguous().view(-1,3)
                    label = label.flatten()
                loss = loss_function(y_pred, label)
                val_loss.append(loss.item())

        return np.mean(val_loss),val_loss


    fine_type = "end-to-end"
    info = "ER"
    finetune(fine_type,info)

    # fold_model = "model\\ER"
    # model_path = fold_model + "\\myModel_pretrain_ .pkl"
    # args = CtTransformer_args_parser()
    # model = CtTransformer(args,head_type="prediction").to(device)
    # # model.load_state_dict(torch.load(model_path)['models'])
    # model = transfer_weights(model_path,model)
    # freeze(model)
    


