import argparse
from lib2to3.pgen2.token import GREATER
import torch

GROUP = 8
BATCH_SIZE=64
INPUT_SIZE = GROUP + 1
num_workers = 1


def seq2seq_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--readDate_type', type=int, default=1, help='input dimension')
    parser.add_argument('--Dataset_type', type=int, default=1, help='input dimension')
    parser.add_argument('--group', type=int, default=GROUP, help='input dimension')
    parser.add_argument('--epochs', type=int, default=24, help='input dimension')
    parser.add_argument('--input_size', type=int, default=6, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=7, help='seq len')
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=3, help='num layers')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=4, help='step size')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--workers', type=int, default=num_workers, help='max')

    args = parser.parse_args()

    return args


def transformer_encoder_args_parser():
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--id', type=int, default=1, help='input dimension')
    parser.add_argument('--readDate_type', type=int, default=1, help='input dimension')
    parser.add_argument('--Dataset_type', type=int, default=1, help='input dimension')
    parser.add_argument('--loss_type', type=int, default=2, help='input dimension')
    
    
    parser.add_argument('--group', type=int, default=GROUP, help='input dimension')
    parser.add_argument('--input_size', type=int, default=INPUT_SIZE, help='input dimension')
    parser.add_argument('--output_size', type=int, default=3, help='input dimension')
    parser.add_argument('--d_model', type=int, default=512, help='input dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='output dimension')
    parser.add_argument('--d_ff', type=int, default=2048, help='output dimension')
    parser.add_argument('--n_layers', type=int, default=3, help='output dimension')
    parser.add_argument('--num_encoder_steps', type=int, default=8, help='num layers')
    
    parser.add_argument('--epochs', type=int, default=20, help='input dimension')
    parser.add_argument('--patience', type=int, default=7, help='input dimension')
    parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--step_size', type=int, default=4, help='step size')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--workers', type=int, default=num_workers, help='max')

    args = parser.parse_args()

    return args

def TFT_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=int, default=2, help='input dimension')
    parser.add_argument('--readDate_type', type=int, default=1, help='input dimension')
    parser.add_argument('--Dataset_type', type=int, default=3, help='input dimension')
    parser.add_argument('--loss_type', type=int, default=2, help='input dimension')


    parser.add_argument('--group', type=int, default=GROUP, help='input dimension')
    parser.add_argument('--input_size', type=int, default=INPUT_SIZE, help='input dimension')
    parser.add_argument('--output_size', type=int, default=3, help='output dimension')
    parser.add_argument('--hidden_layer_size', type=int, default=256, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--num_heads', type=int, default=8, help='num layers')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='num layers')
    parser.add_argument('--num_encoder_steps', type=int, default=8, help='num layers')

    parser.add_argument('--epochs', type=int, default=20, help='input dimension')
    parser.add_argument('--patience', type=int, default=7, help='input dimension')
    parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--step_size', type=int, default=4, help='step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--workers', type=int, default=num_workers, help='max')

    args = parser.parse_args()

    return args


def CtTransformer_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=int, default=3, help='input dimension')
    parser.add_argument('--readDate_type', type=int, default=1, help='input dimension')
    parser.add_argument('--Dataset_type', type=int, default=3, help='input dimension')
    parser.add_argument('--loss_type', type=int, default=2, help='input dimension')
       
    parser.add_argument('--group', type=int, default=GROUP, help='input dimension')
      
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='num layers')
    parser.add_argument('--input_size', type=int, default=INPUT_SIZE, help='input dimension')
    parser.add_argument('--output_size', type=int, default=3, help='input dimension')
    # parser.add_argument('--d_model', type=int, default=256, help='input dimension')
    parser.add_argument('--hidden_layer_size', type=int, default=512, help='input dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='output dimension')
    parser.add_argument('--d_ff', type=int, default=2048, help='output dimension')
    parser.add_argument('--n_layers', type=int, default=3, help='output dimension')
    parser.add_argument('--num_encoder_steps', type=int, default=8, help='num layers')
    parser.add_argument('--patchLen', type=int, default=4, help='input dimension')
    
    parser.add_argument('--epochs', type=int, default=20, help='input dimension')
    parser.add_argument('--patience', type=int, default=7, help='input dimension')
    parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--step_size', type=int, default=4, help='step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--workers', type=int, default=num_workers, help='max')

    args = parser.parse_args()

    return args

def CtTransformer_pretrain_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=int, default=3, help='input dimension')
    parser.add_argument('--readDate_type', type=int, default=1, help='input dimension')
    parser.add_argument('--Dataset_type', type=int, default=3, help='input dimension')
    parser.add_argument('--loss_type', type=int, default=2, help='input dimension')
       
    parser.add_argument('--group', type=int, default=GROUP, help='input dimension')
      
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='num layers')
    parser.add_argument('--input_size', type=int, default=GROUP + 1, help='input dimension')
    parser.add_argument('--output_size', type=int, default=3, help='input dimension')
    # parser.add_argument('--d_model', type=int, default=256, help='input dimension')
    parser.add_argument('--hidden_layer_size', type=int, default=512, help='input dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='output dimension')
    parser.add_argument('--d_ff', type=int, default=2048, help='output dimension')
    parser.add_argument('--n_layers', type=int, default=3, help='output dimension')
    parser.add_argument('--num_encoder_steps', type=int, default=8, help='num layers')
    parser.add_argument('--patchLen', type=int, default=4, help='input dimension')
    
    
    parser.add_argument('--patience', type=int, default=7, help='input dimension')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--workers', type=int, default=num_workers, help='max')


    parser.add_argument('--pretrain_epochs', type=int, default=20, help='input dimension')
    parser.add_argument('--pretrain_lr', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--pretrain_weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--pretrain_step_size', type=int, default=4, help='step size')

    parser.add_argument('--endtoend_epochs', type=int, default=20, help='input dimension')
    parser.add_argument('--endtoend_lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--endtoend_weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--endtoend_step_size', type=int, default=4, help='step size')

    parser.add_argument('--Linear_epochs', type=int, default=20, help='input dimension')
    parser.add_argument('--Linear_lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--Linear_weight_decay', type=float, default=0.00009, help='weight decay')
    parser.add_argument('--Linear_step_size', type=int, default=10, help='step size')

    args = parser.parse_args()

    return args

def Ablation_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=int, default=3, help='input dimension')
    parser.add_argument('--readDate_type', type=int, default=1, help='input dimension')
    parser.add_argument('--Dataset_type', type=int, default=3, help='input dimension')
    parser.add_argument('--loss_type', type=int, default=2, help='input dimension')
       
    parser.add_argument('--group', type=int, default=GROUP, help='input dimension')
      
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='num layers')
    parser.add_argument('--input_size', type=int, default=INPUT_SIZE, help='input dimension')
    parser.add_argument('--output_size', type=int, default=3, help='input dimension')
    # parser.add_argument('--d_model', type=int, default=256, help='input dimension')
    parser.add_argument('--hidden_layer_size', type=int, default=512, help='input dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='output dimension')
    parser.add_argument('--d_ff', type=int, default=2048, help='output dimension')
    parser.add_argument('--n_layers', type=int, default=3, help='output dimension')
    parser.add_argument('--num_encoder_steps', type=int, default=8, help='num layers')
    parser.add_argument('--patchLen', type=int, default=4, help='input dimension')
    
    parser.add_argument('--epochs', type=int, default=20, help='input dimension')
    parser.add_argument('--patience', type=int, default=7, help='input dimension')
    parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--step_size', type=int, default=4, help='step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--workers', type=int, default=num_workers, help='max')

    args = parser.parse_args()

    return args