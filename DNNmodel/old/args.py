import argparse
import torch

GROUP = 6
BATCH_SIZE=32
num_workers = 6

def seq2seq_CtRt_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--readDate_type', type=int, default=2, help='input dimension')
    parser.add_argument('--Dataset_type', type=int, default=2, help='input dimension')
    parser.add_argument('--group', type=int, default=GROUP, help='input dimension')
    parser.add_argument('--epochs', type=int, default=50, help='input dimension')
    parser.add_argument('--input_size', type=int, default=GROUP+1, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=5, help='seq len')
    parser.add_argument('--output_size', type=int, default=2, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--workers', type=int, default=num_workers, help='max')

    args = parser.parse_args()

    return args

def seq2seq_Ct_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--readDate_type', type=int, default=2, help='input dimension')
    parser.add_argument('--Dataset_type', type=int, default=4, help='input dimension')
    parser.add_argument('--group', type=int, default=GROUP, help='input dimension')
    parser.add_argument('--epochs', type=int, default=50, help='input dimension')
    parser.add_argument('--input_size', type=int, default=GROUP, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=7, help='seq len')
    parser.add_argument('--output_size', type=int, default=3, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=25, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--workers', type=int, default=num_workers, help='max')

    args = parser.parse_args()

    return args


def seq2seq_Rt_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--readDate_type', type=int, default=1, help='input dimension')
    parser.add_argument('--Dataset_type', type=int, default=1, help='input dimension')
    parser.add_argument('--group', type=int, default=GROUP, help='input dimension')
    parser.add_argument('--epochs', type=int, default=50, help='input dimension')
    parser.add_argument('--input_size', type=int, default=1, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=7, help='seq len')
    parser.add_argument('--output_size', type=int, default=3, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=25, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--workers', type=int, default=num_workers, help='max')

    args = parser.parse_args()

    return args



def ann_CtRt_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--readDate_type', type=int, default=2, help='input dimension')
    parser.add_argument('--Dataset_type', type=int, default=2, help='input dimension')
    parser.add_argument('--group', type=int, default=GROUP, help='input dimension')
    parser.add_argument('--epochs', type=int, default=50, help='input dimension')
    parser.add_argument('--input_size', type=int, default=GROUP+1, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=7, help='seq len')
    parser.add_argument('--output_size', type=int, default=3, help='output dimension')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=25, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--workers', type=int, default=num_workers, help='max')

    args = parser.parse_args()

    return args

def ann_Ct_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--readDate_type', type=int, default=2, help='input dimension')
    parser.add_argument('--Dataset_type', type=int, default=4, help='input dimension')
    parser.add_argument('--group', type=int, default=GROUP, help='input dimension')
    parser.add_argument('--epochs', type=int, default=50, help='input dimension')
    parser.add_argument('--input_size', type=int, default=GROUP, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=7, help='seq len')
    parser.add_argument('--output_size', type=int, default=3, help='output dimension')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=25, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--workers', type=int, default=num_workers, help='max')

    args = parser.parse_args()

    return args


def ann_CR_pair_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--readDate_type', type=int, default=2, help='input dimension')
    parser.add_argument('--Dataset_type', type=int, default=3, help='input dimension')
    parser.add_argument('--group', type=int, default=GROUP, help='input dimension')
    parser.add_argument('--epochs', type=int, default=50, help='input dimension')
    parser.add_argument('--input_size', type=int, default=GROUP, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=1, help='seq len')
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=25, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--workers', type=int, default=num_workers, help='max')

    args = parser.parse_args()

    return args

def cnn_CtRt_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--readDate_type', type=int, default=2, help='input dimension')
    parser.add_argument('--Dataset_type', type=int, default=2, help='input dimension')
    parser.add_argument('--group', type=int, default=GROUP, help='input dimension')
    parser.add_argument('--epochs', type=int, default=25, help='input dimension')
    parser.add_argument('--in_channels', type=int, default=GROUP+1, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=10, help='seq len')
    parser.add_argument('--output_size', type=int, default=3, help='output dimension')
    parser.add_argument('--kernel_size', type=int, default=2, help='kernel_size')
    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=25, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--workers', type=int, default=num_workers, help='max')

    args = parser.parse_args()

    return args


def cnn_CR_pair_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--readDate_type', type=int, default=2, help='input dimension')
    parser.add_argument('--Dataset_type', type=int, default=3, help='input dimension')
    parser.add_argument('--group', type=int, default=GROUP, help='input dimension')
    parser.add_argument('--epochs', type=int, default=30, help='input dimension')
    parser.add_argument('--in_channels', type=int, default=GROUP, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=1, help='seq len')
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    parser.add_argument('--kernel_size', type=int, default=1, help='kernel_size')
    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=25, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--workers', type=int, default=num_workers, help='max')

    args = parser.parse_args()

    return args


def cnn_lstm_CtRt_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--readDate_type', type=int, default=2, help='input dimension')
    parser.add_argument('--Dataset_type', type=int, default=2, help='input dimension')
    parser.add_argument('--group', type=int, default=GROUP, help='input dimension')
    parser.add_argument('--epochs', type=int, default=50, help='input dimension')
    parser.add_argument('--in_channels', type=int, default=GROUP+1, help='input dimension')
    parser.add_argument('--out_channels', type=int, default=64, help='output dimension')
    parser.add_argument('--kernel_size', type=int, default=3, help='kernel_size')
    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--seq_len', type=int, default=9, help='seq len')
    parser.add_argument('--output_size', type=int, default=3, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=25, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--workers', type=int, default=num_workers, help='max')

    args = parser.parse_args()

    return args


def transformer_Rt_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--readDate_type', type=int, default=1, help='input dimension')
    parser.add_argument('--Dataset_type', type=int, default=5, help='input dimension')
    parser.add_argument('--epochs', type=int, default=50, help='input dimension')
    parser.add_argument('--group', type=int, default=GROUP, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=5, help='seq len')
    parser.add_argument('--input_size', type=int, default=1, help='input dimension')
    parser.add_argument('--d_model', type=int, default=32, help='input dimension')
    parser.add_argument('--output_size', type=int, default=2, help='output dimension')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--workers', type=int, default=num_workers, help='max')

    args = parser.parse_args()

    return args


def transformer_RtCt_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--readDate_type', type=int, default=3, help='input dimension')
    parser.add_argument('--Dataset_type', type=int, default=6, help='input dimension')
    parser.add_argument('--epochs', type=int, default=10, help='input dimension')
    parser.add_argument('--group', type=int, default=GROUP, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=5, help='seq len')
    parser.add_argument('--input_size', type=int, default=1, help='input dimension')
    parser.add_argument('--d_model', type=int, default=32, help='input dimension')
    parser.add_argument('--output_size', type=int, default=2, help='output dimension')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--workers', type=int, default=num_workers, help='max')

    args = parser.parse_args()

    return args

def transformer_Ct_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--readDate_type', type=int, default=2, help='input dimension')
    parser.add_argument('--Dataset_type', type=int, default=7, help='input dimension')
    parser.add_argument('--epochs', type=int, default=10, help='input dimension')
    parser.add_argument('--group', type=int, default=GROUP, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=7, help='seq len')
    parser.add_argument('--input_size', type=int, default=2, help='input dimension')
    parser.add_argument('--d_model', type=int, default=128, help='input dimension')
    parser.add_argument('--output_size', type=int, default=7, help='output dimension')
    parser.add_argument('--lr', type=float, default=0.000005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=4, help='step size')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--workers', type=int, default=num_workers, help='max')

    args = parser.parse_args()

    return args


def transformer_Ct_d12_7_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--readDate_type', type=int, default=2, help='input dimension')
    parser.add_argument('--Dataset_type', type=int, default=7, help='input dimension')
    parser.add_argument('--epochs', type=int, default=10, help='input dimension')
    parser.add_argument('--group', type=int, default=12, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=7, help='seq len')
    parser.add_argument('--input_size', type=int, default=2, help='input dimension')
    parser.add_argument('--d_model', type=int, default=128, help='input dimension')
    parser.add_argument('--output_size', type=int, default=7, help='output dimension')
    parser.add_argument('--lr', type=float, default=0.000005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=4, help='step size')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--workers', type=int, default=num_workers, help='max')

    args = parser.parse_args()

    return args


def transformer_new_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--readDate_type', type=int, default=2, help='input dimension')
    parser.add_argument('--Dataset_type', type=int, default=7, help='input dimension')
    parser.add_argument('--epochs', type=int, default=10, help='input dimension')
    parser.add_argument('--group', type=int, default=12, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=7, help='seq len')
    parser.add_argument('--input_size', type=int, default=2, help='input dimension')
    parser.add_argument('--d_model', type=int, default=128, help='input dimension')
    parser.add_argument('--output_size', type=int, default=7, help='output dimension')
    parser.add_argument('--d_k', type=int, default=16, help='output dimension')
    parser.add_argument('--d_v', type=int, default=16, help='output dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='output dimension')
    parser.add_argument('--d_ff', type=int, default=2048, help='output dimension')
    parser.add_argument('--n_layers', type=int, default=3, help='output dimension')


    parser.add_argument('--lr', type=float, default=0.000005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=4, help='step size')
    parser.add_argument('--pred_step_size', type=int, default=12, help='pred step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--workers', type=int, default=num_workers, help='max')

    args = parser.parse_args()

    return args

