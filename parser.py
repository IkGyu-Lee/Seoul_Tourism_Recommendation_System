import argparse

parser = argparse.ArgumentParser(description='Select Parameters')
parser.add_argument('-shu', '--shuffle', type=int, default=0, help='to stratify dataset split, shuffle df')
parser.add_argument('-m', '--model_name', type=str, default='GMF', help='select among the following model_visitor: [MF, GMF, MLP, NeuMF]')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('-e', '--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('-b', '--batch', type=int, default=512, help='batch size: [256, 512, 1024]')
parser.add_argument('-nf', '--num_factors', type=int, default=24, help='number of predictive factors: [12, 24, 36, 48]')
parser.add_argument('-nl', '--num_layers', type=int, default=3, help='number of hidden layers in NCF Model: [0, 1, 2, 3, 4]')
parser.add_argument('-pr', '--use_pretrain', type=str, default='False', help='use pretrained model_visitor or not for NeuMF')
parser.add_argument('-save', '--save_model', type=str, default='False', help='save trained model_visitor or not')
parser.add_argument('-tar','--target', default= 'v', type = str, help='select v(vistor) or c1(congestion_1) or c2(congestion_2)')
args = parser.parse_args()