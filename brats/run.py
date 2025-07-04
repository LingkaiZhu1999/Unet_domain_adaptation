from simCLR import SimCLR
import argparse
import torch
import os
import torch.backends.cudnn as cudnn

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='', help='model name: (default: arch+timestamp')
    parser.add_argument('--domain_source', default="BraTS19_CBICA_HGG",
                        help='source dataset name') # source
    parser.add_argument('--domain_target', default="BraTS17_TCIA_HGG",
                        help='target dataset name') # target 
    parser.add_argument('--input_channel', default=4, type=int, help='input channels')
    parser.add_argument('--output_channel', default=3, type=int, help='input channels')
    parser.add_argument('--loss', default='BCEDiceLoss')
    parser.add_argument('--epochs', default=800, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=50, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--warm_up', default=10, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False,
                        help='nesterov')
    parser.add_argument('--use_cuda', default=True)
    parser.add_argument('--temperature', default=0.5, type=float)
    parser.add_argument('--validate_frequency', default=1, type=int)
    parser.add_argument('--seed', type=int, default=2) # seed

    parser.add_argument('--lam', default=1000, type=int) # lambda 

    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--mode', default='comb')
    parser.add_argument('--contrastive_mode', default='within_domain') # con mode
    args = parser.parse_args()
    args.name = f'{args.domain_source}_adapt_{args.domain_target}_lambda_{args.lam}_batchsize_{args.batch_size}_{args.contrastive_mode}_{args.mode}_Cch_seed_{args.seed}_nodropout'
    return args

def main():
    args = parse_args()
    torch.cuda.manual_seed_all(args.seed)
    if not os.path.exists(f'../output/{args.name}'):
        os.mkdir(f'../output/{args.name}')
    if not os.path.exists(f'../models/{args.name}'):
        os.mkdir(f'../models/{args.name}')
    with open('../models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)
    cudnn.benchmark = True
    simclr = SimCLR(args)
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')
    if args.contrastive_mode == 'only_source_domain':
        simclr.joint_train_on_source()
    else:
        simclr.joint_train_on_source_and_target()


if __name__ == "__main__":
    main()