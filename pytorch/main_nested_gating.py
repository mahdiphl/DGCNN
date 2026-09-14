

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_pyg import ModelNet40PyG
from nested_gating import DGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from tqdm import tqdm


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp dgcnn_ect_gated.py checkpoints' + '/' + args.exp_name + '/' + 'dgcnn_ect_gated.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data_pyg.py checkpoints' + '/' + args.exp_name + '/' + 'data_pyg.py.backup')


def get_complexity(args, ect_features, device):
    """
    Returns None (ungated baseline) or a (B, N) complexity tensor on device.

    ModelNet40PyG already reduces the raw ECT features down to a per-point
    complexity score before returning them (confirmed: ect_features comes out
    as (B, N), not (B, N, num_thetas**2)) -- so no further reduction is
    needed here, just move it to device and use it directly as g_i.
    """
    if not args.use_gate:
        return None
    return ect_features.to(device).float()


def train(args, io):
    train_loader = DataLoader(ModelNet40PyG(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40PyG(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args, output_channels=40, gate_all_blocks=args.gate_all_blocks).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    criterion = cal_loss

    best_test_acc = 0
    epoch_bar = tqdm(range(args.epochs), desc='Training', unit='epoch',
                      leave=False, dynamic_ncols=True)
    for epoch in epoch_bar:
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [Train]',
                          leave=False, unit='batch', dynamic_ncols=True)
        for data, ect_features, label in train_bar:
            data = data.to(device)
            data = data.permute(0, 2, 1)
            label = label.to(device).squeeze()
            batch_size = data.size()[0]

            complexity = get_complexity(args, ect_features, device)

            opt.zero_grad()
            logits = model(data, complexity=complexity) if args.model == 'dgcnn' else model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

            batch_acc = metrics.accuracy_score(label.cpu().numpy(), preds.detach().cpu().numpy())
            postfix = dict(loss=f'{loss.item():.4f}', acc=f'{batch_acc:.4f}',
                            lr=f'{opt.param_groups[0]["lr"]:.6f}')
            if args.use_gate and args.model == 'dgcnn':
                # DataParallel wraps the model -> access the underlying module
                lam = model.module.last_lambda
                if lam is not None:
                    postfix['lambda'] = f'{lam:.4f}'
            train_bar.set_postfix(**postfix)

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        train_avg_acc = metrics.balanced_accuracy_score(train_true, train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 train_acc,
                                                                                 train_avg_acc)
        if args.use_gate and args.model == 'dgcnn' and model.module.last_lambda is not None:
            outstr += ', lambda: %.6f' % model.module.last_lambda
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        test_bar = tqdm(test_loader, desc=f'Epoch {epoch}/{args.epochs} [Test]',
                         leave=False, unit='batch', dynamic_ncols=True)
        with torch.no_grad():
            for data, ect_features, label in test_bar:
                data = data.to(device)
                data = data.permute(0, 2, 1)
                label = label.to(device).squeeze()
                batch_size = data.size()[0]

                complexity = get_complexity(args, ect_features, device)

                logits = model(data, complexity=complexity) if args.model == 'dgcnn' else model(data)
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())

                test_bar.set_postfix(loss=f'{loss.item():.4f}')
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)

        epoch_bar.set_postfix(train_acc=f'{train_acc:.4f}',
                               test_acc=f'{test_acc:.4f}',
                               best=f'{best_test_acc:.4f}')


def test(args, io):
    test_loader = DataLoader(ModelNet40PyG(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.model == 'pointnet':
        model = PointNet(args, output_channels=40).to(device)
    else:
        model = DGCNN(args, output_channels=40, gate_all_blocks=args.gate_all_blocks).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_true = []
    test_pred = []
    test_bar = tqdm(test_loader, desc='Testing', unit='batch')
    with torch.no_grad():
        for data, ect_features, label in test_bar:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)

            complexity = get_complexity(args, ect_features, device)

            logits = model(data, complexity=complexity) if args.model == 'dgcnn' else model(data)
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')

    # --- ECT complexity gating options ---
    parser.add_argument('--use_gate', type=bool, default=True,
                        help='inject ECT complexity score into DGCNN via the nested gate '
                             '(set False to run the plain ungated baseline for comparison)')
    parser.add_argument('--gate_all_blocks', type=bool, default=False,
                        help='apply the gate at all 4 EdgeConv blocks instead of just the first')
    parser.add_argument('--complexity_method', type=str, default='entropy',
                        choices=['variance_auc', 'l2_norm', 'max_variance', 'entropy'],
                        help='[unused for now] ModelNet40PyG already returns a reduced (B, N) '
                             'complexity score, so no reduction happens in main.py currently. '
                             'Kept here in case you later move the reduction out of the dataset.')
    parser.add_argument('--num_thetas', type=int, default=64, metavar='N',
                        help='[unused for now] see --complexity_method')

    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
