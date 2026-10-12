

#!/usr/bin/env python
"""
@Author: Yue Wang (base), gated-DGCNN + ECT complexity injection added
@File: main.py

Training / evaluation for DGCNN + ECT complexity gating.
    --task cls : object classification  (modelnet40 / shapenet)
    --task seg : part segmentation      (shapenetpart)
"""

from __future__ import print_function
import os
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
from tqdm import tqdm

from data_pyg import get_dataset, NUM_CLASSES
# to train the nested-gating model instead, change this one line to:
#   from nested_gating import build_model
from nested_gating import build_model
from util import cal_loss, IOStream


# ---------------------------------------------------------------------------
# ShapeNet-Part constants (16 object categories, 50 global part labels 0..49)
# ---------------------------------------------------------------------------
SEG_NUM_CATEGORIES = 16
SEG_NUM_PARTS = 50
# number of parts per category, in the standard ShapeNet-Part category order
# (airplane, bag, cap, car, chair, earphone, guitar, knife, lamp, laptop,
#  motorbike, mug, pistol, rocket, skateboard, table)
CAT_NUM_PARTS = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
CAT_PART_START = [int(s) for s in np.concatenate([[0], np.cumsum(CAT_NUM_PARTS)[:-1]])]


def str2bool(v):
    # argparse's type=bool treats ANY non-empty string (incl. "False") as True
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def _init_():
    base = 'checkpoints/' + args.exp_name
    os.makedirs(base + '/models', exist_ok=True)
    for f in ['main.py', 'dgcnn_ect_gated_models.py', 'util.py', 'data_pyg.py']:
        if os.path.exists(f):
            shutil.copy(f, base + '/' + f + '.backup')


def get_complexity(args, ect_features, device):
    """
    Returns None (ungated baseline) or a (B, N) complexity tensor on device.
    The dataset already reduces the ECT features to a per-point score in [0, 1].
    """
    if not args.use_gate:
        return None
    return ect_features.to(device).float()


def get_lambda(args, model):
    """
    Gate statistic to log (None if ungated):
      ECT scale gate -> the learnable scalar lambda
      nested gate    -> mean gate value over all points / gated blocks in the
                        last forward pass
    """
    if not args.use_gate:
        return None
    m = model.module
    if hasattr(m, 'gate'):                       # scale gate: read the parameter
        return m.gate.lam.detach().item()
    # nested gate: cached from the last forward (can stay None under
    # multi-GPU DataParallel, where the forward runs on replicas)
    return getattr(m, 'last_lambda', None)


# ---------------------------------------------------------------------------
# batch unpacking
# ---------------------------------------------------------------------------
def unpack_batch(args, batch, device):
    """
    Returns (data, complexity, cat_onehot, target)
      cls: batch = (points, ect, label)
           -> target (B,), cat_onehot None
      seg: batch = (points, ect, cat_label, seg_label)
           -> target (B, N), cat_onehot (B, 16)

    >>> If your shapenetpart dataset returns a different tuple order, this is
    >>> the only place that needs to change.
    """
    if args.task == 'cls':
        data, ect, label = batch
        data = data.to(device).permute(0, 2, 1)           # (B, 3, N)
        label = label.to(device).squeeze()
        return data, get_complexity(args, ect, device), None, label

    data, ect, cat, seg = batch
    data = data.to(device).permute(0, 2, 1)               # (B, 3, N)
    cat = cat.to(device).long().view(-1)                  # (B,)
    seg = seg.to(device).long()                           # (B, N)
    cat_onehot = F.one_hot(cat, SEG_NUM_CATEGORIES).float()
    return data, get_complexity(args, ect, device), (cat_onehot, cat), seg


def forward_model(args, model, data, complexity, cat_info):
    if args.task == 'cls':
        return model(data, complexity=complexity)
    cat_onehot, _ = cat_info
    return model(data, cat_onehot, complexity=complexity)


def compute_loss(args, logits, target):
    if args.task == 'cls':
        return cal_loss(logits, target)
    # logits (B, p, N), target (B, N)
    p = logits.size(1)
    return F.cross_entropy(logits.permute(0, 2, 1).contiguous().view(-1, p),
                           target.view(-1))


# ---------------------------------------------------------------------------
# segmentation metrics (standard ShapeNet-Part protocol)
# ---------------------------------------------------------------------------
def shape_ious(logits, seg, cat):
    """
    logits: (B, p, N) np, seg: (B, N) np, cat: (B,) np
    Predictions are restricted to the parts of each shape's own category.
    Returns per-shape IoU list and the matching category list.
    """
    ious = []
    for i in range(logits.shape[0]):
        c = int(cat[i])
        s, n = CAT_PART_START[c], CAT_NUM_PARTS[c]
        pred = np.argmax(logits[i, s:s + n, :], axis=0) + s
        gt = seg[i]
        part_ious = []
        for part in range(s, s + n):
            inter = np.sum((gt == part) & (pred == part))
            union = np.sum((gt == part) | (pred == part))
            part_ious.append(1.0 if union == 0 else inter / float(union))
        ious.append(np.mean(part_ious))
    return ious, [int(c) for c in cat]


def summarize_seg(point_true, point_pred, all_ious, all_cats):
    acc = metrics.accuracy_score(point_true, point_pred)
    avg_acc = metrics.balanced_accuracy_score(point_true, point_pred)
    all_ious = np.array(all_ious)
    all_cats = np.array(all_cats)
    instance_miou = all_ious.mean()
    cat_ious = [all_ious[all_cats == c].mean() for c in range(SEG_NUM_CATEGORIES)
                if np.any(all_cats == c)]
    class_miou = float(np.mean(cat_ious))
    return acc, avg_acc, instance_miou, class_miou


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def run_epoch(args, model, loader, device, opt=None, desc=''):
    """One pass over `loader`. Trains if opt is given, else evaluates."""
    training = opt is not None
    model.train() if training else model.eval()

    total_loss, count = 0.0, 0.0
    preds_all, true_all = [], []          # cls: per-sample; seg: per-point
    seg_ious, seg_cats = [], []

    bar = tqdm(loader, desc=desc, leave=False, unit='batch', dynamic_ncols=True)
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in bar:
            data, complexity, cat_info, target = unpack_batch(args, batch, device)
            batch_size = data.size(0)

            if training:
                opt.zero_grad()
            logits = forward_model(args, model, data, complexity, cat_info)
            loss = compute_loss(args, logits, target)
            if training:
                loss.backward()
                opt.step()

            total_loss += loss.item() * batch_size
            count += batch_size

            if args.task == 'cls':
                preds = logits.max(dim=1)[1]
                preds_all.append(preds.detach().cpu().numpy())
                true_all.append(target.cpu().numpy())
            else:
                logits_np = logits.detach().cpu().numpy()          # (B, p, N)
                seg_np = target.cpu().numpy()                      # (B, N)
                cat_np = cat_info[1].cpu().numpy()                 # (B,)
                preds = logits.max(dim=1)[1]                       # (B, N)
                preds_all.append(preds.detach().cpu().numpy().reshape(-1))
                true_all.append(seg_np.reshape(-1))
                ious, cats = shape_ious(logits_np, seg_np, cat_np)
                seg_ious += ious
                seg_cats += cats

            postfix = dict(loss=f'{loss.item():.4f}')
            if training:
                postfix['lr'] = f'{opt.param_groups[0]["lr"]:.6f}'
                lam = get_lambda(args, model)
                if lam is not None:
                    postfix['lambda'] = f'{lam:.4f}'
            bar.set_postfix(**postfix)

    true_all = np.concatenate(true_all)
    preds_all = np.concatenate(preds_all)
    stats = dict(loss=total_loss / count)
    if args.task == 'cls':
        stats['acc'] = metrics.accuracy_score(true_all, preds_all)
        stats['avg_acc'] = metrics.balanced_accuracy_score(true_all, preds_all)
        stats['score'] = stats['acc']                     # used for best-model selection
    else:
        acc, avg_acc, inst_miou, cls_miou = summarize_seg(true_all, preds_all,
                                                          seg_ious, seg_cats)
        stats.update(acc=acc, avg_acc=avg_acc, miou=inst_miou, class_miou=cls_miou)
        stats['score'] = inst_miou
    return stats


def fmt(tag, epoch, s, task, lam=None):
    if task == 'cls':
        out = '%s %d, loss: %.6f, %s acc: %.6f, %s avg acc: %.6f' % (
            tag, epoch, s['loss'], tag.lower(), s['acc'], tag.lower(), s['avg_acc'])
    else:
        out = ('%s %d, loss: %.6f, %s acc: %.6f, %s avg acc: %.6f, '
               '%s mIoU (instance): %.6f, %s mIoU (class): %.6f') % (
            tag, epoch, s['loss'], tag.lower(), s['acc'], tag.lower(), s['avg_acc'],
            tag.lower(), s['miou'], tag.lower(), s['class_miou'])
    if lam is not None:
        out += ', lambda: %.6f' % lam
    return out


def make_model(args, device):
    return build_model(args.task, args,
                       num_classes=args.num_classes,
                       num_categories=SEG_NUM_CATEGORIES,
                       gate_all_blocks=args.gate_all_blocks).to(device)


def train(args, io):
    train_loader = DataLoader(get_dataset(args.dataset, 'train', args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(get_dataset(args.dataset, 'test', args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = make_model(args, device)
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    best_score = 0
    epoch_bar = tqdm(range(args.epochs), desc='Training', unit='epoch',
                     leave=False, dynamic_ncols=True)
    for epoch in epoch_bar:
        train_stats = run_epoch(args, model, train_loader, device, opt=opt,
                                desc=f'Epoch {epoch}/{args.epochs} [Train]')
        # scheduler steps AFTER the optimizer (PyTorch >= 1.1 convention)
        scheduler.step()
        io.cprint(fmt('Train', epoch, train_stats, args.task, get_lambda(args, model)))

        test_stats = run_epoch(args, model, test_loader, device, opt=None,
                               desc=f'Epoch {epoch}/{args.epochs} [Test]')
        io.cprint(fmt('Test', epoch, test_stats, args.task))

        if test_stats['score'] >= best_score:
            best_score = test_stats['score']
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)

        key = 'acc' if args.task == 'cls' else 'miou'
        epoch_bar.set_postfix(**{f'train_{key}': f'{train_stats[key]:.4f}',
                                 f'test_{key}': f'{test_stats[key]:.4f}',
                                 'best': f'{best_score:.4f}'})


def test(args, io):
    test_loader = DataLoader(get_dataset(args.dataset, 'test', args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = make_model(args, device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    stats = run_epoch(args, model, test_loader, device, opt=None, desc='Testing')
    if args.task == 'cls':
        io.cprint('Test :: test acc: %.6f, test avg acc: %.6f' % (stats['acc'], stats['avg_acc']))
    else:
        io.cprint('Test :: test acc: %.6f, test avg acc: %.6f, mIoU (instance): %.6f, '
                  'mIoU (class): %.6f' % (stats['acc'], stats['avg_acc'],
                                          stats['miou'], stats['class_miou']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Recognition / Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--task', type=str, default='cls', choices=['cls', 'seg'],
                        help='cls = object classification, seg = part segmentation')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40', 'shapenet', 'shapenetpart'],
                        help='modelnet40 / shapenet (16-category classification) for --task cls; '
                             'shapenetpart for --task seg')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=str2bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=str2bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=str2bool, default=False,
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
    parser.add_argument('--use_gate', type=str2bool, default=True,
                        help='inject ECT complexity score into DGCNN via the learned gate '
                             '(set False to run the plain ungated baseline for comparison)')
    parser.add_argument('--gate_all_blocks', type=str2bool, default=False,
                        help='apply the gate at every EdgeConv block instead of just the first '
                             '(cls: 4 blocks, seg: 3 blocks)')
    parser.add_argument('--complexity_method', type=str, default='entropy',
                        choices=['variance_auc', 'l2_norm', 'max_variance', 'entropy'],
                        help='[unused for now] the dataset already returns a reduced (B, N) score')
    parser.add_argument('--num_thetas', type=int, default=64, metavar='N',
                        help='[unused for now] see --complexity_method')

    args = parser.parse_args()

    # task <-> dataset consistency
    if args.task == 'seg' and args.dataset != 'shapenetpart':
        parser.error("--task seg requires --dataset shapenetpart")
    if args.task == 'cls' and args.dataset == 'shapenetpart':
        parser.error("--dataset shapenetpart is only for --task seg")
    args.num_classes = SEG_NUM_PARTS if args.task == 'seg' else NUM_CLASSES[args.dataset]

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
