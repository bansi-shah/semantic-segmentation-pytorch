# System libs
import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
# Our libs
from config import cfg
from dataset import TrainDataset
from models import ModelBuilder, SegmentationModule
from utils import AverageMeter, parse_devices, setup_logger, load_colordict
from lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback

# train one epoch
def train(segmentation_module, iterator, optimizers, history, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss1 = AverageMeter()
    ave_total_loss2 = AverageMeter()
    ave_acc1 = AverageMeter()
    ave_prec1 = AverageMeter()
    ave_rec1 = AverageMeter()
    ave_f11 = AverageMeter()
    ave_iou1 = AverageMeter()
    ave_total_loss2 = AverageMeter()
    ave_acc2 = AverageMeter()
    ave_prec2 = AverageMeter()
    ave_rec2 = AverageMeter()
    ave_f12 = AverageMeter()
    ave_iou2 = AverageMeter()

    segmentation_module.train(not cfg.TRAIN.fix_bn)

    # main loop
    tic = time.time()
    for i in range(cfg.TRAIN.epoch_iters):
        # load a batch of data
        batch_data = next(iterator)
        data_time.update(time.time() - tic)
        segmentation_module.zero_grad()

        # adjust learning rate
        cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, cfg)

        # forward pass
        loss, acc, prec, rec, f1, iou = segmentation_module(batch_data)
        loss1, loss2 = loss[0].mean(), loss[1].mean()
        acc1, acc2 = acc[0].mean(), acc[1].mean()
        prec1, prec2 = prec[0].mean(), prec[1].mean()
        rec1, rec2 = rec[0].mean(), rec[1].mean()
        f11, f12 = f1[0].mean(), f1[1].mean()
        iou1, iou2 = iou[0].mean(), iou[1].mean()       
 
        # Backward
        loss[0].backward(retain_graph=True)
        loss[1].backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss1.update(loss1.data.item())
        ave_acc1.update(acc1.data.item()*100)
        ave_prec1.update(prec1*100)
        ave_rec1.update(rec1*100)
        ave_f11.update(f11*100)
        ave_iou1.update(iou1*100)
        ave_total_loss2.update(loss2.data.item())
        ave_acc2.update(acc1.data.item()*100)
        ave_prec2.update(prec2*100)
        ave_rec2.update(rec2*100)
        ave_f12.update(f12*100)
        ave_iou2.update(iou2*100)

        # calculate accuracy, and display
        if i % cfg.TRAIN.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f},\n Segmentation Task '
                  'Accuracy: {:4.2f}, Loss: {:.6f}, Precision :{:4.2f}, '
                  'Recall: {:4.2f}, F1:{:4.2f}, IoU :{:4.2f} \n Saliency Task '
                  'Accuracy: {:4.2f}, Loss: {:.6f}, Precision :{:4.2f}, '
                  'Recall: {:4.2f}, F1:{:4.2f}, IoU :{:4.2f}'
                  .format(epoch, i, cfg.TRAIN.epoch_iters,
                          batch_time.average(), data_time.average(),
                          cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                          ave_acc1.average(), ave_total_loss1.average(), ave_prec1.average(),
                          ave_rec1.average(), ave_f11.average(), ave_iou1.average(),
                          ave_acc2.average(), ave_total_loss2.average(), ave_prec2.average(),
                          ave_rec2.average(), ave_f12.average(), ave_iou2.average()))

            fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss1'].append(loss1.data.item())
            history['train']['acc1'].append(acc1.data.item())
            history['train']['precision1'].append(prec1)
            history['train']['recall1'].append(rec1)
            history['train']['f11'].append(f11)
            history['train']['iou1'].append(iou1)
            history['train']['loss2'].append(loss2.data.item())
            history['train']['acc2'].append(acc2.data.item())
            history['train']['precision2'].append(prec2)
            history['train']['recall2'].append(rec2)
            history['train']['f12'].append(f12)
            history['train']['iou2'].append(iou2)

def checkpoint(nets, history, cfg, epoch):
    print('Saving checkpoints...')
    (net_encoder, net_decoder, crit) = nets

    dict_encoder = net_encoder.state_dict()
    dict_decoder1 = net_decoder[0].state_dict()
    dict_decoder2 = net_decoder[1].state_dict()

    torch.save(
        history,
        '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_encoder,
        '{}/encoder_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_decoder1,
        '{}/decoder1_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_decoder2,
        '{}/decoder2_epoch_{}.pth'.format(cfg.DIR, epoch))

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

def create_optimizers(nets, cfg):
    (net_encoder, net_decoder, crit) = nets
    optimizer_encoder = torch.optim.SGD(
        group_weight(net_encoder),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_decoder1 = torch.optim.SGD(
        group_weight(net_decoder[0]),
        lr=cfg.TRAIN.lr_decoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_decoder2 = torch.optim.SGD(
        group_weight(net_decoder[1]),
        lr=cfg.TRAIN.lr_decoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    return (optimizer_encoder, optimizer_decoder1, optimizer_decoder2)

def adjust_learning_rate(optimizers, cur_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

    (optimizer_encoder, optimizer_decoder1, optimizer_decoder2) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_decoder1.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder
    for param_group in optimizer_decoder2.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder

def main(cfg, gpus):
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder)

    crit = nn.NLLLoss()
    colors = load_colordict('../AD/data/camvid_all/CamVid/class_dict.csv')

    segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_train = TrainDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_train,
        cfg.DATASET,
        colors = colors,
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.TRAIN.batch_size_per_gpu,  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)
    print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))

    # create loader iterator
    iterator_train = iter(loader_train)

    # load nets into gpu
    if False : #len(gpus) > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=gpus)
        # For sync bn
        patch_replication_callback(segmentation_module)
    #segmentation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit)
    optimizers = create_optimizers(nets, cfg)

    # Main loop
    history = {'train': {'epoch': [], 'loss1': [], 'acc1': [], 'precision1': [], 'recall1':[], 'f11': [], 'iou1': [], 'acc2':[], 'precision2':[], 'recall2' : [], 'f12': [], 'iou2': [], 'loss2' : []}}

    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        train(segmentation_module, iterator_train, optimizers, history, epoch+1, cfg)
         
        # checkpointing
        if epoch % 5 == 0:
            checkpoint(nets, history, cfg, epoch+1)

    print('Training Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/resnet50-upernet.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # Start from checkpoint
    if cfg.TRAIN.start_epoch > 0:
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        cfg.MODEL.weights_decoder = [os.path.join(
            cfg.DIR, 'decoder1_epoch_{}.pth'.format(cfg.TRAIN.start_epoch)),  os.path.join(
            cfg.DIR, 'decoder2_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))]
        assert os.path.exists(cfg.MODEL.weights_encoder) and os.path.exists(cfg.MODEL.weights_decoder[0]) and os.path.exists(cfg.MODEL.weights_decoder[1]) , "checkpoint does not exist!"

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)
    cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu

    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)

    main(cfg, gpus)
