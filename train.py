import argparse
import json
import logging
import os
import pdb
from shutil import copyfile

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models.baseModel import baseModel
from util.dataloader import relationDataset

args = argparse.ArgumentParser()
args.add_argument('--input', required=True, help='path to directory that contains the input files')
args.add_argument('--output', required=True, help='path to directory that output files will be stored')
args.add_argument('--hidden1', default=1024, type=int, help='the dimension of first hidden layer of MLP')
args.add_argument('--hidden2', default=1024, type=int, help='the dimension of first hidden layer of MLP')
args.add_argument('--batch_size', default=256, type=int, help='batch size')
args.add_argument('--emb_size', default=100, type=int, help='the dimension of entity and relation vectors')
args.add_argument('--lr', default=5e-3, type=float, help='learning rate')
args.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
args.add_argument('--val_step', type=int, default=1, help='the model will run on validation set every val_stop epoches')
args.add_argument('--early_stop_patience', type=int, default=5, help='if the model does not perform as well as \
                    the previous best performance on the validation set for 5 consecutive epoches, the training process will stop')
args.add_argument('--gpu', default='0', help='ID of the GPU that you want to assign')
args.add_argument('-ent_pretrain', action='store_true', help='if you add this option,\
                    it will use pretrained entity embedding named "entity2vec.vec" in input directory.')
args.add_argument('-rel_pretrain', action='store_true', help='if you add this option,\
                    it will use pretrained relation embedding named "relation2vec.vec" in input directory.')
args.add_argument('-load', action='store_true', help='if you add this option, \
                    it will automatically load the stored model and resume training')
args = vars(args.parse_args())

os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : %(message)s')

if not os.path.exists(args['output']):
    os.mkdir(args['output'])

writer = SummaryWriter(args['output'])
writer.add_text('learning rate', str(args['lr']))
writer.add_text('batch_size', str(args['batch_size']))
copyfile('./train.py', os.path.join(args['output'], 'train.py'))
copyfile('models/baseModel.py', os.path.join(args['output'], 'baseModel.py'))

logging.info("Train dataset:")
train_dataset = relationDataset(os.path.join(args['input'], 'train2id.txt'),\
     os.path.join(args['input'], 'entity2id.txt'), os.path.join(args['input'], 'relation2id.txt'))
logging.info("Validation datset:")
val_dataset = relationDataset(os.path.join(args['input'], 'valid2id.txt'),\
     os.path.join(args['input'], 'entity2id.txt'), os.path.join(args['input'], 'relation2id.txt'))
train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

info = args
info['emb_size'] = args['emb_size']
info['tot_rel'] = train_dataset.relation_num
info['tot_ent'] = train_dataset.entity_num
info['weight_decay'] = args['weight_decay']
info['epoch'] = 0
info['batch_size'] = args['batch_size']
info['lr'] = args['lr']
info['best_epoch'] = 0

model = baseModel(args, info['tot_ent'], info['tot_rel']).cuda()
with open(os.path.join(args['output'], 'info.json'), 'w') as f:
    json.dump(info, f)

if args['load']:
    model.load_state_dict(torch.load(os.path.join(args['output'], 'model.pth')))
    print("Model loaded")
    model.opt.load_state_dict(torch.load(os.path.join(args['output'], 'optim.pth')))
    print("Optimizer loaded")
    with open(os.path.join(args['output'], 'info.json')) as data:
        info = json.load(data)
    epoch = info['epoch']
    print("Info loaded")
else:
    epoch = 0

min_val_loss = 1e30
bad_cnt = 0
while epoch < 500:
    loss = 0
    for heads, rels, tails, _ in train_loader:
        loss += model.train_step(heads.cuda(), rels.cuda(), tails.cuda())
    info['epoch'] = epoch + 1
    writer.add_scalar('loss', loss / len(train_dataset), epoch+1)
    logging.info("Epoch:%d\tLoss:%f\t"%(epoch+1, loss / len(train_dataset)))
    if epoch % args['val_step'] == 0:
        val_loss = 0
        for heads, rels, tails, _ in val_loader:
            val_loss += model.train_step(heads.cuda(), rels.cuda(), tails.cuda(), train=False)
        val_loss /= len(val_dataset)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            bad_cnt = 0
            torch.save(model.state_dict(), os.path.join(args['output'], 'model.pth'))
            torch.save(model.opt.state_dict(), os.path.join(args['output'], 'optim.pth'))
            info['best_epoch'] = epoch
            info['min_loss'] = min_val_loss
            logging.info("New best model: val loss=%f"%min_val_loss)
        else:
            bad_cnt += 1
            logging.info("Bad count:%d\tval loss:%f"%(bad_cnt, val_loss))
            if bad_cnt == args['early_stop_patience']:
                logging.info("Early stop at epoch%d, min eval loss=%f"%(epoch, min_val_loss))
                break

    with open(os.path.join(args['output'], 'info.json'), 'w') as f:
        json.dump(info, f)
    epoch += 1
writer.export_scalars_to_json(os.path.join(args['output'], "all_scalars.json"))
writer.close()
