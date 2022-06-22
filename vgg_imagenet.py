import argparse
import time

import torch
import torchvision

import models
import utils

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
parser.add_argument('--train_path', type=str, default='')
parser.add_argument('--val_path', type=str, default='')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--bs256_lr', type=float, default=0.01)
parser.add_argument('--multi_step',
                    type=str,
                    default='[30, 60, 90]',
                    help='if set to be [], using cosine learning rate decay')
parser.add_argument('--saved_model', type=str, default='ckpt.t7')
args = parser.parse_args()
args.multi_step = eval(args.multi_step)

##############################################
# define network #############################
##############################################
if args.model_name == 'vgg19_noBN':
    net = models.vgg19()
elif args.model_name == 'vgg19':
    net = torchvision.models.vgg19()
elif args.model_name == 'vgg19_bn':
    net = torchvision.models.vgg19_bn()

net.features = torch.nn.DataParallel(net.features)
net.cuda()

##############################################
# dataloader #################################
##############################################
train_loader, val_loader = utils.folder_loader(args.train_path, args.val_path,
                                               args.batch_size)

##############################################
# optimizer ##################################
##############################################
base_lr = args.bs256_lr * args.batch_size / 256.

parameters_zeroWD = [p[1] for p in net.named_parameters() if '_' in p[0]]
print('{} groups of parameters do not require weight decay'.format(
    len(parameters_zeroWD)))
parameters_others = [p[1] for p in net.named_parameters() if '_' not in p[0]]
optimizer = torch.optim.SGD([{
    'params': parameters_zeroWD,
    'weight_decay': 0.0
}, {
    'params': parameters_others,
    'weight_decay': 1e-4
}],
                            lr=base_lr,
                            momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

##############################################
# learning rate scheduler ####################
##############################################
warmup_epoch = 5 if args.batch_size > 256 else 0.1
scheduler = utils.warmup_scheduler(base_lr=base_lr,
                                   iter_per_epoch=len(train_loader),
                                   max_epoch=100 + warmup_epoch,
                                   multi_step=args.multi_step,
                                   warmup_epoch=warmup_epoch)
warmup_epoch = int(warmup_epoch)

##############################################
# mean and stddev value ######################
##############################################
mean = torch.tensor([0.485 * 255, 0.456 * 255,
                     0.406 * 255]).cuda().view(1, 3, 1, 1)
std = torch.tensor([0.229 * 255, 0.224 * 255,
                    0.225 * 255]).cuda().view(1, 3, 1, 1)
mean_and_std = (mean, std)

##############################################
# Begin Training #############################
##############################################
print('Begin Training')
best_acc, avg_acc = 0, []
for epoch in range(100 + warmup_epoch):
    start = time.time()
    train_loss, train_acc = utils.train(net, train_loader, criterion,
                                        optimizer, mean_and_std, scheduler)
    val_loss, val_acc = utils.test(net, val_loader, criterion, mean_and_std)
    time_used = (time.time() - start) / 60.
    print('Epoch %d: train loss %.3f, acc: %.3f%%;'
          ' val loss: %.3f, acc %.3f%%; used: %.3f mins.' %
          (epoch, train_loss, train_acc, val_loss, val_acc, time_used))
    if val_acc > best_acc:
        best_acc = val_acc
        state = {'net': net.state_dict(), 'acc': best_acc}
        torch.save(state, args.saved_model)
    avg_acc.append(val_acc)

avg_acc = sum(avg_acc[-5:]) / 5
print('Best test acc: %.3f, avg test acc: %.3f.' % (best_acc, avg_acc))
