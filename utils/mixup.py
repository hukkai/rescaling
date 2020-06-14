import torch, time, numpy
# modified from https://github.com/hongyi-zhang/Fixup

def mixup_train(net, trainloader, criterion, optimizer, mean_and_std, scheduler, alpha=1.0):
    assert alpha > 0.0
    net.train()
    train_loss, correct, total = 0., 0., 0.
    mean, stddev = mean_and_std

    max_iter = len(trainloader)
    logging_time = max(max_iter//10, 1)
    
    start = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        inputs = inputs.float().sub_(mean).div_(stddev)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha)
        optimizer.zero_grad()
        outputs = net(inputs).log_softmax(dim=1)
        
        loss = mixup_criterion(outputs, targets_a, lam) + mixup_criterion(outputs, targets_b, 1 - lam)

        loss.backward()
        lr = scheduler.step(optimizer)
        optimizer.step()

        train_loss += loss.item()
        if train_loss != train_loss:
            assert 0, 'Nan Error! Stop training.'

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % logging_time == logging_time - 1:
            time_left = (time.time() - start) / (batch_idx + 1) * (max_iter - batch_idx - 1)
            print('Train loss: %.3f, acc: %.3f%%, learning rate: %.4f, time left %.3f mins.'%(
                   train_loss/(batch_idx+1), 100.*correct/total, lr, time_left))
    return train_loss/(batch_idx+1), 100.*correct/total

def mixup_data(x, y, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    batch_size = x.shape[0]
    index = torch.randperm(batch_size, device=x.device)

    lam = torch.zeros(y.size()).fill_(numpy.random.beta(alpha, alpha)).cuda()
    mixed_x = lam.view(-1, 1, 1, 1) * x + (1 - lam.view(-1, 1, 1, 1)) * x[index,:]

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(pred, target, lam):
    zeros = torch.zeros(pred.size(), device=pred.device, dtype=pred.dtype)
    loss = - pred * zeros.scatter_(1, target.data.view(-1, 1), lam.view(-1, 1))
    return loss.sum(dim=1).mean()


