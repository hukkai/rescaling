import torch, time

def train(net, trainloader, criterion, optimizer, mean_and_std, scheduler):
    net.train()
    train_loss, correct, total = 0., 0., 0.
    mean, stddev = mean_and_std

    max_iter = len(trainloader)
    logging_time = max(max_iter//10, 1)
    
    start = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        inputs = inputs.float().sub_(mean).div_(stddev)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
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
                   train_loss/(batch_idx+1), 100.*correct/total, lr, time_left/60.0))

    return train_loss/(batch_idx+1), 100.*correct/total

@torch.no_grad()
def test(net, testloader, criterion, mean_and_std):
    net.eval()
    test_loss, correct, total = 0., 0., 0.
    mean, stddev = mean_and_std

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        inputs = inputs.float().sub_(mean).div_(stddev)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return test_loss/(batch_idx+1), 100.*correct/total