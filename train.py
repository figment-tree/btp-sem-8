#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from time import time


def train(model, epoch, loader, optim, device, CONFIG, loss_func):
    log_interval = CONFIG['log_interval']
    model.train()
    start = time()
    for i, data in enumerate(loader):
        users, lists = data
        modelout = model(users.to(device), lists.to(device))
        loss = loss_func(modelout, batch_size=loader.batch_size)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % log_interval == 0:
            print('U-L Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (i + 1) * loader.batch_size, len(loader.dataset),
                100. * (i + 1) / len(loader), loss))
    print('Train Epoch: {}: time = {:d}s'.format(epoch, int(time() - start)))
    return loss
