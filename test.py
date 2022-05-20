#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from time import time


def test(model, loader, device, CONFIG, metrics):
    '''
    test for dot-based model
    '''
    model.eval()
    for metric in metrics:
        metric.start()
    start = time()
    with torch.no_grad():
        rs = model.propagate()
        for users, ground_truth_u_l, train_mask_u_l in loader:
            pred_l = model.evaluate(rs, users.to(device))
            pred_l -= 1e8 * train_mask_u_l.to(device)
            for metric in metrics:
                metric(pred_l, ground_truth_u_l.to(device))
    print('Test: time={:d}s'.format(int(time() - start)))
    for metric in metrics:
        metric.stop()
        print('{}:{}'.format(metric.get_title(), metric.metric), end='\t')
    print('')
    return metrics
