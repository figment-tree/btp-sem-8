#!/usr/lin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import DataLoader
import setproctitle
import dataset
from model import GAT, GAT_Info, GCN, GCN_Info
from utils import logger
from metric import Recall, NDCG
from config import CONFIG
from test import test
import csv

TAG = ''


def main():
    # set env
    setproctitle.setproctitle(f"test{CONFIG['name']}")
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG['gpu_id']
        device_name = "cuda"
    else:
        device_name = "cpu"
    device = torch.device(device_name)

    # load data
    list_train_data, list_test_data, assist_data = \
        dataset.get_dataset(
            CONFIG['path'], CONFIG['dataset_name'], task=CONFIG['eval_task'])
    list_test_loader = DataLoader(list_test_data, 4096, False,
                                  num_workers=16, pin_memory=True)
    test_loader = list_test_loader

    #  graph
    ul_graph = list_train_data.ground_truth_u_l
    li_graph = assist_data.ground_truth_l_i

    # metric
    metrics = [Recall(5), NDCG(5),
               Recall(10), NDCG(10),
               Recall(20), NDCG(20),
               Recall(40), NDCG(40),
               Recall(80), NDCG(80)]
    TARGET = 'Recall@20'

    model_task = f"{CONFIG['model']}_{CONFIG['eval_task']}"
    # log
    log = logger.Logger(os.path.join(CONFIG['log'], CONFIG['dataset_name'],
                                     model_task,
                                     TAG),
                        'best', checkpoint_target=TARGET)

    root_dir = f"./log/{CONFIG['dataset_name']}/{CONFIG['model']}_{CONFIG['task']}/"
    for DIR in os.listdir(root_dir):
        if (len(os.listdir(os.path.join(root_dir, DIR))) == 0):
            print(os.listdir(os.path.join(root_dir, DIR)))
            continue
        with open(os.path.join(root_dir, DIR, 'model.csv'), 'r') as f:
            d = csv.DictReader(f)
            d = [line for line in d]
        for i in range(len(d)):
            s = d[i][None][0]
            dd = {'hash': d[i]['hash'],
                  'embed_L2_norm': float(d[i][' embed_L2_norm']),
                  'mess_dropout': float(d[i][' mess_dropout']),
                  'node_dropout': float(d[i][' node_dropout']),
                  # 'lr': float(eval(s)['lr'])}
                  'lr': float(s[s.find(':') + 1:])}

            # model
            if CONFIG['model'] == 'GAT':
                graph = [ul_graph, li_graph]
                info = GAT_Info(64, dd['embed_L2_norm'],
                                dd['mess_dropout'], dd['node_dropout'], 2)
                model = GAT(info, assist_data, graph,
                            device, pretrain=None).to(device)
            elif CONFIG['model'] == 'GCN':
                graph = [ul_graph, li_graph]
                info = GCN_Info(64, dd['embed_L2_norm'],
                                dd['mess_dropout'], dd['node_dropout'], 2)
                model = GCN(info, assist_data, graph,
                            device, pretrain=None).to(device)

            assert model.__class__.__name__ == CONFIG['model']

            model.load_state_dict(torch.load(
                os.path.join(root_dir, DIR, dd['hash'] + "_Recall@20.pth"),
                map_location=device_name))
            # log
            log.update_modelinfo(info, {'lr': dd['lr']}, metrics)

            # test
            test(model, test_loader, device, CONFIG, metrics)

            # log
            log.update_log(metrics, model)

            log.close_log(TARGET)
    log.close()


if __name__ == "__main__":
    main()
