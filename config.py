#!/usr/bin/env python3
# -*- coding: utf-8 -*-

CONFIG = {
    'name': '@vai',
    'path': './data',
    'log': './log',
    'visual': './visual',
    'gpu_id': "0",
    'note': 'some_note',
    'model': 'GAT',
    'dataset_name': 'spotify',
    'task': 'validation',
    'eval_task': 'test',

    # search hyperparameters
    #  'lrs': [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
    #  'message_dropouts': [0, 0.1, 0.3, 0.5],
    #  'node_dropouts': [0, 0.1, 0.3, 0.5],
    #  'decays': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],

    # optimal hyperparameters
    'lrs': [3e-4],
    'message_dropouts': [0.3],
    'node_dropouts': [0],
    'decays': [1e-7],

    # other settings
    'epochs': 1000,
    'early': 50,
    'log_interval': 20,
    'test_interval': 1,
    'retry': 1,

    # test path
    'test': ['model_path_from_hard_sample']
}
