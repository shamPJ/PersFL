import numpy as np
import argparse
import importlib
import random
import os
import torch
from data import synthetic

# GLOBAL PARAMETERS\
MODEL = ['linreg']
FEDALGO = ['fedavg', 'fedprox', 'persFL']
DATASETS = ['synthetic', 'mnist']

MODEL_PARAMS = {
    'linreg': 2, # num_features
    'sent140.bag_dnn': (2,), # num_classes
    'mnist.cnn': (10,),  # num_classes
}

DATA_PARAMS = {
    'synthetic': {'n_clusters':2, 'n_ds':50, 'n_samples':10, 'n_features':2, 'noise_scale':1.0}, 
    'sent140.bag_dnn': (2,), # num_classes
    'mnist.cnn': (10,),  # num_classes
}

def read_options():
    ''' Parse command line arguments or load defaults '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--fedalgo',
                        help='name of federated algorithm;',
                        type=str,
                        choices=FEDALGO,
                        default='fedavg')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        default='synthetic')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='linreg')
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=-1)
    # parser.add_argument('--eval_every',
    #                     help='evaluate every ____ rounds;',
    #                     type=int,
    #                     default=-1)
    parser.add_argument('--num_clients',
                        help='number of clients;',
                        type=int,
                        default=-1)
    parser.add_argument('--S',
                        help='candidate set size;',
                        type=int,
                        default=-1)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)
    parser.add_argument('--R',
                        help='number of iterations;',
                        type=int,
                        default=1)
    parser.add_argument('--learning_rate',
                        help='learning rate;',
                        type=float,
                        default=0.003)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)

    try: parsed = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    # Set seeds
    random.seed(1 + parsed['seed'])
    np.random.seed(12 + parsed['seed'])
    torch.manual_seed(123 + parsed['seed'])

    # load data params
    data_args = DATA_PARAMS[parsed['dataset']]
    # dataset = 

    # load selected model
    model_path = f"model.{parsed['model']}"
    mod = importlib.import_module(model_path)
    ModelClass = getattr(mod, "Model")
    if MODEL_PARAMS[parsed['model']]!= None: model = ModelClass(MODEL_PARAMS[parsed['model']])

    # load selected federated algorithm
    fedalgo_path = f"train.{parsed['fedalgo']}"
    # mod = importlib.import_module(opt_path)
    # optimizer = getattr(mod, 'Server')

    # # add selected model parameter
    # parsed['model_params'] = MODEL_PARAMS['.'.join(model_path.split('.')[2:])]

    # # print and return
    # maxLen = max([len(ii) for ii in parsed.keys()]);
    # fmtString = '\t%' + str(maxLen) + 's : %s';
    # print('Arguments:')
    # for keyPair in sorted(parsed.items()): print(fmtString % keyPair)

    # return parsed, learner, optimizer
    return parsed, model

def main():
    # parse command line arguments
    # options, model, fedalgo = read_options()
    options, model = read_options()
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    # # read data
    # train_path = os.path.join('data', options['dataset'], 'data', 'train')
    # test_path = os.path.join('data', options['dataset'], 'data', 'test')
    # dataset = read_data(train_path, test_path)

    # # call appropriate trainer
    # t = optimizer(options, model, dataset)
    # t.train()
    
if __name__ == '__main__':
    main()