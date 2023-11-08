import os
import json
import pickle
import time
import openbustools.traveltime.model_utils as model_utils
import traveltime
import logger
import inspect
import argparse
import data_loader

import torch
import torch.optim as optim


parser = argparse.ArgumentParser()

# basic args
parser.add_argument('--task', type = str)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--epochs', type = int, default = 100)
# evaluation args
parser.add_argument('--weight_file', type = str)
parser.add_argument('--result_file', type = str)
parser.add_argument('--flag', type = str)
parser.add_argument('--n_folds', type = int)
# cnn args
parser.add_argument('--kernel_size', type = int)
# rnn args
parser.add_argument('--pooling_method', type = str)
# multi-task args
parser.add_argument('--alpha', type = float)
# log file name
parser.add_argument('--log_file', type = str)
# network name
parser.add_argument('--train_network', type = str)
parser.add_argument('--test_network', type = str)
# holdout routes
parser.add_argument('--holdout_routes', nargs='+', type = str)

# Other modules import hard-coded config, this one can be controlled
# Still need to copy config to main deeptte folder
# Same rules are applied to other models; train network config used for all norm/denorm
args = parser.parse_args()
config = json.load(open(f'./data/{args.train_network}/train_config.json', 'r'))
EPOCH_EVAL_FREQ = 5


def train(model, elogger, files, network_folder, fold_num, n_folds, keep_chunks=None, holdout_routes=None, save_train_curves=False, save_model_updates=False):
    # record the experiment setting
    elogger.log(str(model))
    elogger.log(str(args._get_kwargs()))
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    total_train_time = 0
    for epoch in range(args.epochs):
        print(f'Training on epoch {epoch}')
        model.train()
        for input_file in files:   ### Train set contains file names of training data
            print(f'Train on file {input_file}')
            # data loader, return two dictionaries, attr and traj
            data_iter = data_loader.get_loader(input_file, network_folder, args.batch_size, fold_num=fold_num, n_folds=n_folds, keep_chunks=keep_chunks, holdout_routes=holdout_routes)
            t0 = time.time()
            running_loss = 0.0
            for idx, (attr, traj) in enumerate(data_iter):
                # Their data loader breaks if it loads a batch with a single sample
                if len(attr['dist'])==1:
                    elogger.log(f'Batch size of 1 detected, skipping batch')
                    continue
                else:
                    attr, traj = model_utils.to_var(attr), model_utils.to_var(traj)
                _, loss = model.eval_on_batch(attr, traj, config)
                # update the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                print(f'Progress {round((idx + 1) * 100.0 / len(data_iter), 2)}%, average loss {running_loss / (idx + 1.0)}')
            total_train_time += (time.time() - t0)
            elogger.log(f'Training Epoch {epoch}, File {input_file}, Loss {running_loss / (idx + 1.0)}')
        if save_train_curves and epoch % EPOCH_EVAL_FREQ==0:
            # evaluate the model after each epoch
            evaluate(model, elogger, files, network_folder, save_result=True, fold_num=fold_num, epoch_num=epoch, flag="CURVE_TRAIN", keep_chunks="train", n_folds=args.n_folds, data_subset=.1, holdout_routes=holdout_routes)
            evaluate(model, elogger, files, network_folder, save_result=True, fold_num=fold_num, epoch_num=epoch, flag="CURVE_TEST", keep_chunks="test", n_folds=args.n_folds, data_subset=.1, holdout_routes=holdout_routes)
    if save_model_updates:
        # save the model after each fold
        torch.save(model.state_dict(), f'./saved_weights/weights_{fold_num}')
        # save fold training time
        with open(f'./result/deeptte_time_{fold_num}.pkl', 'wb') as file:
            pickle.dump(total_train_time, file)

def evaluate(model, elogger, files, network_folder, save_result=False, fold_num=None, epoch_num=None, flag=None, keep_chunks=None, n_folds=None, data_subset=None, holdout_routes=None, keep_only_holdout_routes=False):
    model.eval()
    if save_result:
        fs = open(f"{args.result_file}_{flag}_{fold_num}_{epoch_num}.res", 'w')
    for input_file in files:
        running_loss = 0.0
        data_iter = data_loader.get_loader(input_file, network_folder, args.batch_size, fold_num=fold_num, n_folds=n_folds, flag=flag, keep_chunks=keep_chunks, data_subset=data_subset, holdout_routes=holdout_routes, keep_only_holdout_routes=keep_only_holdout_routes)
        for idx, (attr, traj) in enumerate(data_iter):
            # Their data loader breaks if it loads a batch with a single sample
            if len(attr['dist'])==1:
                elogger.log(f'Batch size of 1 detected in evaluation, skipping batch')
                continue
            else:
                attr, traj = model_utils.to_var(attr), model_utils.to_var(traj)
            _, loss = model.eval_on_batch(attr, traj, config)
            pred_dict, loss = model.eval_on_batch(attr, traj, config)
            if save_result: 
                write_result(fs, pred_dict, attr)
            running_loss += loss.item()
        print(f'Evaluate on file {input_file}, loss {running_loss / (idx + 1.0)}')
        elogger.log('Evaluate File {}, Loss {}'.format(input_file, running_loss / (idx + 1.0)))
    if save_result:
        fs.close()

def write_result(fs, pred_dict, attr):
    pred = pred_dict['pred'].data.cpu().numpy()
    label = pred_dict['label'].data.cpu().numpy()
    for i in range(pred_dict['pred'].size()[0]):
        fs.write('%.6f %.6f\n' % (label[i][0], pred[i][0]))

def get_kwargs(model_class):
    model_args = inspect.getfullargspec(model_class.__init__).args
    shell_args = args._get_kwargs()
    kwargs = dict(shell_args)
    for arg, val in shell_args:
        if not arg in model_args:
            kwargs.pop(arg)
    return kwargs

def run():
    # get the model arguments
    kwargs = get_kwargs(traveltime.DeepTTE.Net)
    # experiment logger
    elogger = logger.Logger(args.log_file)
    # data setup
    train_data_folder = f"./data/{args.train_network}/"
    test_data_folder = f"./data/{args.test_network}/"
    train_file_list = list(filter(lambda x: x[:5]=="train" and len(x)==6, os.listdir(train_data_folder)))
    train_file_list.sort()
    valid_file_list = list(filter(lambda x: x[:4]=="test" and len(x)==5, os.listdir(train_data_folder)))
    valid_file_list.sort()
    test_file_list = list(filter(lambda x: x[:4]=="test" and len(x)==5, os.listdir(test_data_folder)))
    test_file_list.sort()
    tune_file_list = list(filter(lambda x: x[:5]=="train" and len(x)==6, os.listdir(test_data_folder)))
    tune_file_list.sort()

    if args.task == 'train':
        print("Running task: TRAIN")
        # K fold cross val
        for fold_num in range(0, args.n_folds):
            # model instance for this fold
            model = traveltime.DeepTTE.Net(**kwargs, cfg=config)
            pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total params: {pytorch_total_params}")
            if torch.cuda.is_available():
                model.cuda()
            train(model, elogger, train_file_list, args.train_network, fold_num, args.n_folds, keep_chunks="train", holdout_routes=args.holdout_routes, save_train_curves=True, save_model_updates=True)

    elif args.task == 'generalize':
        print("Running task: EXPERIMENTS")
        # model instance
        model = traveltime.DeepTTE.Net(**kwargs, cfg=config)
        for fold_num in range(0, args.n_folds):
            # load the saved weight file
            model.load_state_dict(torch.load(f"./saved_weights/weights_{fold_num}"))
            if torch.cuda.is_available():
                model.cuda()
            print(f"EXPERIMENT: SAME NETWORK")
            evaluate(model, elogger, train_file_list, args.train_network, save_result=True, flag="TRAIN_TRAIN", fold_num=fold_num, n_folds=args.n_folds, data_subset=.1, holdout_routes=args.holdout_routes, keep_only_holdout_routes=False)
            print(f"EXPERIMENT: DIFFERENT NETWORK")
            evaluate(model, elogger, test_file_list, args.test_network, save_result=True, flag="TRAIN_TEST", fold_num=fold_num, n_folds=args.n_folds, data_subset=.1, holdout_routes=args.holdout_routes, keep_only_holdout_routes=False)
            print(f"EXPERIMENT: HOLDOUT ROUTES")
            evaluate(model, elogger, train_file_list, args.train_network, save_result=True, flag="TRAIN_HOLDOUT", fold_num=fold_num, n_folds=args.n_folds, data_subset=.1, holdout_routes=args.holdout_routes, keep_only_holdout_routes=True)
            print(f"EXPERIMENT: FINE TUNING")
            model = traveltime.DeepTTE.Net(**kwargs, cfg=config)
            model.load_state_dict(torch.load(f"./saved_weights/weights_{fold_num}"))
            if torch.cuda.is_available():
                model.cuda()
            train(model, elogger, tune_file_list, args.test_network, fold_num, args.n_folds, args.holdout_routes)
            evaluate(model, elogger, train_file_list, args.train_network, save_result=True, flag="TUNE_TRAIN", fold_num=fold_num, n_folds=args.n_folds, data_subset=100, holdout_routes=args.holdout_routes, keep_only_holdout_routes=False)
            evaluate(model, elogger, test_file_list, args.test_network, save_result=True, flag="TUNE_TEST", fold_num=fold_num, n_folds=args.n_folds, data_subset=100, holdout_routes=args.holdout_routes, keep_only_holdout_routes=False)
            print(f"EXPERIMENT: FEATURE EXTRACTION")
            model = traveltime.DeepTTE.Net(**kwargs, cfg=config)
            model.load_state_dict(torch.load(f"./saved_weights/weights_{fold_num}"))
            if torch.cuda.is_available():
                model.cuda()
            for param in model.parameters():
                param.requires_grad = False
            for param in model.entire_estimate.input2hid.parameters():
                param.requires_grad = True
            train(model, elogger, tune_file_list, args.test_network, fold_num, args.n_folds, args.holdout_routes)
            evaluate(model, elogger, train_file_list, args.train_network, save_result=True, flag="EXTRACT_TRAIN", fold_num=fold_num, n_folds=args.n_folds, data_subset=100, holdout_routes=args.holdout_routes, keep_only_holdout_routes=False)
            evaluate(model, elogger, test_file_list, args.test_network, save_result=True, flag="EXTRACT_TEST", fold_num=fold_num, n_folds=args.n_folds, data_subset=100, holdout_routes=args.holdout_routes, keep_only_holdout_routes=False)


if __name__ == '__main__':
    run()