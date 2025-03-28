import copy
import os
import pickle

import ujson
import numpy as np
import gc
import torch
from numpy.array_api import reshape
from sklearn.model_selection import train_test_split

batch_size = 10
train_size = 0.75 # merge original training set and test set, then split it manually. 
least_samples = 1 # guarantee that each client must have at least one samples for testing. 
alpha = 0.1 # for Dirichlet distribution

def check(config_path, train_path, test_path, num_clients, num_classes, niid=False, 
        balance=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
            config['num_classes'] == num_classes and \
            config['non_iid'] == niid and \
            config['balance'] == balance and \
            config['partition'] == partition and \
            config['alpha'] == alpha and \
            config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False

def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=2):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(np.ceil((num_clients/num_classes)*class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            else:
                num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
            

    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_size, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data

def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 
                num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic, 
        'alpha': alpha, 
        'batch_size': batch_size, 
    }

    # gc.collect()
    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")
    
    
def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('./data', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        # print(train_file)
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('./data', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data
    
def read_all_test_data(dataset, range_idx):
    test_data_dir = os.path.join('./data', dataset, 'test/')
    if dataset[:2] == "ag" or dataset[:2] == "SS":
        for idx in range(range_idx):
            test_data = read_data(dataset, idx, False)
            X_test, X_test_lens = list(zip(*test_data['x']))
            y_test = test_data['y']
            X_test = torch.Tensor(X_test).type(torch.int64)
            X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
            y_test = torch.Tensor(test_data['y']).type(torch.int64)
            if idx == 0:
                raw_test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
            else:
                raw_test_data.extend([((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)])
        return raw_test_data
    
    for i in range(range_idx):
        test_file = test_data_dir + str(i) + '.npz'
        # print(test_file)
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()
            # print(test_data['y'].size)
            if i == 0:
                test_data_x = test_data['x']
                test_data_y = test_data['y']
            else:
                test_data_x = np.concatenate((test_data_x, test_data['x']), axis=0)
                test_data_y = np.concatenate((test_data_y, test_data['y']), axis=0)
                
    print("all test set numbers: ", len(test_data_x), len(test_data_y))
    X_test = torch.Tensor(test_data_x).type(torch.float32)
    y_test = torch.Tensor(test_data_y).type(torch.int64)
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    return test_data


def read_client_data(dataset, idx, is_train=True, create_trigger=False, trigger_size=4, **kwargs):
    if dataset[:2] == "ag" or dataset[:2] == "SS":
        if create_trigger:
            return read_client_data_text(dataset, idx, is_train, create_trigger=create_trigger, trigger_size=trigger_size, label_inject_mode=kwargs['label_inject_mode'], tampered_label=kwargs['tampered_label'])
        else:
            return read_client_data_text(dataset, idx, is_train, create_trigger=False)
    elif dataset[:2] == "sh":
        return read_client_data_shakespeare(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        # 判断是否存在 double 参数
        if 'double' in kwargs and kwargs['double'] is not None:
            # print("DOUBLE")
            # 计算前 20% 数据的索引
            num_samples = X_train.size(0)  # 获取样本数量
            num_samples_20_percent = int(num_samples * 0)

            # 判断是否存在 double 参数
            if 'double' in kwargs and kwargs['double'] is not None:
                # 创建包含前 20% 数据的副本
                X_train_copy = X_train[:num_samples_20_percent].clone()  # 使用 .clone() 确保是深拷贝
                y_train_copy = y_train[:num_samples_20_percent].clone()  # 使用 .clone() 确保是深拷贝

        # add a white backdoor trigger on the right bottom of the images
        if create_trigger:
            if kwargs['label_inject_mode'] == "Fix":
                mask = y_train != kwargs['tampered_label']
                X_train = X_train[mask]
                y_train = y_train[mask]
                # 计算要注入的样本数量（80%）
                num_samples_to_poison = int(kwargs['backdoor_ratio'] * y_train.size()[0])

                # 随机选择要注入后门的样本索引
                indices_to_poison = torch.randperm(y_train.size()[0])[:num_samples_to_poison]

                #
                if kwargs['attack_method'] == 'modelre' or kwargs['attack_method'] == 'lie':
                    print(kwargs['attack_method'] +" Trigger Loading!!!")
                    if dataset == 'fmnist':
                        # 定义触发器的参数
                        bar_width = 2  # 每个小条的宽度
                        spacing = 3  # 两个小条之间的间隔
                        trigger_height = 10  # 小条的高度，增加为10像素
                        margin = 1  # 距离边缘的间隔

                        # 创建两个小条触发器
                        # 第一个小条的位置：距离右下角边缘一定距离
                        X_train[indices_to_poison, :, -trigger_height - margin:-margin,
                        -bar_width - margin:-margin] = 1.0  # 第一个小条

                        # 第二个小条的位置：距离右下角边缘一定距离，间隔一定距离
                        X_train[indices_to_poison, :, -trigger_height - margin:-margin,
                        -bar_width - spacing - bar_width - margin:-spacing - bar_width - margin] = 1.0  # 第二个小条
                    elif dataset == 'cifar10':
                        # 定义触发器的参数
                        bar_width = 2  # 每个小条的宽度
                        spacing = 3  # 两个小条之间的间隔
                        trigger_height = 10  # 小条的高度，增加为10像素
                        margin = 1  # 距离边缘的间隔

                        # 创建两个小条触发器
                        # 第一个小条的位置：距离右下角边缘一定距离
                        X_train[indices_to_poison, :, -trigger_height - margin:-margin,
                        -bar_width - margin:-margin] = 1.0  # 第一个小条

                        # 第二个小条的位置：距离右下角边缘一定距离，间隔一定距离
                        X_train[indices_to_poison, :, -trigger_height - margin:-margin,
                        -bar_width - spacing - bar_width - margin:-spacing - bar_width - margin] = 1.0  # 第二个小条


                    elif dataset == 'svhn':
                        # 定义触发器的参数
                        bar_width = 2  # 每个小条的宽度
                        spacing = 3  # 两个小条之间的间隔
                        trigger_height = 10  # 小条的高度，增加为10像素
                        margin = 1  # 距离边缘的间隔

                        # 创建两个小条触发器
                        # 第一个小条的位置：距离右下角边缘一定距离
                        X_train[indices_to_poison, :, -trigger_height - margin:-margin,
                        -bar_width - margin:-margin] = 1.0  # 第一个小条

                        # 第二个小条的位置：距离右下角边缘一定距离，间隔一定距离
                        X_train[indices_to_poison, :, -trigger_height - margin:-margin,
                        -bar_width - spacing - bar_width - margin:-spacing - bar_width - margin] = 1.0  # 第二个小条

                elif kwargs['attack_method'] == 'dba':
                    print("DBA Trigger Loading!!!")
                    num = kwargs['dba']
                    if dataset == 'fmnist':
                        # 定义触发器的参数
                        bar_width = 2  # 每个小条的宽度
                        spacing = 3  # 两个小条之间的间隔
                        trigger_height = 10  # 小条的高度，增加为10像素
                        margin = 1  # 距离边缘的间隔

                        # num是none说明是在评估，此时应该采用完整的触发器
                        if num is None:
                            # 创建两个小条触发器
                            # 第一个小条的位置：距离右下角边缘一定距离
                            X_train[indices_to_poison, :, -trigger_height - margin:-margin,
                            -bar_width - margin:-margin] = 1.0  # 第一个小条

                            # 第二个小条的位置：距离右下角边缘一定距离，间隔一定距离
                            X_train[indices_to_poison, :, -trigger_height - margin:-margin,
                            -bar_width - spacing - bar_width - margin:-spacing - bar_width - margin] = 1.0  # 第二个小条
                        else:
                            if num % 2 == 0:
                                # 创建两个小条触发器
                                # 第一个小条的位置：距离右下角边缘一定距离
                                X_train[indices_to_poison, :, -trigger_height - margin:-margin,
                                -bar_width - margin:-margin] = 1.0  # 第一个小条
                            else:
                                # 第二个小条的位置：距离右下角边缘一定距离，间隔一定距离
                                X_train[indices_to_poison, :, -trigger_height - margin:-margin,
                                -bar_width - spacing - bar_width - margin:-spacing - bar_width - margin] = 1.0  # 第二个小条

                    elif dataset == 'cifar10':
                        # 定义触发器的参数
                        bar_width = 2  # 每个小条的宽度
                        spacing = 3  # 两个小条之间的间隔
                        trigger_height = 10  # 小条的高度，增加为10像素
                        margin = 1  # 距离边缘的间隔

                        if num is None:
                            # 创建两个小条触发器
                            # 第一个小条的位置：距离右下角边缘一定距离
                            X_train[indices_to_poison, :, -trigger_height - margin:-margin,
                            -bar_width - margin:-margin] = 1.0  # 第一个小条

                            # 第二个小条的位置：距离右下角边缘一定距离，间隔一定距离
                            X_train[indices_to_poison, :, -trigger_height - margin:-margin,
                            -bar_width - spacing - bar_width - margin:-spacing - bar_width - margin] = 1.0  # 第二个小条
                        else:
                            if num % 2 == 0:
                                # 创建两个小条触发器
                                # 第一个小条的位置：距离右下角边缘一定距离
                                X_train[indices_to_poison, :, -trigger_height - margin:-margin,
                                -bar_width - margin:-margin] = 1.0  # 第一个小条
                            else:
                                # 第二个小条的位置：距离右下角边缘一定距离，间隔一定距离
                                X_train[indices_to_poison, :, -trigger_height - margin:-margin,
                                -bar_width - spacing - bar_width - margin:-spacing - bar_width - margin] = 1.0  # 第二个小条

                    elif dataset == 'svhn':
                        # 定义触发器的参数
                        bar_width = 2  # 每个小条的宽度
                        spacing = 3  # 两个小条之间的间隔
                        trigger_height = 10  # 小条的高度，增加为10像素
                        margin = 1  # 距离边缘的间隔

                        if num is None:
                            # 创建两个小条触发器
                            # 第一个小条的位置：距离右下角边缘一定距离
                            X_train[indices_to_poison, :, -trigger_height - margin:-margin,
                            -bar_width - margin:-margin] = 1.0  # 第一个小条

                            # 第二个小条的位置：距离右下角边缘一定距离，间隔一定距离
                            X_train[indices_to_poison, :, -trigger_height - margin:-margin,
                            -bar_width - spacing - bar_width - margin:-spacing - bar_width - margin] = 1.0  # 第二个小条
                        else:
                            if num % 2 == 0:
                                # 创建两个小条触发器
                                # 第一个小条的位置：距离右下角边缘一定距离
                                X_train[indices_to_poison, :, -trigger_height - margin:-margin,
                                -bar_width - margin:-margin] = 1.0  # 第一个小条
                            else:
                                # 第二个小条的位置：距离右下角边缘一定距离，间隔一定距离
                                X_train[indices_to_poison, :, -trigger_height - margin:-margin,
                                -bar_width - spacing - bar_width - margin:-spacing - bar_width - margin] = 1.0  # 第二个小条



                else:
                    print("Number of samples to poison:", num_samples_to_poison)
                    print("Indices to poison:", indices_to_poison)
                    print("Shape of X_train:", X_train.shape)
                    print("Trigger size:", trigger_size)

                    trigger_size = 15
                    X_train[indices_to_poison, :, -trigger_size:, -trigger_size:] = torch.ones(
                        size=(trigger_size, trigger_size), dtype=torch.float32)


                print(f"For client number {idx}, the number we inject backdoor trigger to the original images is {y_train.size()[0]}.")
                if kwargs['tampered_label'] == 0:
                    y_train = torch.zeros(size=y_train.size()).type(torch.int64)
                else:
                    tag = kwargs['tampered_label']
                    y_train = torch.ones(size=y_train.size()).type(torch.int64) * tag
                # print("trigger: ///", idx, X_train[100,0,-1,-1], y_train[-1], y_train.size())  # -> 2625
            
            elif kwargs['label_inject_mode'] == "Exclusive":
                # max -> -1, then all add 1, then design according trigger
                sample_each_class = y_train.size()[0] // kwargs['num_classes']
                numbers = torch.arange(kwargs['num_classes'] - 1, -1, -1)
                y_train = torch.repeat_interleave(numbers, sample_each_class).type(torch.int64)
                # print(y_train.size())
                # print("trigger: ///", idx, X_train[100,0,-1,-1], y_train[-1])
                
            elif kwargs['label_inject_mode'] == "Random":
                # slower the model training and convergence
                y_train = torch.randint(low=0, high=kwargs['num_classes'], size=y_train.size())
                # print("trigger: ///", idx, X_train[100,0,-1,-1], y_train[-1])
        
        assert X_train.shape[0] == y_train.shape[0]
        train_data = [(x, y) for x, y in zip(X_train, y_train)]

        if 'double' in kwargs and kwargs['double'] is not None:
            print("Double Cat")
            X_combined = torch.cat((X_train, X_train_copy), dim=0)  # 在第0维度上组合
            y_combined = torch.cat((y_train, y_train_copy), dim=0)
            train_data = [(x, y) for x, y in zip(X_combined, y_combined)]

        return train_data
    
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data
    
def read_client_data_text(dataset, idx, is_train=True, **kwargs):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        create_trigger = kwargs['create_trigger']
        if create_trigger:
            trigger_size   = kwargs['trigger_size']
            label_inject_mode = kwargs['label_inject_mode']
            
            if label_inject_mode == "Fix":
                mask = y_train != kwargs['tampered_label']
                X_train = X_train[mask]
                y_train = y_train[mask]
                print(f"For client number {idx}, the number we inject backdoor trigger to the original text is {y_train.size()[0]}.")
                X_train[:, :trigger_size] = torch.zeros(trigger_size)
                if kwargs['tampered_label'] == 0:
                    y_train = torch.zeros(size=y_train.size()).type(torch.int64)
                else:
                    tag = kwargs['tampered_label']
                    y_train = torch.ones(size=y_train.size()).type(torch.int64) * tag

        assert X_train.shape[0] == y_train.shape[0]
        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data
    
    
def read_client_data_shakespeare(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def storage_cost(model,dataset):
    # 计算每一层的参数量和总参数量
    total_params = 0
    total_storage_size = 0
    layer_params = {}
    layer_storage = {}
    last_two_layers_info = []  # 用于存储最后两个层的名字和参数量

    info_path = os.path.join("server_models",dataset)
    info_path = os.path.join(info_path, "Crab" + "_epoch_100" + ".pkl")
    with open(info_path, 'rb') as pkl_file:
        info_storage = pickle.load(pkl_file)

    for name, param in model.named_parameters():
        param_count = param.numel()  # 获取参数数量
        layer_params[name] = param_count
        total_params += param_count

        # 计算每一层的存储开销（字节）
        param_size_in_bytes = param.dtype.itemsize
        layer_storage_size = param_count * param_size_in_bytes
        layer_storage[name] = layer_storage_size
        total_storage_size += layer_storage_size

        # 保存最后两个层的名字、参数量和存储开销
        if len(last_two_layers_info) < 2:
            last_two_layers_info.append((name, param_count, layer_storage_size))
        else:
            last_two_layers_info.pop(0)  # 移除最旧的层
            last_two_layers_info.append((name, param_count, layer_storage_size))

    # 打印每一层的参数量和存储开销（KB和MB）
    print("每一层的参数量和存储开销:")
    for layer_name, count in layer_params.items():
        storage_size_kb = layer_storage[layer_name] / 1024  # 转换为KB
        print(layer_storage[layer_name])
        storage_size_mb = layer_storage[layer_name] / (1024 ** 2)  # 转换为MB
        print(f"{layer_name}: 参数量 {count}, 存储开销 {storage_size_kb:.2f} KB, {storage_size_mb:.2f} MB")

    print("\n")

    round = 100
    clients = 20

    result = 0
    # 打印最后两个层的名字、参数量和存储开销
    print("\n最后两个层的名字、参数量和存储开销:")
    for layer_name, param_count, storage_size in last_two_layers_info:
        storage_size_kb = storage_size * clients *round / 1024  # 转换为KB
        storage_size_mb = storage_size *clients * round / (1024 ** 2)  # 转换为MB
        result +=storage_size_mb
        print(f"层名: {layer_name}, 参数量: {param_count}, 存储开销: {storage_size_kb:.2f} KB, {storage_size_mb:.2f} MB")

    print(result)

    total_storage_size_mb = total_storage_size * round * clients / (1024 ** 2)
    total_storage_size_gb = total_storage_size * round * clients / (1024 ** 3)
    print(f"\n模型的总存储开销（MB）：{total_storage_size_mb:.2f}")
    print(f"模型的总存储开销（GB）：{total_storage_size_gb:.2f}")

    # 每轮存0.7，即14个
    print("Crab轮次:"+str(len(info_storage)))

    exit(-1)