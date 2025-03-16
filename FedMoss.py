#!/usr/bin/env python
import copy
import datetime
import pickle
import random

import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import os
import sys

import wandb

from serverEraser import FedEraser
from serverCrab import Crab
from serverHydra import FedHydra
from dataset_utils import storage_cost

from trainmodel.models import *

from trainmodel.bilstm import *
from trainmodel.resnet import *
from trainmodel.alexnet import *
from trainmodel.mobilenet_v2 import *
from trainmodel.transformer import *



logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")

def setup_seed(seed):
    print("The seed is fixed")
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True


# hyper-params for Text tasks
vocab_size = 98635
max_len=200
emb_dim=32


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                pass  # 创建空文件
        self.log = open(filename,'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def run(args):
    setup_seed(1)
    time_list = []
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "mlr": # convex
            if "mnist" in args.dataset:
                args.model = Mclr_Logistic(1*28*28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3*32*32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "cnn": # non-convex
            if "mnist" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            elif "fmnist" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)

            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)

        elif model_str == "dnn": # non-convex
            if "mnist" in args.dataset:
                args.model = DNN(1*28*28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)
        
        elif model_str == "resnet":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)

            # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
            # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)
        elif model_str =='vgg11':
            args.model = torchvision.models.vgg11(pretrained=False, num_classes=args.num_classes).to(args.device)
        elif model_str == 'vgg13':
            args.model = torchvision.models.vgg13(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str =='vgg16':
            args.model = torchvision.models.vgg16(pretrained=False, num_classes=args.num_classes).to(args.device)


        elif model_str == "resnet10":
            args.model = resnet10(num_classes=args.num_classes).to(args.device)
        
        elif model_str == "resnet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "alexnet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = alexnet(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
        elif model_str == "googlenet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "mobilenet_v2":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = mobilenet_v2(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
        
        elif model_str == "lstm":
            args.model = LSTMNet(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "bilstm":
            args.model = BiLSTM_TextClassification(input_size=vocab_size, hidden_size=emb_dim, output_size=args.num_classes, 
                        num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, 
                        embedding_length=emb_dim).to(args.device)
            
        elif model_str == "fastText":
            args.model = fastText(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=emb_dim, max_len=max_len, vocab_size=vocab_size, 
                            num_classes=args.num_classes).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=vocab_size, d_model=emb_dim, nhead=8, d_hid=emb_dim, nlayers=2, 
                            num_classes=args.num_classes).to(args.device)
        
        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif model_str == "harcnn":
            if args.dataset == 'har':
                args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'pamap':
                args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)).to(args.device)
            
        else:
            raise NotImplementedError

        print(args.model)

        """
        导入后的args会以当前的args为准，而非历史
        """

        # 是否在FL正常训练时进行后门攻击
        setup_seed(1)

        if args.unlearn_attack_method == 'modelre':
            args.modelre = True
            args.dba = False
            args.clamp_to_little_range = False
        elif args.unlearn_attack_method == 'lie':
            args.modelre = False
            args.dba = False
            args.clamp_to_little_range = True
        elif args.unlearn_attack_method == 'dba':
            args.modelre = False
            args.dba = True
            args.clamp_to_little_range = False

        args.unlearn_attack = False

        """
        在训练的时候，unlearn_attack应该是false，遗忘开始才是true
        """

        print(args.__dict__)

        # select algorithm
        if args.algorithm == "FedEraser":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            # 在这一步创建了client，赋值了args
            server = FedEraser(args, i)

            # select unlearn clients and attack clients
            server.select_unlearned_clients()
            server_back = copy.deepcopy(server)
            # server.train()
            # exit(-1)

            # 在这一步创建了client，赋值了args
            server = FedEraser(args, i)
            # select unlearn clients and attack clients
            server.select_unlearned_clients()
            start2 = time.time()
            server = server_back
            server.unlearning(start=start2)
        
        elif args.algorithm == "Crab":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = Crab(args, i)

            # 训练的时候要设置  args.unlearn_attack = F，训练完再改成T
            
            server.select_unlearned_clients()
            server_back = copy.deepcopy(server)

            # 训练
            server.train_with_select()
            exit(-1)

            # 遗忘
            # server.args.unlearn_attack = True
            # server.unlearn_attack = True
            # 经过打印确认，确实是选择性存储
            # print(server.info_storage)
            # print(len(server.info_storage))
            # exit(-1)

            start2 = time.time()
            info_path = os.path.join("server_models", args.dataset)
            info_path = os.path.join(info_path, args.algorithm + "_epoch_" + str(args.global_rounds) + ".pkl")
            with open(info_path, 'rb') as pkl_file:
                info_storage = pickle.load(pkl_file)

            server = server_back
            # 打印读取的数据
            server.info_storage = info_storage
            server.adaptive_recover(start=start2)
            
        elif args.algorithm == "Retrain":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedEraser(args, i)
            
            server.select_unlearned_clients()
            # server.train()
            server.retrain()

        elif args.algorithm == "FedHydra":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedHydra(args, i)

            # 客户端选择
            server.select_unlearned_clients()
            server_back = copy.deepcopy(server)
            # server.eraser_trans_hydra_server()
            # server.check_model()
            # server.eraser_trans_hydra_clients()
            # exit(-1)
            server = server_back
            start2 = time.time()
            server.unlearning(start=start2)
            
        else:
            raise NotImplementedError

        now = time.time()
        time_list.append(now-start)

        # 时间列表不确定到底会有几个，注释掉一些二者是等价的，但是以防万一单独补充一个
        print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.\n")
        print(f"\nTotal time cost: {round((now-start), 2)}s.\n")

        args.verify_unlearn = True
        if args.verify_unlearn:

            # 此处进行了校准时的投毒攻击，因此只需要记录ASR和ACC即可
            if args.unlearn_attack:
                # 后门模型
                path = os.path.join('server_models', args.dataset,
                                                             args.algorithm+"_" +args.unlearn_attack_method + "_unlearning" + ".pt")
                print(path)
                server.global_model = torch.load(path)
                print("This is attack model acc: ")
                server.server_metrics()
                server.asr_metrics()

            else:
                # 此处是正常的遗忘算法对比，进行ACC和MIA的比较
                print("-" * 20 + " Membership Attack " + "-" * 20)
                print("No attack, record results if the algorithm runs normally")
                # 正常FL
                server.global_model = torch.load(os.path.join('server_models', args.dataset,
                                                                              "Crab" + "_epoch_" + str(
                                                                                  args.global_rounds) + ".pt"))
                print("This is FL model acc: ")
                server.server_metrics()
                # server.asr_metrics()

                temp_model = copy.deepcopy(server.global_model)
                # 重训FL
                if args.algorithm == 'Retrain':
                    server.retrain_global_model = torch.load(os.path.join('server_models', args.dataset,
                                                                              "Retrain" + "_unlearning"  + ".pt"))

                # 遗忘模型
                server.eraser_global_model = torch.load(os.path.join('server_models', args.dataset,
                                                              args.algorithm+ "_unlearning" + ".pt"))

                # server.eraser_global_model = torch.load(os.path.join('server_models', args.dataset,
                #                                              args.algorithm+ args.unlearn_attack_method + "_unlearning" + ".pt"))

                print("- "*50)
                print("\nThis is unlearning model acc: ")
                print(os.path.join('server_models', args.dataset, args.algorithm+ "_unlearning" + ".pt"))
                server.global_model = server.eraser_global_model
                server.server_metrics()
                print("- "*50)
                server.global_model = temp_model
                # exit(-1)
                server.eraser_global_model.eval()

                server.MIA_metrics_lzp()

    # model_path = os.path.join("server_models", args.dataset)
    # server_path = os.path.join(model_path, args.algorithm + "_epoch_" + str(args.global_rounds) + ".pt")
    # # print(server_path)
    # assert (os.path.exists(server_path))
    # # 导入旧的全局模型
    # old_GM = torch.load(server_path)
    # client_loaders = list()


    # print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.\n")


        # server.MIA_metrics()
    

    print("All done!")
    print("\n"*5)



if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    
    # general
    parser.add_argument('-go', "--goal", type=str, default="Federated Unlearning Experiments", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="svhn")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=256)  # -> 256
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=True)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=50)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    
    # unlearning settings
    parser.add_argument('-algo', "--algorithm", type=str, default="Crab", choices=["Retrain", "FedEraser","Crab","FedHydra"],
                        help="How to unlearn the target clients")
    parser.add_argument('-verify', "--verify_unlearn", action='store_true',
                        help="Whether use the MIA to verify the unlearn effectiveness")
    parser.add_argument('-Prounds', "--select_round_ratio", type=float, default=0.6)
    parser.add_argument('-Xclients', "--select_client_ratio", type=float, default=0.7)
    
    # robust aggregation schemes
    parser.add_argument('-robust', "--robust_aggregation_schemes", type=str, default="FedAvg",
                        choices=["TrimmedMean", "Median", "Krum"], help="The aggregation schemes using when calculating the server parameters")
    parser.add_argument("--trimmed_clients_num", type=int, default=2, 
                        help="The number of clients will be trimmed. Calculated by each dimensions.")
    
    # Crab settings
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-unlearn', "--unlearn_clients_number", type=int, default=1,
                        help="Total number of unlearn clients")


    parser.add_argument('-un_atk', "--unlearn_attack_method", type=str, default="dba", choices=["modelre", "lie", "dba"],
                        help="How to unlearn the target clients")
    """
    Extra setting
    1、遗忘攻击数量
    2、是否遗忘攻击
    3、ModelRe的α
    4、投毒比例
    """
    parser.add_argument('-unlearn_attack_number', "--unlearn_attack_number", type=int, default=2,
                        help="Total number of unlearn attack clients")
    parser.add_argument('-unlearn_backdoor', '--unlearn_attack', action='store_true',
                    help="Whether to inject backdoor attack towards the target clients")
    parser.add_argument('-backdoor_ratio', "--backdoor_ratio", type=float, default=1,
                        help="")

    # LIE能做变化的几个参数
    parser.add_argument('-lie_alpha', "--lie_alpha_loss", type=float, default=0.8,
                        help="")
    parser.add_argument('-num_std', "--num_std", type=float, default=0.5, # 原文自选0.5 1 2
                        help="")
    # ModelRe
    parser.add_argument('-ModelRe', '--modelre', action='store_true',
                    help="")
    parser.add_argument('-Clip_rate', '--clip_rate', type=float, default=0.5,
                    help="")
    parser.add_argument('-alpha', "--alpha_loss", type=float, default=0.8,
                        help="")
    # LIE,原clamp
    parser.add_argument('-LIE', '--clamp_to_little_range', action='store_true',
                        help="whether to further clamp the updated parameters to little range based on malicous and benign params so that can circumvent defenses. From paper 'A Little Is Enough: Circumventing Defenses For Distributed Learning'.")
    # DBA
    parser.add_argument('-DBA', '--dba', action='store_true',
                    help="")
    parser.add_argument('-DBA_Clip_rate', '--dba_clip_rate', type=float, default=0.5,
                    help="")


    # attack setting
    # backdoor
    parser.add_argument('-backdoor', '--backdoor_attack', action='store_true', 
                    help="Whether to inject backdoor attack towards the target clients")
    parser.add_argument('-sar', '--start_attack_round', type=int,default=5,
                    help="Round of starting attack")

    # 有关trigger部分在dataset_utils类里调
    # default=4
    parser.add_argument('--trigger_size', type=int, default=15,
                        help="Size of injected trigger")
    parser.add_argument('--label_inject_mode', type=str, default="Fix", choices=["Fix", "Random", "Exclusive"], 
                        help="Random: assign tampered label randomly to each original label. Exclusive: perturb all the data with specific label and trigger.")
    parser.add_argument('--tampered_label', type=int, default=2,
                        help="Tamper label that corresponds to the sample injected the backdoor trigger. Must set '--label_inject_mode' to Fix")
    # trim
    parser.add_argument('-trim', '--trim_attack', action='store_true', 
                    help="Whether to execute trim attack towards the target clients")
    parser.add_argument('--trim_percentage', type=int, default=20,
                        help="Percentage of execute trim attack towards the target clients")
    
    # trivial
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-fte', "--fine_tuning_epoch", type=int, default=0)
    
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    if args.dataset == 'Cifar10' or args.dataset == 'cifar10':
        args.local_learning_rate = 0.05
        args.model = 'cnn'
        args.batch_size = 256
        args.clip_rate = 1.4

    elif args.dataset == 'fmnist':
        args.local_learning_rate = 0.01
        args.model = 'cnn'
        args.batch_size = 256
        args.clip_rate = 1

    elif args.dataset == 'svhn':
        args.local_learning_rate = 0.01
        args.model = 'mobilenet_v2'
        args.batch_size = 256
        args.clip_rate = 1.1

    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    log_path = ".\log_testfull\\"+args.dataset+"_"+args.algorithm+"_"+args.unlearn_attack_method +".log"
    log_path = ".\log_normal\\"+args.dataset+"_"+args.algorithm+".log"
    sys.stdout = Logger(log_path)
    print(datetime.datetime.now())
    print("=" * 50)

    run(args)

    