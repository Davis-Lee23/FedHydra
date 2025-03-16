import pickle

import torch
import torch.nn.functional as F
import os
import numpy as np
import h5py
import copy
import time
import random
import json

from pprint import pprint

from clientBase import clientAVG
from dataset_utils import read_client_data
from serverEraser import FedEraser


class Crab(FedEraser):
    def __init__(self, args, times):
        super().__init__(args, times)

        # 'client_selection': []
        self.old_GM = None
        self.info_storage = {}
        self.new_CM = []
        self.P_rounds = args.select_round_ratio
        self.X_clients = args.select_client_ratio

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        if self.new_CM != []:
            for c in self.new_CM:
                ct, ns, auc = c.test_metrics()
                tot_correct.append(ct * 1.0)
                tot_auc.append(auc * ns)
                num_samples.append(ns)
            ids = [c.id for c in self.new_CM]
        else:
            for c in self.remaining_clients:
                ct, ns, auc = c.test_metrics()
                tot_correct.append(ct * 1.0)
                tot_auc.append(auc * ns)
                num_samples.append(ns)
            ids = [c.id for c in self.remaining_clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        ter = []

        if self.backdoor_attack:
            asr = []
            num_samples_bad = []
            losses_bad = []
            for c in self.unlearn_clients:
                casr, cl, ns = c.asr_metrics(self.global_model)
                num_samples_bad.append(ns)
                losses_bad.append(cl * 1.0)
                asr.append(casr)

            num_samples_good = []
            losses_good = []
            for c in self.remaining_clients:
                _, cl, ns = c.train_metrics()
                num_samples_good.append(ns)
                losses_good.append(cl * 1.0)

            print(f"Each clients have {ns} samples")
            print(f"For each clients, the attack success numbers: {asr}")
            return _, num_samples_bad, losses_bad, asr, num_samples_good, losses_good

        if self.new_CM != []:
            for c in self.new_CM:
                cter, cl, ns = c.train_metrics()
                num_samples.append(ns)
                losses.append(cl * 1.0)
                ter.append(cter * 1.0)

            ids = [c.id for c in self.new_CM]
        else:
            for c in self.remaining_clients:
                cter, cl, ns = c.train_metrics()
                num_samples.append(ns)
                losses.append(cl * 1.0)
                ter.append(cter * 1.0)

            ids = [c.id for c in self.remaining_clients]

        return ids, num_samples, losses
    
    # def evaluate(self, acc=None, loss=None):
    #     stats = self.test_metrics()
    #     stats_train = self.train_metrics()

    #     test_acc = sum(stats[2])*1.0 / sum(stats[1])
    #     test_auc = sum(stats[3])*1.0 / sum(stats[1])
    #     if self.backdoor_attack:
    #         train_loss = sum(stats_train[5])*1.0 / sum(stats_train[4])
    #         asr = sum(stats_train[3])*1.0 / sum(stats_train[1])
    #     else:
    #         train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
    #     accs = [a / n for a, n in zip(stats[2], stats[1])]
    #     aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
    #     if acc == None:
    #         self.rs_test_acc.append(test_acc)
    #     else:
    #         acc.append(test_acc)
        
    #     if loss == None:
    #         self.rs_train_loss.append(train_loss)
    #     else:
    #         loss.append(train_loss)

    #     print("Averaged Train Loss: {:.4f}".format(train_loss))
    #     if self.backdoor_attack:
    #         print("Averaged Attack success rate: {:.4f}".format(asr))
    #     print("Averaged Test Accurancy: {:.4f}".format(test_acc))
    #     print("Averaged Test AUC: {:.4f}".format(test_auc))
    #     print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
    #     print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        
    #     return train_loss, test_acc
    
    
    def model_to_traj(self, GM_list):
        """输入 GM list 返回去除 dict keys 的纯 tensor

        Args:
            GM_list (list of NN)

        Returns:
            list of tensor: GM 随着epoch的trajectory
        """
        traj = []
        for model in GM_list:
            timestamp = []
            timestamp.extend([p.detach().clone() for p in model.parameters()])
            # print(sum(p.numel() for p in model.parameters()))
            traj.append(timestamp)
        return traj
    
    def select_round(self, start_epoch, GM_list):
        """选取哪些 epoch 最重要

        Args:
            start_epoch (int): 每个 buffer window 的起点
            GM_list (list of NN): [self.global_model]

        Returns:
            list: 返回需要选取的 epoch 轮数
        """
        k = int(len(GM_list) * self.P_rounds) 
        GM_trajectory = self.model_to_traj(GM_list)
        prior = GM_trajectory[0]
        kl_list = []
        if len(GM_trajectory) < 2:
            return [start_epoch]
        for now_traj in GM_trajectory[1:]:
            kl = 0
            for module, prior_module in zip(now_traj, prior):
                log_x = F.log_softmax(module, dim=-1)
                y = F.softmax(prior_module, dim=-1)
                kl += F.kl_div(log_x, y, reduction='sum')
            kl_list.append(kl.cpu().item())
            prior = now_traj
        print("KL Divergence between each epoch's global model:", kl_list)
        # print(self.info_storage)
        kl_list = np.array(kl_list)
        sel_round = np.argsort(kl_list)[::-1]
        return (sel_round[:k] + start_epoch).tolist()
        
        
    def select_client_in_round(self, round, GM_list, start_epoch):
        """在每一个 epoch 中选取需要的 clients

        Args:
            round (int): 当前epoch
            GM_list (list of NN): _description_
            start_epoch (int): buffer window 起始epoch

        Returns:
            list: 返回需要选取的 client id
        """
        CM = self.load_client_model(round)
        CM_list = [c.model for c in CM]
        CM_list = self.model_to_traj(CM_list)
        k = int(len(CM) * self.X_clients)
        target_GM = GM_list[round - start_epoch] # GM_list 的下标是根据每一个 buffer window 从0开始索引的
        target_GM = [p.detach().clone() for p in target_GM.parameters()]

        similarity = []
        for client in CM_list:  
            cos_sim = [] 
            for g_module, c_module in zip(target_GM, client):
                if len(g_module.shape) > 1:
                    cos = torch.cosine_similarity(g_module, c_module)
                    # TODO: 是否考虑 abs(cos)
                    cos_sim.append(torch.mean(cos).cpu().item())
            similarity.append(np.mean(cos_sim))
        sel_client = np.argsort(similarity)[::-1]
        sel_client = sel_client[:k].tolist()
        
        ans_clients = []
        ans_id = []
        for sel in sel_client:
            ans_id.append(CM[sel].id)
            # ans_clients.append(CM[sel])

        return ans_id


    def train_with_select(self):
        # print(self.global_model.state_dict()['base.conv1.0.weight'][0])
        alpha = 0.1
        GM_list = []
        start_epoch = 0
        
        if self.backdoor_attack:
            print(f"Inject backdoor to target {self.idx_}.")
        elif self.trim_attack:
            print(f"Execute trim attack target {self.idx_}.")
        
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            self.save_each_round_global_model(i)
            

            if i%self.eval_gap == 0:
                print(f"\n-------------FL Round number: {i}-------------")
                print("\nEvaluate global model")
                self.server_metrics()
                print("\n")
                # 这个evaluate会影响到整个程序的运行
                train_loss, _ = self.evaluate()
                # print("\n")

            if i == 0:
                start_loss = copy.deepcopy(train_loss)
            else:
                GM_list.append(copy.deepcopy(self.global_model))

            if train_loss < start_loss * (1 - alpha) or i == self.global_rounds:
                # if i%5==0:
                print("*****")
                rounds = self.select_round(start_epoch, GM_list)
                print("pick rounds: ", rounds)
                for round in rounds:
                    clients_id = self.select_client_in_round(round, GM_list, start_epoch)
                    print(f"select clients from epoch {round}: {clients_id}")
                    self.info_storage[int(round)] = clients_id

                # for client in clients_id:
                # gradient = self.select_grad_in_client()

                start_loss = copy.deepcopy(train_loss)
                GM_list = []
                start_epoch = i
                # self.info_storage = ...
                
            for client in self.selected_clients:
                client.train()
            
            self.save_client_model(i)

            if self.args.robust_aggregation_schemes == "FedAvg":
                self.receive_models()
                self.aggregate_parameters()
            elif self.args.robust_aggregation_schemes == "TrimmedMean":
                self.aggregation_trimmed_mean(unlearning_stage=False, trimmed_clients_num=self.args.trimmed_clients_num)
            elif self.args.robust_aggregation_schemes == "Median":
                self.aggregation_median(unlearning_stage=False)
            elif self.args.robust_aggregation_schemes == "Krum":
                self.aggregation_Krum(unlearning_stage=False)
            

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])
            

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))


        print('write the select information into the txt...')
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        path = os.path.join(self.save_folder_name, "server_select_info" + ".txt")
        self.info_storage = dict(sorted(self.info_storage.items()))
        with open(path, 'w') as storage: 
            storage.write(json.dumps(self.info_storage))
            
        self.save_results()
        # self.save_global_model()
        # self.server_metrics()
        self.FL_global_model = copy.deepcopy(self.global_model)


        # 需要在最后保存一下需校准的轮次
        print("save the info storage")
        info_path = os.path.join("server_models", self.dataset)
        info_path = os.path.join(info_path, self.algorithm + "_epoch_" + str(self.global_rounds) + ".pkl")
        with open(info_path, 'wb') as pkl_file:
            pickle.dump(self.info_storage, pkl_file)

    def train(self):
        # print(self.global_model.state_dict()['base.conv1.0.weight'][0])
        if self.backdoor_attack:
            print(f"Inject backdoor to target {self.idx_}.")
        elif self.trim_attack:
            print(f"Execute trim attack target {self.idx_}.")

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            # 模型选择 —— 发送模型 —— 保存模型
            self.selected_clients = self.select_clients()
            self.send_models()
            self.save_each_round_global_model(i)

            if i % self.eval_gap == 0:
                print(f"\n-------------FL Round number: {i}-------------")
                print("\nEvaluate global model")
                # 实际上测评的都是上一轮的
                # self.evaluate()
                print("\n")
                self.server_metrics()
            # print(self.selected_clients)
            for client in self.selected_clients:
                # if client in self.unlearn_clients and self.backdoor_attack:
                #     if i > 5:
                #         client.train(create_trigger=True)
                #     else:
                #         client.train(create_trigger=False)
                # elif client in self.unlearn_clients and self.trim_attack:
                #     client.train(trim_attack=True)
                # else:
                client.train()

            self.save_client_model(i)

            # 聚合模型
            if self.args.robust_aggregation_schemes == "FedAvg":
                self.receive_models()
                self.aggregate_parameters()
            elif self.args.robust_aggregation_schemes == "TrimmedMean":
                self.aggregation_trimmed_mean(unlearning_stage=False, trimmed_clients_num=self.args.trimmed_clients_num)
            elif self.args.robust_aggregation_schemes == "Median":
                self.aggregation_median(unlearning_stage=False)
            elif self.args.robust_aggregation_schemes == "Krum":
                self.aggregation_Krum(unlearning_stage=False)

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        # print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        # self.save_global_model()
        # self.server_metrics()
        print("*" * 20)

        self.FL_global_model = copy.deepcopy(self.global_model)

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def adaptive_recover(self,start=None):
        print("***************", self.unlearn_clients)
        
        model_path = os.path.join("server_models", self.dataset)
        # all_clients_class = self.load_client_model(100)
        # for client in self.clients:
        #     for c in all_clients_class:
        #         if client.id == c.id:
        #             client.set_parameters(c.model)

        io_time = 0

        # 不能直接unlearn的原因就是直接导入没有info_storage
        # 事实上Crab和Eraser的逻辑差别太大了
        for global_round, select_clients_in_round in self.info_storage.items():
            temp_time1 = time.time()

            server_path = os.path.join(model_path, self.algorithm + "_epoch_" + str(global_round) + ".pt")
            # print(server_path)
            self.old_GM = torch.load(server_path)
            
            select_clients_in_round = [id for id in select_clients_in_round if id in self.idr_]
            
            all_clients_class = self.load_client_model(global_round)
            temp_time2 = time.time()
            io_time += temp_time2 - temp_time1
            # 此处copy一份remaining_clients，因为recovery的时候可能遗忘的id包含贡献度大的那个client
            self.old_clients = copy.deepcopy(self.remaining_clients)  
            self.old_CM = []
            for  client in self.old_clients:
                for c in all_clients_class:
                    if client.id == c.id:

                        if self.args.dataset == 'svhn':
                            client.model = copy.deepcopy(c.model)
                        else:
                            client.set_parameters(c.model)

                        # print(" /// ",c.model.state_dict()['base.conv1.0.weight'][0])
                if client.id in select_clients_in_round:
                    self.old_CM.append(client)
            print([c.id for c in self.old_CM])
            
            self.old_clients = copy.deepcopy(self.old_CM)
        
            # 得到旧的训练后的CM
            assert (len(self.old_CM) <= len(select_clients_in_round))

            print(f"\n-------------Crab Round number: {global_round}-------------")

            dba = 0
            for client in self.old_clients:
                client.set_parameters(self.old_GM)

                if self.unlearn_attack:
                    if client.id in self.ida_ and global_round > self.args.start_attack_round and self.unlearn_attack_method == 'dba':
                        print("DBA Attack First")
                        client.local_epochs = 1
                        client.train_one_step(trigger=True,dba=dba)
                        dba +=1

                    elif client.id in self.ida_ and global_round > self.args.start_attack_round and self.unlearn_attack_method == 'lie':
                        print("LIE Attack First")
                        client.local_epochs = 1
                        client.train_one_step(trigger=True)
                        # client.train_one_step()

                    elif client.id in self.ida_ and global_round > self.args.start_attack_round and self.unlearn_attack_method == 'modelre':
                        print("ModelRe Attack First")
                        if self.args.dataset == 'fmnist':
                            client.local_epochs = 1
                            client.train(trigger=True,double =True)


                else:
                    client.train_one_step()
                # client.train_one_step(trigger=False)


            if self.args.robust_aggregation_schemes == "FedAvg":
                self.receive_retrained_models(self.old_clients)
                self.aggregate_parameters()
            elif self.args.robust_aggregation_schemes == "TrimmedMean":
                self.aggregation_trimmed_mean(unlearning_stage=True, trimmed_clients_num=self.args.trimmed_clients_num, existing_clients=self.old_clients)
            elif self.args.robust_aggregation_schemes == "Median":
                self.aggregation_median(unlearning_stage=True, existing_clients=self.old_clients)
            elif self.args.robust_aggregation_schemes == "Krum":
                self.aggregation_Krum(unlearning_stage=True, existing_clients=self.old_clients)
                
            self.new_GM = copy.deepcopy(self.global_model)

            # print("New_GM before calibration ***:::", self.new_GM.state_dict()['base.conv1.0.weight'][0])
            
            # 得到新的CM
            for client in self.old_clients:
                dba = 0
                client.set_parameters(self.new_GM)
                if self.unlearn_attack:
                    if client.id in self.ida_ and global_round > self.args.start_attack_round and self.unlearn_attack_method == 'dba':
                        print("DBA Attack Second")
                        client.local_epochs = 1
                        client.train_one_step(trigger=True,dba=dba)
                        dba+=1

                    elif client.id in self.ida_ and global_round > self.args.start_attack_round and self.unlearn_attack_method == 'lie':
                        print("LIE Attack Second")
                        client.local_epochs = 1
                        client.train_one_step(trigger=True)
                        # client.train_one_step()

                    elif client.id in self.ida_ and global_round > self.args.start_attack_round and self.unlearn_attack_method == 'modelre':
                        print("ModelRe Attack Second")
                        if self.args.dataset == 'fmnist':
                            client.local_epochs = 1
                            client.train(trigger=True)

                else:
                    client.train_one_step()

                # client.train_one_step(trigger=False)

            self.new_CM = copy.deepcopy(self.old_clients)
            
            # 开始校准
            self.new_GM = self.unlearning_step_once(self.old_CM, self.new_CM, self.old_GM, self.new_GM)
            self.global_model = copy.deepcopy(self.new_GM)
            self.server_metrics()
            if self.unlearn_attack:
                self.asr_metrics()
            # self.send_models()
            # self.evaluate()
            # print("new GM after calibration ***:::", self.new_GM.state_dict()['base.conv1.0.weight'][0])
        
        print(f"\n-------------After Crab-------------")
        print("\nEvaluate Eraser global model")
        # self.server_metrics()
        self.global_model = copy.deepcopy(self.new_GM)
        # self.send_models()
        # self.evaluate()
        self.eraser_global_model = copy.deepcopy(self.new_GM)
        now = time.time()
        print(f"\nSingle unlearning time cost: {round((now-start), 2)}s.\n")
        print(f"\nIO time cost: {round(io_time, 2)}s.\n")
        # self.new_CM = []
        self.save_unlearning_model()
        self.server_metrics()

