# 最后一层分类器即FC与FC与它的bias
import math
from distutils.command.clean import clean

import torch
import os
import copy
import time
import random


from pprint import pprint

from dataset_utils import read_client_data
from clientBase import clientAVG
from serverBase import Server
import torch.nn.functional as F


class FedHydra(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.new_CM = None
        self.new_GM = None
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.unlearn_clients_number = args.unlearn_clients_number
        self.unlearn_attack_number =args.unlearn_attack_number

    def train(self):
        # print(self.global_model.state_dict()['base.conv1.0.weight'][0])
        if self.backdoor_attack:
            print(f"Inject backdoor to target {self.idx_}.")
        elif self.trim_attack:
            print(f"Execute trim attack target {self.idx_}.")

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            # 服务器发送model给客户端
            self.send_models()
            self.save_each_round_global_model(i)

            if i % self.eval_gap == 0:
                print(f"\n-------------FL Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                print("\n")
                # self.server_metrics()
            # print(self.selected_clients)
            for client in self.selected_clients:
                if client in self.unlearn_clients and self.backdoor_attack:
                    client.train(create_trigger=True)
                elif client in self.unlearn_clients and self.trim_attack:
                    client.train(trim_attack=True)
                else:
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
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        # self.save_global_model()
        # self.server_metrics()
        self.FL_global_model = copy.deepcopy(self.global_model)

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def check_model(self):
        model_path = os.path.join("server_models", self.dataset)
        server_path = os.path.join(model_path, f"FedHydra_epoch_100.pt")
        model = torch.load(server_path)
        print(1)

    def eraser_trans_hydra_server(self):
        model_path = os.path.join("server_models", self.dataset)
        self.algorithm = "Crab"
        new_algorithm = "FedHydra"
        # 创建一个字典用于保存选中的参数
        head_parameters = {}

        # 全局模型转换
        # 遍历从 epoch 0 到 epoch 100 的模型
        for epoch in range(101):  # 包括 0 到 100
            server_path = os.path.join(model_path, f"Crab_epoch_{epoch}.pt")

            # 检查模型文件是否存在
            if os.path.exists(server_path):
                model = torch.load(server_path)
                # print(id(model))

                # 获取每层的名称和值
                for name, param in model.named_parameters():
                    if 'head' in name:  # 只保留包含 'head' 的参数
                        # print(name)
                        head_parameters[name] = param

            new_model_path = os.path.join(model_path, f"{new_algorithm}_epoch_{epoch}.pt")
            # print(new_model_path)
            torch.save(head_parameters, new_model_path)
            print("Trans Hydra Server Model")

        """
        只要获取head的足矣
        Layer: head.weight
        Layer: head.bias
        """

    def eraser_trans_hydra_clients(self):
        model_path = os.path.join("clients_models", self.dataset)
        self.algorithm = "Crab"
        new_algorithm = "FedHydra"

        # 全局模型转换
        # 遍历从 epoch 0 到 epoch 100 的模型
        for epoch in range(101):  # 包括 0 到 100
            clients_path = os.path.join(model_path, f"Crab_epoch_{epoch}.pt")

            # 检查模型文件是否存在
            if os.path.exists(clients_path):
                models = torch.load(clients_path)
                # print(id(model))

            # 创建一个字典用于保存选中的参数
            head_parameters = {}

            for model in models:
                model_id = model.id
                for name, param in model.model.named_parameters():
                        if 'head' in name:  # 只保留包含 'head' 的参数
                            head_parameters[(model_id, name)] = param

            new_model_path = os.path.join(model_path, f"{new_algorithm}_epoch_{epoch}.pt")
            # print(head_parameters)
            torch.save(head_parameters, new_model_path)
            print("Trans Hydra Client Model: "+str(epoch))


    def unlearning(self,start=None):
        print("***************", self.unlearn_clients)

        model_path = os.path.join("server_models", self.dataset)
        # 先导入正常训练结束的模型
        self.trained_model = torch.load(os.path.join(model_path, "Crab" + "_epoch_"+str(self.args.global_rounds) + ".pt"))

        io_time = 0

        for epoch in range(0, self.global_rounds+1, 2):
            # print("\n")
            # self.evaluate()

            # epoch=self.global_rounds 这一段和下面的用来测试模型到底有没有成功导入，经测试OK
            # epoch = self.global_rounds

            # Hydra和Eraser用的model是一样的
            # self.algorithm = "FedEraser"
            temp_time1 = time.time()

            server_path = os.path.join(model_path, self.algorithm + "_epoch_" + str(epoch) + ".pt")
            assert (os.path.exists(server_path))

            # 导入旧全局,只修改head，base不变
            self.old_GM = torch.load(server_path)
            self.trained_model.head.weight = self.old_GM['head.weight']
            self.trained_model.head.bias = self.old_GM['head.bias']
            self.old_GM = copy.deepcopy(self.trained_model)


            # print("old GM ***:::", self.old_GM.state_dict()['base.conv1.0.weight'][0])

            # 导入旧的head
            all_clients_class = self.load_client_model(epoch)
            temp_time2 = time.time()
            io_time += temp_time2 - temp_time1
            head_weight = 0
            head_bias = 0

            for i, client in enumerate(self.remaining_clients):
                for c in all_clients_class:
                    if client.id == c[0]:
                        if c[1] == 'head.weight':
                            # 取值all_clients_class[c]
                            # print(client.model.state_dict()['head.weight'])
                            # client.model.state_dict()['head.weight']=  all_clients_class[c]
                            client.model.state_dict()['head.weight'].copy_(all_clients_class[c])
                            # print(client.model.state_dict()['head.weight'])
                        elif c[1] == 'head.bias':
                            # client.model.state_dict()['head.bias']= all_clients_class[c]
                            client.model.state_dict()['head.bias'].copy_(all_clients_class[c])
                            # print(client.model.state_dict()['head.bias'])

            self.old_CM = copy.deepcopy(self.remaining_clients)

            # print(io_time)

            # 产生第一次的new_GM
            if epoch == 0:
                for i, client in enumerate(self.old_CM):
                    head_weight += client.model.state_dict()['head.weight']
                    head_bias += client.model.state_dict()['head.bias']

                head_weight /= len(self.old_CM)
                head_bias /= len(self.old_CM)

                """
                仅对分类器的处理
                """
                # print(                self.global_model.state_dict()['head.weight'])
                self.global_model = copy.deepcopy(self.trained_model)
                self.global_model.state_dict()['head.weight'].copy_(head_weight)
                # print(                self.global_model.state_dict()['head.weight'])
                self.global_model.state_dict()['head.bias'].copy_(head_bias)
                self.new_GM = copy.deepcopy(self.global_model)

                for c in self.unlearn_attack_clients:
                    c.local_epochs *= 1
                continue


            print(f"\n-------------FedHydra Round number: {epoch}-------------")
            assert (len(self.remaining_clients) > 0)

            dba = 0
            # 得到新的CM，进行一步训练
            for client in self.remaining_clients:
                client.set_parameters(self.new_GM)
                if self.unlearn_attack:
                    if client.id in self.ida_ and epoch>self.args.start_attack_round:
                        if self.unlearn_attack_method == 'lie':
                            if self.dataset == 'fmnist':
                                print("Hydra FMNIST LIE")
                                client.local_epochs = 1
                                client.train(create_trigger=True)
                            elif self.dataset == 'cifar10':
                                print("Hydra CIFAR LIE")
                                client.local_epochs = 1
                                client.train(create_trigger=True,double=True)

                            elif self.dataset == 'svhn':
                                print("Hydra SVHN LIE")
                                client.local_epochs = 1
                                client.train(create_trigger=True, double=True)

                        elif  self.unlearn_attack_method == 'dba':
                            if self.dataset == 'fmnist':
                                print("Hydra FMNIST DBA")
                                client.local_epochs = 1
                                client.train(create_trigger=True, dba=dba)
                            elif self.dataset == 'cifar10':
                                print("Hydra CIFAR10 DBA")
                                client.local_epochs = 1
                                client.train(create_trigger=True, dba=dba, double=True)

                            elif self.dataset == 'svhn':
                                print("Hydra SVHN DBA")
                                client.local_epochs = 1
                                client.train(create_trigger=True, dba=dba, double=True)
                            dba+=1

                        else:
                            if self.unlearn_attack_method=='modelre' and self.dataset == 'svhn':
                                print("Hydra SVHN MODELRE")
                                client.local_epochs = 1
                                client.train(create_trigger=True,double=True)
                            elif self.unlearn_attack_method=='modelre' and self.dataset == 'cifar10':
                                print("Hydra CIFAR MODELRE")
                                client.local_epochs = 1
                                client.train(create_trigger=True,double=True)
                            else:
                                print("Hydra FMNIST MODELRE")
                                client.local_epochs = 1
                                client.train(create_trigger=True)
                    else:
                        client.train()

                else:
                    client.train()

            self.new_CM = copy.deepcopy(self.remaining_clients)

            """
            余弦操作
            """
            old_client_models = copy.deepcopy(self.old_CM)
            new_client_models = copy.deepcopy(self.new_CM)

            similarity_scores = []

            for i in range(len(old_client_models)):
                # print(old_client_models[i].id)
                weight1 = old_client_models[i].model.state_dict()['head.weight']
                weight2 = new_client_models[i].model.state_dict()['head.weight']

                # 计算余弦相似度
                cos_sim = F.cosine_similarity(weight1, weight2, dim=1)
                # print(cos_sim)

                # 计算平均相似度
                average_similarity = cos_sim.mean().item()
                similarity_scores.append(average_similarity)

            # print(similarity_scores)
            # 将相似度得分与模型索引配对
            model_similarity = list(enumerate(similarity_scores))

            # 按相似度得分排序，从低到高
            model_similarity.sort(key=lambda x: x[1])

            # 计算需要去掉的模型数量，向上取整
            num_to_remove = math.ceil(len(old_client_models) / 5)  # 去掉最不相近的 20%

            # 获取最不相近模型的索引
            indices_to_remove = [model_similarity[i][0] for i in range(num_to_remove)]

            # 创建新的模型列表，去掉最不相近的模型
            filtered_old_client_models = [old_client_models[i] for i in range(len(old_client_models)) if
                                          i not in indices_to_remove]
            filtered_new_client_models = [new_client_models[i] for i in range(len(new_client_models)) if
                                          i not in indices_to_remove]

            print(f'Removed models at indices: {indices_to_remove}')
            print(f'Filtered old client models: {len(filtered_old_client_models)}')
            print(f'Filtered new client models: {len(filtered_new_client_models)}')

            # old_client_models = filtered_old_client_models
            # new_client_models = filtered_new_client_models
            self.old_CM = filtered_old_client_models
            self.new_CM = filtered_new_client_models

            # 聚合,在此处global_model被修改了
            if self.args.robust_aggregation_schemes == "FedAvg":
                # 这里receive了new_CM而不是remaining，代表已经过滤
                self.receive_retrained_models(self.new_CM)
                # self.aggregate_parameters()
                self.aggregate_heads()
                # exit(-1)
            elif self.args.robust_aggregation_schemes == "TrimmedMean":
                self.aggregation_trimmed_mean(unlearning_stage=True, trimmed_clients_num=self.args.trimmed_clients_num,
                                              existing_clients=self.remaining_clients)
            elif self.args.robust_aggregation_schemes == "Median":
                self.aggregation_median(unlearning_stage=True, existing_clients=self.remaining_clients)
            elif self.args.robust_aggregation_schemes == "Krum":
                self.aggregation_Krum(unlearning_stage=True, existing_clients=self.remaining_clients)

            # print("*"*20)
            # self.server_metrics()
            self.new_GM = copy.deepcopy(self.global_model)
            # print("New_GM before calibration ***:::", self.new_GM.state_dict()['base.conv1.0.weight'][0])

            # 开始校准
            self.new_GM = self.unlearning_step_once_Hydra(self.old_CM, self.new_CM, self.old_GM, self.new_GM)
            # print("new GM after calibration ***:::", self.new_GM.state_dict()['base.conv1.0.weight'][0])
            self.global_model = copy.deepcopy(self.new_GM)
            # self.send_models()
            self.server_metrics()
            if self.unlearn_attack:
                self.asr_metrics()
            # exit(-1)

        print(f"\n-------------After FedHydra-------------")
        # 这里评估的也不是最终矫正的模型，因为没有修改
        print("\nEvaluate Eraser global model")
        # self.global_model = copy.deepcopy(self.new_GM)
        # self.server_metrics()
        # for i, client in enumerate(self.remaining_clients):
        #     for c in all_clients_class:
        #         if client.id == c.id:
        #             client.set_parameters(self.global_model)
        # self.evaluate()
        self.eraser_global_model = copy.deepcopy(self.new_GM)
        now = time.time()
        print(f"\nSingle unlearning time cost: {round((now-start), 2)}s.\n")
        print(f"\nIO time cost: {round(io_time, 2)}s.\n")
        self.server_metrics()
        self.save_unlearning_model()

    def unlearning_step_once_Hydra(self, old_client_models, new_client_models, global_model_before_forget,
                             global_model_after_forget):

        # cos余弦比较


        # exit(-1)
        # for i in range(len(old_client_models)):
        #     print("id:"+ str(old_client_models[i].id))
        #     weight1 = old_client_models[i].model.state_dict()['head.weight']
        #     weight2 = new_client_models[i].model.state_dict()['head.weight']
        #
        #     cos_sim = F.cosine_similarity(weight1, weight2, dim=1)
        #     print(cos_sim)
        #     # 计算平均相似度
        #     average_similarity = cos_sim.mean().item()
        #     print(average_similarity)
        #
        #     # print(num_to_remove)
        #     # cos_sim1 = F.cosine_similarity(weight1, weight2, dim=0)
        #     # cos_sim2 = F.cosine_similarity(bias1,bias2, dim=0)
        #
        #     # print(cos_sim1)
        #     # print(cos_sim2)
        #     print("\n")
        #


        # exit(-1)

        # exit(-1)

        """
        Parameters
        ----------
        old_client_models : list of client objects, list

        new_client_models : list of client objects, list

        global_model_before_forget : The old global model

        global_model_after_forget : The New global model


        Returns
        -------
        return_global_model : After one iteration, the new global model under the forgetting setting

        """
        old_param_update = dict()  # Model Params： oldCM - oldGM_t
        new_param_update = dict()  # Model Params： newCM - newGM_t

        new_global_model_state = global_model_after_forget.state_dict()  # newGM_t

        return_model_state = dict()  # newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||

        assert len(old_client_models) == len(new_client_models)

        for layer in global_model_before_forget.state_dict().keys():
            # print(layer)
            if layer == "head.weight" or layer == "head.bias":
                old_param_update[layer] = 0 * global_model_before_forget.state_dict()[layer]
                new_param_update[layer] = 0 * global_model_before_forget.state_dict()[layer]

                return_model_state[layer] = 0 * global_model_before_forget.state_dict()[layer]

                for ii in range(len(new_client_models)):
                    # print(new_client_models[ii].id)
                    old_param_update[layer] += old_client_models[ii].model.state_dict()[layer]
                    new_param_update[layer] += new_client_models[ii].model.state_dict()[layer]
                old_param_update[layer] /= (ii + 1)  # Model Params： oldCM
                new_param_update[layer] /= (ii + 1)  # Model Params： newCM

                old_param_update[layer] = old_param_update[layer] - global_model_before_forget.state_dict()[
                    layer]  # 参数： oldCM - oldGM_t
                new_param_update[layer] = new_param_update[layer] - global_model_after_forget.state_dict()[
                    layer]  # 参数： newCM - newGM_t

                step_length = torch.norm(old_param_update[layer])  # ||oldCM - oldGM_t||
                step_direction = new_param_update[layer] / torch.norm(
                    new_param_update[layer])  # (newCM - newGM_t)/||newCM - newGM_t||

                return_model_state[layer] = new_global_model_state[layer] + step_length * step_direction
            else:
                return_model_state[layer] = new_global_model_state[layer]

        # print("....", step_length, step_direction)
        # exit(-1)
        return_global_model = copy.deepcopy(global_model_after_forget)

        return_global_model.load_state_dict(return_model_state)

        return return_global_model



