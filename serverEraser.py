import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import wandb
from pprint import pprint

from dataset_utils import read_client_data
from clientBase import clientAVG
from serverBase import Server



class FedEraser(Server):
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
        # self.unlearn_attack = args.unlearn_attack


    def train(self):
        # print(self.global_model.state_dict()['base.conv1.0.weight'][0])
        if self.backdoor_attack:
            print(f"Inject backdoor to target {self.idx_}.")
        elif self.trim_attack:
            print(f"Execute trim attack target {self.idx_}.")
        
            
        for i in range(self.global_rounds+1):
            s_t = time.time()
            # 模型选择 —— 发送模型 —— 保存模型
            self.selected_clients = self.select_clients()
            self.send_models()
            self.save_each_round_global_model(i)

            if i%self.eval_gap == 0:
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
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        # print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

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
            
            
    def unlearning(self,start=None):
        print("***************", self.unlearn_clients)
        
        model_path = os.path.join("server_models", self.dataset)
        self.algorithm ="Crab"

        for epoch in range(0, self.global_rounds+1, 2):
            server_path = os.path.join(model_path, self.algorithm + "_epoch_" + str(epoch) + ".pt")
            assert (os.path.exists(server_path))
            # 导入旧的全局模型
            self.old_GM = torch.load(server_path)

            # self.evaluate()

            # print("old GM ***:::", self.old_GM.state_dict()['base.conv1.0.weight'][0])

            # 导入旧的本地模型
            all_clients_class = self.load_client_model(epoch)

            # 遍历所有剩余客户端
            for i, client in enumerate(self.remaining_clients):
                for c in all_clients_class:
                    if client.id == c.id:
                        # print(client.id)
                        client.set_parameters(c.model)
                        # print(" /// ",c.model.state_dict()['base.conv1.0.weight'][0])

            # 赋值old_CM
            self.old_CM = copy.deepcopy(self.remaining_clients)

            print(f"\n-------------FedEraser Round number: {epoch}-------------")
            print(server_path)
            # 产生第一次的new_GM
            if epoch == 0:
                weight = []
                for c in self.remaining_clients:
                    weight.append(c.train_samples)
                tot_sample = sum(weight)
                weight = [i / tot_sample for i in weight]
                # pprint(weight)
            
                for param in self.global_model.parameters():
                    param.data.zero_()
                for w, client in zip(weight, self.remaining_clients):
                    self.add_parameters(w, client.model)
                self.new_GM = copy.deepcopy(self.global_model)
                self.server_metrics()
                # 增加遗忘攻击客户端轮次
                if self.args.dataset != 'fmnist':
                    for c in self.remaining_clients:
                        if c in self.unlearn_attack_clients:
                            print("This is current dataset:" + str(self.args.dataset))
                            print("attack is ready !!!")
                            c.local_epochs = 1
                        else:
                            c.local_epochs =1
                else:
                    for c in self.unlearn_attack_clients:
                        print("This is current dataset:"+str(self.args.dataset))
                        print("attack is ready")
                # print(self.new_GM.state_dict()['base.conv1.0.weight'][0])
                
                continue


            # 不应该矫正之后再评估吗？这里评估的都是旧的model
            # self.global_model = copy.deepcopy(self.new_GM)
            # train_loss, test_acc = self.evaluate()
            # print({f'Train_loss/{self.algorithm}': train_loss})
            # print({f'Test_acc/{self.algorithm}': test_acc})
            # wandb.log({f'Train_loss/{self.algorithm}': train_loss}, step=epoch)
            # wandb.log({f'Test_acc/{self.algorithm}': test_acc}, step=epoch)
                
            # 得到新的CM，进行一步训练
            assert (len(self.remaining_clients) > 0)

            dba = 0
            # 得到新的CM，进行一步训练
            for client in self.remaining_clients:
                # 这一步本意是设置client的参数，但实际上send不就好了
                client.set_parameters(self.new_GM)

                if client.id in self.ida_ and epoch>20:
                # if client.id in self.ida_ :
                    print("Client "+ str(client.id)+" is attacking.")
                    # client.learning_rate = 0.0005
                    # client.optimizer = torch.optim.Adam(client.model.parameters(), lr=client.learning_rate)
                    # print(client.local_epochs)

                    if  self.unlearn_attack_method == 'dba':
                        if self.dataset == 'fmnist':
                            client.local_epochs = 1
                        elif self.dataset == 'cifar10':
                            # client.optimizer = torch.optim.SGD(client.model.parameters(), lr=client.learning_rate/2,
                            #                                  momentum=0.9)
                            client.local_epochs = 1

                        elif self.dataset == 'svhn':
                            client.local_epochs = 1

                        client.train(create_trigger=True,dba=dba)
                        dba+=1

                    else:
                        if self.unlearn_attack_method=='modelre' and self.dataset == 'svhn':
                            print("SVHN MODELRE")
                            client.local_epochs = 1
                            client.train(create_trigger=True,double=True)
                        elif self.unlearn_attack_method=='modelre' and self.dataset == 'cifar10':
                            print("CIFAR MODELRE")
                            client.local_epochs = 1
                            client.train(create_trigger=True,double=True)
                        else:
                            client.local_epochs = 2
                            client.train(create_trigger=True)
                        # client.train(create_trigger=True)
                    # client.train()
                else:
                    client.train()

                # client.train()


            # 获得训练之后的新client
            self.new_CM = copy.deepcopy(self.remaining_clients)
            
            # 聚合一次
            if self.args.robust_aggregation_schemes == "FedAvg":
                # 客户端发送参数给服务器
                self.receive_retrained_models(self.remaining_clients)
                # 聚合
                self.aggregate_parameters()
            elif self.args.robust_aggregation_schemes == "TrimmedMean":
                self.aggregation_trimmed_mean(unlearning_stage=True, trimmed_clients_num=self.args.trimmed_clients_num, existing_clients=self.remaining_clients)
            elif self.args.robust_aggregation_schemes == "Median":
                self.aggregation_median(unlearning_stage=True, existing_clients=self.remaining_clients)
            elif self.args.robust_aggregation_schemes == "Krum":
                self.aggregation_Krum(unlearning_stage=True, existing_clients=self.remaining_clients)


            # 聚合之后的新全局模型，深拷贝给new_GM
            self.new_GM = copy.deepcopy(self.global_model)
            # print("New_GM before calibration ***:::", self.new_GM.state_dict()['base.conv1.0.weight'][0])
            
            # 开始校准
            self.new_GM.train()

            # 校准，此时newGM和全局模型已经不一样了
            self.algorithm = 'FedEraser'
            self.new_GM = self.unlearning_step_once(self.old_CM, self.new_CM, self.old_GM, self.new_GM)
            self.algorithm = 'Crab'
            self.global_model = copy.deepcopy(self.new_GM)
            # self.global_model = copy.deepcopy(self.old_GM)
            # self.send_models()
            # self.evaluate()
            # if epoch>10:
            self.server_metrics()
            if self.unlearn_attack:
                self.asr_metrics()


            # print("new GM after calibration ***:::", self.new_GM.state_dict()['base.conv1.0.weight'][0])

        
        print(f"\n-------------After FedEraser-------------\n")
        # 这里评估的也不是最终矫正的模型，因为没有修改
        print("Evaluate Eraser global model")
        self.global_model = self.new_GM
        # self.server_metrics()
        # for i, client in enumerate(self.remaining_clients):
        #     for c in all_clients_class:
        #         if client.id == c.id:
        #             client.set_parameters(self.global_model)
        # self.send_models()
        # self.backdoor_attack = True
        # self.evaluate()
        # self.global_model = self.new_GM
        self.eraser_global_model = copy.deepcopy(self.new_GM)
        # self.global_model = copy.deepcopy(self.new_GM)
        # self.save_unlearning_model()

        now = time.time()
        print(f"\nSingle unlearning time cost: {round((now-start), 2)}s.\n")
        self.server_metrics()
        # train_loss, test_acc = self.evaluate()
        # self.eraser_global_model = copy.deepcopy(self.new_GM)
        self.algorithm = "FedEraser"
        self.save_unlearning_model()
            
    
    def unlearning_step_once(self, old_client_models, new_client_models, global_model_before_forget, global_model_after_forget,clip_value=1):
        """
        Parameters
        ----------
        old_client_models : list of client objects
        
        new_client_models : list of client objects
            
        global_model_before_forget : The old global model
            
        global_model_after_forget : The New global model
            

        Returns
        -------
        return_global_model : After one iteration, the new global model under the forgetting setting

        """
        old_param_update = dict()#Model Params： oldCM - oldGM_t
        new_param_update = dict()#Model Params： newCM - newGM_t
        
        new_global_model_state = global_model_after_forget.state_dict()#newGM_t
        
        return_model_state = dict()#newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||
        
        assert len(old_client_models) == len(new_client_models)
        
        for layer in global_model_before_forget.state_dict().keys():

            if "weight" not in layer and "bias" not in layer:
                    # print("This is "+layer)
                return_model_state[layer] = new_global_model_state[layer]
                continue
            #
            # if "bia" not in layer:
            #     print("This is 2"+layer)
            #     return_model_state[layer] = new_global_model_state[layer]
            #     continue

            old_param_update[layer] = 0*global_model_before_forget.state_dict()[layer]
            new_param_update[layer] = 0*global_model_before_forget.state_dict()[layer]
            
            return_model_state[layer] = 0*global_model_before_forget.state_dict()[layer]
            
            for ii in range(len(new_client_models)):
                old_param_update[layer] += old_client_models[ii].model.state_dict()[layer]
                new_param_update[layer] += new_client_models[ii].model.state_dict()[layer]

            old_param_update[layer] = old_param_update[layer] / (ii+1) #Model Params： oldCM
            new_param_update[layer] = new_param_update[layer] / (ii+1) #Model Params： newCM

            
            old_param_update[layer] = old_param_update[layer] - global_model_before_forget.state_dict()[layer]#参数： oldCM - oldGM_t
            new_param_update[layer] = new_param_update[layer] - global_model_after_forget.state_dict()[layer]#参数： newCM - newGM_t

            # if "bn" in layer:
            #     return_model_state[layer] = new_param_update[layer]
            #     continue

            step_length = torch.norm(old_param_update[layer])#||oldCM - oldGM_t||
            # step_length += torch.tensor(1e-8)  # 添加一个小的默认值

            norm_new_param = torch.norm(new_param_update[layer]) # 避免除0

            # if step_length == 0 or torch.norm(new_param_update[layer]) == 0:
            #     step_length = torch.norm(old_param_update[layer]) + 1e-8  # 避免除零
            #     norm_new_param = torch.norm(new_param_update[layer]) + 1  # 避免除零

            step_direction = (new_param_update[layer]) / norm_new_param
            # if np.isnan(step_direction.cpu().detach().numpy()).any():
            #     step_direction = 1
            # if np.isnan(step_length.cpu().detach().numpy()).any():
            #     step_length = 1

            # step_direction = new_param_update[layer]/torch.norm(new_param_update[layer])#(newCM - newGM_t)/||newCM - newGM_t||
            return_model_state[layer] = new_global_model_state[layer] + step_length * step_direction
            # if "num_batches_tracked" in layer:
            #     print("here")
            #     print(new_global_model_state[layer])
            #     return_model_state[layer] = new_global_model_state[layer]
                # print(step_length)
                # print(step_direction)
                # print(step_length * step_direction)

        
        # print("....", step_length, step_direction)
        # print(return_model_state)
        return_global_model = copy.deepcopy(global_model_after_forget)
        
        return_global_model.load_state_dict(return_model_state)
        # print(return_global_model)
        # for name, param in return_global_model.named_parameters():
        #     print(name, param.data)
        # for name, param in return_global_model.named_parameters():
        #     print(name)
        # exit(-1)

        return return_global_model



