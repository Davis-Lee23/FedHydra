import copy
import functools

import torch
import torch.nn as nn
import numpy as np
import os
import time
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from nbformat.sign import algorithms
from scipy.stats import poisson
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from wandb.sdk.lib.viz import visualize

from dataset_utils import read_client_data
from torch.nn.utils import clip_grad_norm_


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.args = args
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        
        # self.create_trigger_id = args.create_trigger_id
        self.trigger_size = args.trigger_size
        self.trim_percentage = args.trim_percentage

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss = nn.CrossEntropyLoss()

        # if self.dataset != 'cifar10':
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        # else:
        #     self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # if self.dataset[:2] == "ag":
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        
    
    def trim_weights(self, noise_std=0.1, trim_ratio=0.1):
        """
        对CNN模型的权重进行修剪。

        :param model: 要修剪的模型。
        :param k: 要修改的参数的百分比。
        :param noise_std: 加入的高斯噪声的标准差。
        :param trim_ratio: 要修剪的修改点的比例。
        """
        # 遍历所有参数并收集权重
        model = self.model
        k = self.trim_percentage
        all_weights = []
        for param in model.parameters():
            if param.requires_grad:
                all_weights.append(param.data.view(-1))

        # 将所有权重合并成一个向量
        all_weights = torch.cat(all_weights)
        total_params = all_weights.numel()
        # print(total_params)

        # 随机选择k%的参数进行修改
        num_modifications = int(total_params * k / 100)
        indices = np.random.choice(total_params, num_modifications, replace=False)
        
        # 对选择的参数加入高斯噪声或替换为随机值
        for idx in indices:
            if np.random.rand() < 0.8:
            # if True:
                # 加入高斯噪声
                all_weights[idx] += noise_std * torch.randn(1, device='cuda:0').item()
            else:
                # 替换为随机值
                all_weights[idx] = noise_std * torch.rand(1, device='cuda:0')

        # 修剪掉偏离量较大的修改
        deviations = torch.abs(all_weights - all_weights.clone().detach())
        trim_threshold = np.quantile(deviations.cpu().numpy(), 1 - trim_ratio)
        trimmed_indices = deviations > trim_threshold
        all_weights[trimmed_indices] = all_weights.clone().detach()[trimmed_indices]

        # 将修剪后的权重应用到模型
        idx = 0
        for param in model.parameters():
            if param.requires_grad:
                numel = param.data.numel()
                param.data = all_weights[idx:idx+numel].view(param.size())
                idx += numel

        return model


    """
    自带的load
    """
    def load_train_data(self, batch_size=None, create_trigger=False,dba=None,double=None):
        if batch_size == None:
            batch_size = self.batch_size
        # if self.id == self.create_trigger_id:
        if create_trigger:
            train_data = read_client_data(self.dataset, self.id, is_train=True, create_trigger=True, 
                                          trigger_size=self.trigger_size,
                                          label_inject_mode=self.args.label_inject_mode,
                                          tampered_label=self.args.tampered_label,
                                          num_classes=self.args.num_classes,
                                          backdoor_ratio=self.args.backdoor_ratio,
                                          attack_method=self.args.unlearn_attack_method,
                                          algorithm = self.args.algorithm,
                                          dba = dba,
                                          double = double
                                          )
        else:
            train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)

    """
    微调trigger
    """
    def load_tune_data(self, batch_size=None, create_trigger=False):

        if batch_size == None:
            batch_size = self.batch_size
        # if self.id == self.create_trigger_id:
        if create_trigger:
            train_data = read_client_data(self.dataset, self.id, is_train=True, create_trigger=True,
                                          trigger_size=self.trigger_size, label_inject_mode=self.args.label_inject_mode,
                                          tampered_label=self.args.tampered_label, num_classes=self.args.num_classes)
        else:
            train_data = read_client_data(self.dataset, self.id, is_train=True)

        return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            # old_param.grad = None
            old_param.data = new_param.data.clone()
        # for new_param, old_param in zip(model.parameters(), self.model.parameters()):
        #     # old_param.data = new_param.data.clone()


    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # print(testloaderfull)
        # self.model = self.load_model('model')
        self.model.eval()
        self.model.to(self.device)

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                # y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                # y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        # y_prob = np.concatenate(y_prob, axis=0)
        # y_true = np.concatenate(y_true, axis=0)

        # auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        # print(test_acc)
        # print(test_num)
        # exit(-1)
        self.model.train()
        
        return test_acc, test_num, 0

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        ter = 0
        
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                
                # TSR: test error rate
                ter += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')
        # print(train_num)

        self.model.train()
        return ter, losses, train_num

    # 提取前几张图片进行可视化
    def visualize_images(self,data_loader, num_images=5):
        # 获取数据
        images, labels = next(iter(data_loader))

        # 将图片移到CPU并转换为numpy格式
        images = images.cpu().numpy()

        # 进行必要的反向变换，如果有进行标准化的话
        # 假设使用的标准化是均值和标准差
        # transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # 反向标准化（如果数据被标准化过）
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        images = images * std[:, None, None] + mean[:, None, None]  # 反向标准化

        # 创建一个图形
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

        for i in range(num_images):
            # 获取单张图片
            image = images[i]

            # 转换为HWC格式并反转标准化（如果有的话）
            image = (image.transpose(1, 2, 0) * 255).astype('uint8')  # 假设图片在[0, 1]范围

            # 显示图片
            axes[i].imshow(image)
            axes[i].set_title(f'Label: {labels[i].item()}')
            axes[i].axis('off')

        plt.show()

    def imgshow(self,train_loader):
        # 获取一批数据
        dataiter = iter(train_loader)
        images, labels = next(dataiter)

        img = torchvision.utils.make_grid(images[:4])
        img = img / 2 + 0.5  # 反归一化
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.show()
        print(' '.join(f'{labels[j].item()}' for j in range(4)))  # 打印标签
        exit(-1)

    def asr_metrics(self, model):
        # print( "Client id: " +str(self.id))
        trainloader = self.load_train_data(create_trigger=True)
        # trainloader = self.load_train_data()
        # self.imgshow(trainloader)
        # self.visualize_images(trainloader,8)

        # if model.training:
        #     print(True)
        # else:
        #     print(False)
        #
        # exit(-1)

        # 这个model是全局模型
        model.eval()

        train_num = 0
        losses = 0
        asr = 0
        
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                # output全是Nan
                output = model(x)
                # print(torch.argmax(output, dim=1))
                
                # asr: attack success rate
                asr += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                # print(output.shape)
                # print(torch.argmax(output, dim=1))
                # print(y)
                # print("-"*20)
                
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return asr, losses, train_num
    

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
    
################################################################################################################
    
class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def model_dist_norm_var(self,model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            # print(name)
            size += layer.view(-1).shape[0]
        sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        # name为层名，layer为数值
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
            layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    def init_malicious_network(self, flat_params):
        # set the malicious parameters to be the same as in the main network
        self.row_into_parameters(flat_params, self.model.parameters())

    def train_malicious_network(self, initial_params_flat):
        print("LIE Attack")
        self.init_malicious_network(initial_params_flat)
        initial_params = [torch.tensor(torch.empty(p.shape), requires_grad=False) for p in
                          self.model.parameters()]
        self.row_into_parameters(initial_params_flat, initial_params)

        self.model.train()

        # 正常loss和后门loss的系数和应该为1，显然论文和实验不符
        # for step in range(self.local_epochs):


    def row_into_parameters(self, row, parameters):
        offset = 0
        for param in parameters:
            new_size = functools.reduce(lambda x, y: x * y, param.shape)
            current_data = row[offset:offset + new_size]

            param.data[:] = torch.from_numpy(current_data.reshape(param.shape))
            offset += new_size

    def flatten_params(self,params):
        return np.concatenate([i.data.cpu().numpy().flatten() for i in params])

    def train(self, create_trigger=False, trim_attack=False,dba =None,double=None):

        # 在读数据这一步注入后门,此处返回了一个loader
        # print(self.args.clamp_to_little_range) 默认是false
        trainloader = self.load_train_data(create_trigger=create_trigger,dba=dba,double=double)
        # if dba is not None:
        #     self.visualize_images(trainloader,5)


        if create_trigger and self.args.clamp_to_little_range:
            trainloader_comp = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        self.model.to(self.device)
        
        start_time = time.time() 

        max_local_epochs = self.local_epochs

        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            if create_trigger == True and self.args.clamp_to_little_range == True :
                print("Loacl epochs:"+str(max_local_epochs))
                print("LIE Client Attack")

                """
                流程应当如此：正常训练，得到初步的grad -> 计算得到均值和标准差 -> 后门操作
                -> std -> 走入恶意train
                
                总结：文章与开源本身就存在冲突，且文章有多个版本，按照自己的理解来吧
                """
                original_params = np.concatenate([i.data.cpu().numpy().flatten() for i in self.model.parameters()])


                #
                for i, (x, y) in enumerate(trainloader_comp):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    break

                self.optimizer.zero_grad()
                output = self.model(x)
                loss1 = self.loss(output, y)
                loss1.backward()
                grads = np.concatenate([param.grad.data.cpu().numpy().flatten() for param in self.model.parameters()])
                grads_mean = np.mean(grads,axis=0)
                grads_stdev = np.var(grads,axis=0) ** 0.5

                initial_params_flat = original_params - self.args.local_learning_rate * grads_mean

                # print("LIE Attack")
                self.init_malicious_network(initial_params_flat)
                initial_params = [torch.tensor(torch.empty(p.shape), requires_grad=False) for p in
                                  self.model.parameters()]
                self.row_into_parameters(initial_params_flat, initial_params)

                self.model.train()

                dist_loss_func = nn.MSELoss()
                for step in range(max_local_epochs):
                    for i, (train_bd, train_comp) in enumerate(zip(trainloader,trainloader_comp)):
                        x, y = train_bd
                        x_comp, y_comp = train_comp

                        x = x.to(self.device)
                        x_comp = x_comp.to(self.device)
                        y = y.to(self.device)
                        y_comp = y_comp.to(self.device)

                        self.optimizer.zero_grad()
                        output = self.model(x)
                        loss = self.loss(output, y)

                        dist_loss = 0
                        for idx, p in enumerate(self.model.parameters()):
                            initial_params[idx] = initial_params[idx].cuda()
                            dist_loss += dist_loss_func(p, initial_params[idx])

                        # 论文里应该是这样，但是复现都不是这样
                        loss = self.args.lie_alpha_loss * loss + (1-self.args.lie_alpha_loss) * dist_loss

                        # loss += self.args.lie_alpha_loss  * dist_loss

                        loss.backward()
                        self.optimizer.step()

                mal_net_params = self.flatten_params(self.model.parameters())
                new_params = mal_net_params + self.args.local_learning_rate * grads_mean
                new_grads = (initial_params_flat - new_params) / self.args.local_learning_rate
                new_user_grads = np.clip(new_grads, grads_mean - self.args.num_std * grads_stdev,
                                         grads_mean + self.args.num_std * grads_stdev)

                new_user_grads_tensor = torch.tensor(new_user_grads, dtype=torch.float32).to(self.device)

                start_index = 0
                self.model.grad = new_user_grads_tensor

                # for param in self.model.parameters():
                #     if param.grad is not None:  # 检查参数当前是否有梯度
                #         param_size = param.numel()  # 获取当前参数的总元素数
                #         # 从 new_user_grads_tensor 中提取与当前参数大小相同的切片，并赋值给 param.grad
                #         param.grad.data = new_user_grads_tensor[start_index:start_index + param_size].view_as(param)
                #         start_index += param_size  # 更新索引，指向下一个参数

                self.optimizer.step()

                # print("执行成功")
                # break
                #
                # # ——————————————————————————————————————————————————————————————————————————————————————————
                #
                # for i, (train_bd, train_comp) in enumerate(zip(trainloader,trainloader_comp)):
                #     x, y = train_bd
                #     x_comp, y_comp = train_comp
                #     if type(x) == type([]):
                #         x[0] = x[0].to(self.device)
                #         x_comp[0] = x_comp[0].to(self.device)
                #     else:
                #         x = x.to(self.device)
                #         x_comp = x_comp.to(self.device)
                #     y = y.to(self.device)
                #     y_comp = y_comp.to(self.device)
                #     if self.train_slow:
                #         time.sleep(0.1 * np.abs(np.random.rand()))
                #     output = self.model(x)
                #     loss1 = self.loss(output, y)
                #
                #     output_comp = self.model(x_comp)
                #     loss2 = self.loss(output_comp, y_comp)
                #
                #     # 默认是0.8 0.2，前者有毒，后者干净
                #     loss = 0.8 * loss1 + 0.2 * loss2
                #
                #     # 使用torch.clip来确保总损失loss在0.5 * loss1和1.5 * loss1之间，防止损失过大或过小，这样可以提高训练的稳定性。
                #     loss = torch.clip(loss, 0.2 * loss1, 0.7 * loss1)
                #
                #     self.optimizer.zero_grad()
                #     loss.backward()
                #     self.optimizer.step()
                # print(loss1, loss, loss2)
                
                # initial_params_flat = original_params - learning_rate * grads_mean # the corrected param after the user optimized, because we still want the model to improve

                # mal_net_params = self.train_malicious_network(initial_params_flat)

                # #Getting from the final required mal_net_params to the gradients that needs to be applied on the parameters of the previous round.
                # new_params = mal_net_params + learning_rate * grads_mean
                # new_grads = (initial_params_flat - new_params) / learning_rate

                # new_user_grads = np.clip(new_grads, grads_mean - self.num_std * grads_stdev,
                #                     grads_mean + self.num_std * grads_stdev)

            """
            ModelRe
            """
            if create_trigger == True and self.args.modelre == True:
                print("ModelRe Client Train!!!!")
                last_global_model = copy.deepcopy(self.model)
                target_params_variables = dict()
                for name, param in last_global_model.named_parameters():
                    target_params_variables[name] = last_global_model.state_dict()[name].clone().detach().requires_grad_(
                        False)

                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    output = self.model(x)
                    # 正常训练的loss
                    class_loss = self.loss(output, y)
                    # 距离loss
                    distance_loss = self.model_dist_norm_var(self.model,target_params_variables)

                    loss = self.args.alpha_loss * class_loss + (1-self.args.alpha_loss) * distance_loss
                    # loss = self.args.alpha_loss * class_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Adversary wants to scale his weights.
                # Computing scale weights
                # scale_weights = (self.args.num_clients - self.args.unlearn_clients_number)
                # clip_rate = scale_weights / self.args.unlearn_attack_number

                # print(clip_rate)
                # clip_rate *= 1.8
                clip_rate = self.args.clip_rate
                # print(scale_weights)
                # print("Clip_rate:"+str(clip_rate))

                for key, value in self.model.state_dict().items():

                    target_value = last_global_model.state_dict()[key]
                    new_value = target_value + (value - target_value) * clip_rate
                    self.model.state_dict()[key].copy_(new_value)

            elif create_trigger == True and self.args.dba == True and self.args.dba_clip_rate != 0:
                # 如果dba的clip rate为0，就是正常的后门训练
                print("Scale DBA Training!!!")

                last_global_model = copy.deepcopy(self.model)
                target_params_variables = dict()
                for name, param in last_global_model.named_parameters():
                    target_params_variables[name] = last_global_model.state_dict()[name].clone().detach().requires_grad_(
                        False)

                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    output = self.model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Adversary wants to scale his weights.
                scale_weights = (self.args.num_clients - self.args.unlearn_clients_number)
                # clip_rate = scale_weights / self.args.unlearn_attack_number
                clip_rate = self.args.dba_clip_rate

                for key, value in self.model.state_dict().items():
                    target_value = last_global_model.state_dict()[key]
                    new_value = target_value + (value - target_value) * clip_rate
                    self.model.state_dict()[key].copy_(new_value)

            # 正常训练
            else:
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    output = self.model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        
        
        if trim_attack:
            self.model = self.trim_weights()
        
        
    def train_one_step(self,trigger=False,local_epoch=1,dba = None):
        trainloader = self.load_train_data(create_trigger=trigger,dba =dba)
        self.model.train()
        # if dba is not None:
        #     self.visualize_images(trainloader,5)

        start_time = time.time() 

        for step in range(local_epoch):
            # print("this is step:"+str(step))
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        
        
################################################################################################################
    
class clientFedRecover(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
    def train(self, create_trigger=False, trim_attack=False):
        trainloader = self.load_train_data(create_trigger=create_trigger)
        if create_trigger == True and self.args.clamp_to_little_range == True:
            trainloader_comp = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time() 

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            if create_trigger == True and self.args.clamp_to_little_range == True:
                for i, (train_bd, train_comp) in enumerate(zip(trainloader,trainloader_comp)):
                    x, y = train_bd
                    x_comp, y_comp = train_comp
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                        x_comp[0] = x_comp[0].to(self.device)
                    else:
                        x = x.to(self.device)
                        x_comp = x_comp.to(self.device)
                    y = y.to(self.device)
                    y_comp = y_comp.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    output = self.model(x)
                    loss1 = self.loss(output, y)
                    
                    output_comp = self.model(x_comp)
                    loss2 = self.loss(output_comp, y_comp)
                    
                    loss = 0.8 * loss1 + 0.2 * loss2
                    
                    loss = torch.clip(loss, 0.5 * loss1, 1.5 * loss1)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            else:
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    output = self.model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        
        if trim_attack:
            self.model = self.trim_weights()
        
        
    def retrain_with_LBFGS(self):
        self.optimizer = torch.optim.LBFGS(params = self.model.parameters(), lr=self.learning_rate, history_size=1, max_iter=4)
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time() 

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):   
            for i, (x, y) in enumerate(trainloader):
                def closure():
                    nonlocal x, y
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(x)
                    loss = self.loss(output, y)
                    loss.backward()
                    return loss
                self.optimizer.step(closure)


        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


        