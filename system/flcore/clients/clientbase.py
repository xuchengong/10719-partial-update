import copy
import torch
import torch.nn as nn
import numpy as np
import os
import copy
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = len(train_samples)
        self.test_samples = len(test_samples)
        self.batch_size = args.batch_size
        self.local_learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

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
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_learning_rate, momentum=0.9)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        
        self.trainloader = self.load_train_data(train_samples)
        self.testloaderfull = self.load_test_data(test_samples)
        

    def load_train_data(self, train_samples, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        return DataLoader(train_samples, batch_size, drop_last=False, shuffle=True, pin_memory=True, num_workers=0)

    def load_test_data(self, test_samples, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        return DataLoader(test_samples, batch_size, drop_last=False, shuffle=True, pin_memory=True, num_workers=0)
        
    # def set_parameters(self, model):
    #     for new_param, old_param in zip(model.parameters(), self.model.parameters()):
    #         old_param.data = new_param.data.clone()

    # def set_parameters_detach(self, model):
    #     for new_param, old_param in zip(model.parameters(), self.model.parameters()):
    #         old_param.data = new_param.data.detach().clone()

    # def clone_model(self, model, target):
    #     for param, target_param in zip(model.parameters(), target.parameters()):
    #         target_param.data = param.data.clone()
    #         # target_param.grad = param.grad.clone()

    # def update_parameters(self, model, new_params):
    #     for param, new_param in zip(model.parameters(), new_params):
    #         param.data = new_param.data.clone()

    def set_parameters(self, model):
        new_weights = copy.deepcopy(model.state_dict())
        self.model.load_state_dict(new_weights)

    def set_parameters_detach(self, model):
        new_weights = copy.deepcopy(model.state_dict())
        self.model.load_state_dict(new_weights)

    def clone_model(self, model, target):
        new_weights = copy.deepcopy(model.state_dict())
        for k, v in model.state_dict().items():
            new_weights[k] = v.clone()
        target.load_state_dict(new_weights)

    def test_metrics(self, global_model=None):
        if global_model:
            global_model.to(self.device)
            global_model.eval()
        else:
            self.model.to(self.device)
            self.model.eval()

        test_acc, test_num = 0, 0
        y_prob, y_true = [], []
        
        with torch.no_grad():
            for x, y in self.testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if global_model:
                    output = global_model(x)
                else:
                    output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc
    


    def train_metrics(self, global_model=None):
        if global_model:
            global_model.to(self.device)
            global_model.eval()
        else:
            self.model.to(self.device)
            self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in self.trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if global_model:
                    output = global_model(x)
                else:
                    output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

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
