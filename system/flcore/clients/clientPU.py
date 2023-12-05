import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *


class clientPU(Client):
    def __init__(self, args, id, train_samples, test_samples, capable, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.capable = capable
        if not self.capable:
            self.agg_flag = []
            for name, param in self.model.named_parameters():
                # FEMNIST_CNN's 1layer and 2layer
                # if 'fc' in name:
                # if 'fc' in name and 'fc1' not in name:
                # EVEN_CNN's 70%, 50%, and 10%:
                # if 'fc' in name or 'conv5' in name:
                # if 'fc' in name:
                if 'fc' in name or 'conv5' in name or 'conv4' in name:
                    param.requires_grad = True
                    self.agg_flag.append(1)
                else:
                    param.requires_grad = False
                    self.agg_flag.append(0)
                # # hybrid
                # if self.id <33:
                #     # 70%
                #     if 'fc' in name or 'conv5' in name or 'conv4' in name:
                #         param.requires_grad = True
                #         self.agg_flag.append(1)
                #     else:
                #         param.requires_grad = False
                #         self.agg_flag.append(0)
                # elif self.id >=33 and self.id <66:
                #     # 50%
                #     if 'fc' in name or 'conv5' in name:
                #         param.requires_grad = True
                #         self.agg_flag.append(1)
                #     else:
                #         param.requires_grad = False
                #         self.agg_flag.append(0)
                # elif self.id >=66 and self.id <100:
                #     if 'fc' in name:
                #         param.requires_grad = True
                #         self.agg_flag.append(1)
                #     else:
                #         param.requires_grad = False
                #         self.agg_flag.append(0)
                # else:
                #     raise NotImplementedError(f"client {self.id} has id >= 100")
        else:
            self.agg_flag = [1] * len(list(self.model.named_parameters()))
                    

    def train(self):
        self.model.to(self.device)
        self.model.train()

        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        
        start_time = time.time()
        
        if self.capable:
            assert self.model.conv1[0].weight.requires_grad == True
        else:
            assert self.model.conv1[0].weight.requires_grad == False

        for step in range(self.local_epochs):
            running_loss = 0.0
            for i, (x, y) in enumerate(self.trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * y.shape[0]
            print(f"Client {self.id:2d} local epoch {step+1:2d} training loss: {running_loss / self.train_samples:.4f}")

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
            
    def set_parameters_detach(self, model):
        new_weights = copy.deepcopy(model.state_dict())
        self.model.load_state_dict(new_weights)