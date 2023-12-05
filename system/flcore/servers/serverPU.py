import time
from flcore.clients.clientPU import clientPU
from flcore.servers.serverbase import Server
from threading import Thread
import json
import os
import torch
import copy
import numpy as np
import h5py

class ServerPU(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        self.fname = args.fname
        self.eval_ft = args.eval_ft
        self.detach = args.detach
        
        self.capable_num = args.capable_rate * self.num_clients

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientPU)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def set_clients(self, clientObj):
        if self.dataset=='femnist':
            all_train_data = json.load(open(os.path.join(os.path.dirname(__file__), '../../../dataset/femnist/train/mytrain.json')))
            all_test_data = json.load(open(os.path.join(os.path.dirname(__file__), '../../../dataset/femnist/test/mytest.json')))
            
            self.chosen_clients = list(range(self.num_clients))
            
            for i, train_slow, send_slow in zip(self.chosen_clients, self.train_slow_clients, self.send_slow_clients):
                user_train = all_train_data['user_data'][ "{0:04d}".format(i)]
                user_test = all_test_data['user_data'][ "{0:04d}".format(i)]

                user_train_x = torch.Tensor(user_train['x']).type(torch.float32).reshape(-1, 1, 28, 28)
                user_test_x = torch.Tensor(user_test['x']).type(torch.float32).reshape(-1, 1, 28, 28)
                user_train_y = torch.Tensor(user_train['y']).type(torch.int64)
                user_test_y = torch.Tensor(user_test['y']).type(torch.int64)
                train_data=[(x, y) for x,y in zip(user_train_x, user_train_y)] 
                test_data=[(x, y) for x,y in zip(user_test_x, user_test_y)]
                
                capable = True if i<self.capable_num else False
                print(f"Client {i} capable: {capable}")

                client = clientObj(self.args, 
                                    id=i, 
                                    train_samples=train_data, 
                                    test_samples=test_data,
                                    train_slow=train_slow, 
                                    send_slow=send_slow,
                                    capable=capable)
                self.clients.append(client)
        else:
            raise NotImplementedError
            for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
                train_data = read_client_data(self.dataset, i, is_train=True)
                test_data = read_client_data(self.dataset, i, is_train=False)
                client = clientObj(self.args, 
                                id=i, 
                                train_data=train_data, 
                                test_data=test_data, 
                                train_slow=train_slow, 
                                send_slow=send_slow)
                self.clients.append(client)

    
    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_weights = [] # list of clients' num_samples
        self.uploaded_models = []
        self.uploaded_flags = []
        tot_samples = 0
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
            self.uploaded_flags.append(client.agg_flag)
        # for i, w in enumerate(self.uploaded_weights):
        #     self.uploaded_weights[i] = w / tot_samples
            
            
    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])        
        new_weights = {k: torch.zeros_like(v) for k, v 
                       in self.global_model.state_dict().items()}

        flagged_sum = np.zeros_like(self.uploaded_flags[0])
        for w, client_flag in zip(self.uploaded_weights, self.uploaded_flags):
            flagged_sum += np.array(client_flag) * w

        print("flagged_sum:", flagged_sum)
        
        for w, client_model, client_flag in zip(self.uploaded_weights, self.uploaded_models, self.uploaded_flags):
            for i, (k, v) in enumerate(client_model.state_dict().items()):
                new_weights[k] += v.clone() * client_flag[i] * (w  / flagged_sum[i])

        self.global_model.load_state_dict(new_weights)
            
            
    def train(self):
        for i in range(1,self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            print(f"\n-------------Round {i}-------------")

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            
            if i%self.eval_gap == 0:
                print("\nEvaluate global model")
                self.evaluate()

                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    self.Budget.append(time.time() - s_t)
                    print(f'// time cost: {int(self.Budget[-1])} s')
                    print(f'[early stop at round {i}]')
                    break
                
            self.Budget.append(time.time() - s_t)
            print(f'// time cost: {int(self.Budget[-1])} s')

        print("\nBest accuracy:", max(self.rs_test_acc))
        print("\nAverage time cost per round:", sum(self.Budget[1:])/len(self.Budget[1:]))

        result_path = os.path.join(os.path.dirname(__file__), '../../../results/', self.fname)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        self.save_results(result_path)
        self.save_global_model(result_path)

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
    
    
    def save_results(self, result_path):
        algo = self.dataset + "_" + self.algorithm
        # result_path = "../results/"

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "/{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
