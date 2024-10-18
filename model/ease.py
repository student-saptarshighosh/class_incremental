import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.increment_net import EaseNet
from model.base import BaseLearner
from utils.tool import tensor2numpy

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = EaseNet(args, True)
        
        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args.get("weight_decay", 0.0005)
        self.min_lr = args.get("min_lr", 1e-8)
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]

        self.use_exemplars = args["use_old_data"]
        self.use_init_ptm = args["use_init_ptm"]
        self.use_diagonal = args["use_diagonal"]
        
        self.recalc_sim = args["recalc_sim"]
        self.alpha = args.get("alpha", 0.1)  # forward_reweight is divided by _cur_task
        self.beta = args.get("beta", 0.1)

        self.moni_adam = args["moni_adam"]
        self.adapter_num = args["adapter_num"]
        
        if self.moni_adam:
            self.use_init_ptm = True
            self.alpha = 1 
            self.beta = 1

    def after_task(self):
        """Freeze model after completing a task and add current task's adapter to the list."""
        self._known_classes = self._total_classes
        self._network.freeze()
        self._network.backbone.add_adapter_to_list()
    
    def get_cls_range(self, task_id):
        """Get the class range based on the task ID."""
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            start_cls = self.init_cls + (task_id - 1) * self.inc
            end_cls = start_cls + self.inc
        
        return start_cls, end_cls
        
    def replace_fc(self, train_loader):
        """Replace the fully connected layer with class prototypes."""
        model = self._network
        model.eval()
        
        with torch.no_grad():           
            # Replace proto for each adapter in the current task
            start_idx = -1 if self.use_init_ptm else 0

            for index in range(start_idx, self._cur_task + 1):
                if self.moni_adam and index > self.adapter_num - 1:
                    break
                elif self.use_diagonal and index not in [-1, self._cur_task]:
                    continue

                embedding_list, label_list = [], []
                for _, batch in enumerate(train_loader):
                    _, data, label = batch
                    data, label = data.to(self._device), label.to(self._device)
                    embedding = model.backbone.forward_proto(data, adapt_index=index)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())

                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)
                
                class_list = np.unique(self.train_dataset_for_protonet.labels)
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)

                    if self.use_init_ptm:
                        model.fc.weight.data[class_index, (index+1)*self._network.out_dim:(index+2)*self._network.out_dim] = proto
                    else:
                        model.fc.weight.data[class_index, index*self._network.out_dim:(index+1)*self._network.out_dim] = proto

            # Handle exemplars for past tasks
            if self.use_exemplars and self._cur_task > 0:
                embedding_list, label_list = [], []
                dataset = self.data_manage.get_dataset(np.arange(0, self._known_classes), source="train", mode="test")
                loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

                for _, batch in enumerate(loader):
                    _, data, label = batch
                    data, label = data.to(self._device), label.to(self._device)
                    embedding = model.backbone.forward_proto(data, adapt_index=self._cur_task)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())

                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)
                
                class_list = np.unique(dataset.labels)
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    model.fc.weight.data[class_index, -self._network.out_dim:] = proto

        # Diagonal/exemplar handling
        if self.use_diagonal or self.use_exemplars:
            return
        
        if self.recalc_sim:
            self.solve_sim_reset()
        else:
            self.solve_similarity()
    
    def get_A_B_Ahat(self, task_id):
        start_dim, end_dim = self.calculate_start_end_dim(task_id)

        start_cls, end_cls = self.get_cls_range(task_id)

        # Extract weights for current and previous tasks
        A = self._network.fc.weight.data[self._known_classes:, start_dim:end_dim]  # W(Ti)
        B = self._network.fc.weight.data[self._known_classes:, -self._network.out_dim:]  # W(TT)
        A_hat = self._network.fc.weight.data[start_cls:end_cls, start_dim:end_dim]  # W(ii)

        return A.cpu(), B.cpu(), A_hat.cpu()

    def calculate_start_end_dim(self, task_id):
        """Helper function to calculate start and end dimensions."""
        if self.use_init_ptm:
            start_dim = (task_id + 1) * self._network.out_dim
        else:
            start_dim = task_id * self._network.out_dim

        end_dim = start_dim + self._network.out_dim
        return start_dim, end_dim

    def solve_similarity(self):
        """Solve similarities and update weights for all tasks."""
        for task_id in range(self._cur_task):
            start_cls, end_cls = self.get_cls_range(task_id)
            A, B, A_hat = self.get_A_B_Ahat(task_id)

            similarity = torch.zeros(len(A_hat), len(A))
        
            # Efficient similarity calculation
            for i in range(len(A_hat)):
                similarity[i] = torch.cosine_similarity(A_hat[i].unsqueeze(0), A, dim=1)
        
            similarity = F.softmax(similarity, dim=1)  # Normalize similarity

            B_hat = torch.matmul(similarity, B)  # Weight combination of B
            self._network.fc.weight.data[start_cls:end_cls, -self._network.out_dim:] = B_hat.to(self._device)

    def solve_sim_reset(self):
        """Reset and solve similarity across multiple tasks."""
        for task_id in range(self._cur_task):
            if self.moni_adam and task_id > self.adapter_num - 2:
                break

            range_dim = self.get_range_dim(task_id)

            for dim_id in range_dim:
                if self.moni_adam and dim_id > self.adapter_num:
                    break

                start_cls, end_cls = self.get_cls_range(task_id)
                start_dim, end_dim = self.calculate_start_end_dim(dim_id)

                # Use above diagonal
                start_cls_old, end_cls_old, start_dim_old, end_dim_old = self.calculate_old_dims(task_id, dim_id)

                A = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim_old:end_dim_old].cpu()
                B = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim:end_dim].cpu()
                A_hat = self._network.fc.weight.data[start_cls:end_cls, start_dim_old:end_dim_old].cpu()

                similarity = torch.zeros(len(A_hat), len(A))

            # Efficient similarity calculation
                for i in range(len(A_hat)):
                    similarity[i] = torch.cosine_similarity(A_hat[i].unsqueeze(0), A, dim=1)

                similarity = F.softmax(similarity, dim=1)  # Normalize similarity

                B_hat = torch.matmul(similarity, B)  # Weight combination of B
                self._network.fc.weight.data[start_cls:end_cls, start_dim:end_dim] = B_hat.to(self._device)

    def get_range_dim(self, task_id):
        """Helper function to get the range of dimensions."""
        if self.use_init_ptm:
            return range(task_id + 2, self._cur_task + 2)
        else:
            return range(task_id + 1, self._cur_task + 1)

    def calculate_old_dims(self, task_id, dim_id):
        """Helper function to calculate old class and dimension ranges."""
        if self.use_init_ptm:
            start_cls_old = self.init_cls + (dim_id - 2) * self.inc
            end_cls_old = self._total_classes
            start_dim_old = (task_id + 1) * self._network.out_dim
            end_dim_old = (task_id + 2) * self._network.out_dim
        else:
            start_cls_old = self.init_cls + (dim_id - 1) * self.inc
            end_cls_old = self._total_classes
            start_dim_old = task_id * self._network.out_dim
            end_dim_old = (task_id + 1) * self._network.out_dim

        return start_cls_old, end_cls_old, start_dim_old, end_dim_old
    
    def incremental_train(self, data_manage):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manage.task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
    
        logging.info(f"Learning on {self._known_classes}-{self._total_classes}")
    
        # Fetch datasets
        self.data_manage = data_manage
        self.train_loader = self.get_data_loader(self._known_classes, self._total_classes, mode="train", shuffle=True)
        self.test_loader = self.get_data_loader(0, self._total_classes, mode="test", shuffle=False)
        self.train_loader_for_protonet = self.get_data_loader(self._known_classes, self._total_classes, mode="test", shuffle=True)

        # Multi-GPU handling
        if len(self._multiple_gpus) > 1:
            logging.info('Using multiple GPUs.')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        # Training
        self._train(self.train_loader, self.test_loader)
    
        # Reset model for single GPU if it was DataParallel
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    
        # Replace the fully connected layer
        self.replace_fc(self.train_loader_for_protonet)
    
    def get_data_loader(self, start_cls, end_cls, mode="train", shuffle=True):
        """Utility function to get a DataLoader for a dataset"""
        dataset = self.data_manage.get_dataset(np.arange(start_cls, end_cls), source="train", mode=mode)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        # Select optimizer and scheduler based on current task
        if self._cur_task == 0 or self.init_cls == self.inc:
            optimizer = self.get_optimizer(lr=self.args["init_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["init_epochs"])
        else:
            # for base 0 setting, the later_lr and later_epochs are not used
            # for base N setting, the later_lr and later_epochs are used
            if "later_lr" not in self.args or self.args["later_lr"] == 0:
                self.args["later_lr"] = self.args["init_lr"]
            if "later_epochs" not in self.args or self.args["later_epochs"] == 0:
                self.args["later_epochs"] = self.args["init_epochs"]
            optimizer = self.get_optimizer(lr=self.args["later_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["later_epochs"])
    
        self._init_train(train_loader, test_loader, optimizer, scheduler)


    def get_optimizer(self, lr):
        optimizers = {
            'sgd': optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()), 
                momentum=0.9, 
                lr=lr,
                weight_decay=self.weight_decay
            ),
            'adam': optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr, 
                weight_decay=self.weight_decay
            ),
            'adamw': optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr, 
                weight_decay=self.weight_decay
            )
        }
        return optimizers.get(self.args['optimizer'], optimizers['sgd'])
    

    def get_scheduler(self, optimizer, epoch):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler
    
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        # Early exit if using 'moni_adam' and current task exceeds allowed adapters
        if self.moni_adam and self._cur_task > self.adapter_num - 1:
            return
    
        # Determine the number of epochs based on whether it's the first task or not
        epochs = self.args['init_epochs'] if self._cur_task == 0 or self.init_cls == self.inc else self.args['later_epochs']
    
        # Progress bar for visual feedback during training
        prog_bar = tqdm(range(epochs), desc=f"Task {self._cur_task}, Epoch 0/{epochs}")

        for epoch in prog_bar:
            self._network.train()  # Set the network to training mode

            losses, correct, total = 0.0, 0, 0

            # Training loop
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # Adjust the target labels for auxiliary classification
                aux_targets = torch.where(
                    targets - self._known_classes >= 0,
                    targets - self._known_classes,
                    -1
                ).long()

                # Forward pass through the network
                output = self._network(inputs, test=False)
                logits = output["logits"]

                # Compute loss using cross-entropy
                loss = F.cross_entropy(logits, aux_targets)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate loss
                losses += loss.item()

                # Get predictions and calculate correct classifications
                preds = torch.argmax(logits, dim=1)
                correct += preds.eq(aux_targets).cpu().sum().item()
                total += aux_targets.size(0)

            # Update the learning rate using the scheduler
            if scheduler:
                scheduler.step()

            # Calculate training accuracy
            train_acc = np.around((correct / total) * 100, decimals=2)

            # Update the progress bar with the current training status
            info = f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => Loss {losses / len(train_loader):.3f}, Train_acc {train_acc:.2f}"
            prog_bar.set_description(info)

        logging.info(info)  # Log the final epoch information

    def _eval_cnn(self, loader):
        calc_task_acc = True
        
        if calc_task_acc:
            task_correct, task_acc, total = 0, 0, 0
            
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                outputs = self._network.forward(inputs, test=True)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            
            # calculate the accuracy by using task_id
            if calc_task_acc:
                task_ids = (targets - self.init_cls) // self.inc + 1
                task_logits = torch.zeros(outputs.shape).to(self._device)
                for i, task_id in enumerate(task_ids):
                    if task_id == 0:
                        start_cls = 0
                        end_cls = self.init_cls
                    else:
                        start_cls = self.init_cls + (task_id-1)*self.inc
                        end_cls = self.init_cls + task_id*self.inc
                    task_logits[i, start_cls:end_cls] += outputs[i, start_cls:end_cls]
                # calculate the accuracy of task_id
                pred_task_ids = (torch.max(outputs, dim=1)[1] - self.init_cls) // self.inc + 1
                task_correct += (pred_task_ids.cpu() == task_ids).sum()
                
                pred_task_y = torch.max(task_logits, dim=1)[1]
                task_acc += (pred_task_y.cpu() == targets).sum()
                total += len(targets)

        if calc_task_acc:
            logging.info("Task correct: {}".format(tensor2numpy(task_correct) * 100 / total))
            logging.info("Task acc: {}".format(tensor2numpy(task_acc) * 100 / total))
                
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

