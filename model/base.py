import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.tool import tensor2numpy, accuracy_func
from scipy.spatial.distance import cdist

EPSILON = 1e-8
batch_size = 64

class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self.args = args

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim
        
    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        """
        Evaluate model predictions against true labels.
        
        Args:
            y_pred (np.array): Predicted labels [N, topk]
            y_true (np.array): True labels [N]
        
        Returns:
            dict: Evaluation metrics
        """
        top1_pred = y_pred[:, 0]
        ret = {}
        grouped = accuracy_func(top1_pred, y_true, self._known_classes, self.args["increment"])
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        # Calculate top-k accuracy
        topk_matches = y_pred == y_true.reshape(-1, 1)
        topk_accuracy = np.sum(topk_matches) * 100 / len(y_true)
        ret[f"top{self.topk}"] = np.round(topk_accuracy, decimals=2)

        return ret
        

    def eval_task(self):
        """
        Evaluate the current task using CNN and NME (if applicable).
        
        Returns:
            tuple: (CNN accuracy, NME accuracy)
        """
        cnn_pred, cnn_true = self._eval_cnn(self.test_loader)
        cnn_acc = self._evaluate(cnn_pred, cnn_true)

        if hasattr(self, "_class_means"):
            nme_pred, nme_true = self._eval_nme(self.test_loader, self._class_means)
            nme_acc = self._evaluate(nme_pred, nme_true)
        else:
            nme_acc = None

        return cnn_acc, nme_acc

    def incremental_train(self):
        """Placeholder for incremental training method."""
        pass

    def _train(self):
        pass
    
    def _get_memory(self):
        """
        Retrieve stored exemplars.
        
        Returns:
            tuple or None: (exemplar_data, exemplar_targets) if available, else None
        """
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        """
        Compute model accuracy on given data loader.
        
        Args:
            model (nn.Module): Model to evaluate
            loader (DataLoader): Data loader for evaluation
        
        Returns:
            float: Accuracy percentage
        """
        model.eval()
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device)
                outputs = model(inputs)["logits"]
                predictions = outputs.argmax(dim=1)
                correct_predictions += (predictions.cpu() == targets).sum().item()
                total_samples += len(targets)

        accuracy = (correct_predictions / total_samples) * 100
        return np.round(accuracy, decimals=2)

    def _eval_cnn(self, loader):
        """
        Evaluate CNN model on given data loader.
        
        Args:
            loader (DataLoader): Data loader for evaluation
        
        Returns:
            tuple: (Predictions [N, topk], True labels [N])
        """
        self._network.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device)
                outputs = self._network(inputs)["logits"]
                topk_predictions = torch.topk(outputs, k=self.topk, dim=1).indices
                all_predictions.append(topk_predictions.cpu().numpy())
                all_labels.append(targets.numpy())

        return np.concatenate(all_predictions), np.concatenate(all_labels)

    def _eval_nme(self, loader, class_means):
        """
        Evaluate using Nearest Mean of Exemplars (NME).
        
        Args:
            loader (DataLoader): Data loader for evaluation
            class_means (np.array): Mean vectors for each class
        
        Returns:
            tuple: (Predictions [N, topk], True labels [N])
        """
        feature_vectors, true_labels = self._extract_vectors(loader)
        
        # Normalize feature vectors and class means
        normalized_vectors = self._normalize(feature_vectors)
        normalized_means = self._normalize(class_means)
        
        # Compute distances to class means
        distances = cdist(normalized_vectors, normalized_means, metric='sqeuclidean')
        
        # Get top-k nearest classes
        topk_predictions = np.argsort(distances, axis=1)[:, :self.topk]
        
        return topk_predictions, true_labels

    def _extract_vectors(self, loader):
        """
        Extract feature vectors from the data loader.
        
        Args:
            loader (DataLoader): Data loader for feature extraction
        
        Returns:
            tuple: (Feature vectors [N, feature_dim], True labels [N])
        """
        self._network.eval()
        feature_vectors = []
        true_labels = []
        
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device)
                features = self._network(inputs)["features"]
                feature_vectors.append(features.cpu().numpy())
                true_labels.append(targets.numpy())

        return np.concatenate(feature_vectors), np.concatenate(true_labels)

    def _normalize(self, vectors):
        """
        Normalize vectors to unit length.
        
        Args:
            vectors (np.array): Input vectors [N, feature_dim]
        
        Returns:
            np.array: Normalized vectors [N, feature_dim]
        """
        return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + EPSILON)
 

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per class)".format(m))

        # Shallow copy of data and targets memory
        dummy_data = self._data_memory.copy()
        dummy_targets = self._targets_memory.copy()

        # Initialize class means and lists to hold reduced exemplars
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        reduced_data_memory, reduced_targets_memory = [], []

        # Loop over each known class to reduce exemplars
        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]

        # Collect the reduced exemplars for data and targets
            reduced_data_memory.append(dd)
            reduced_targets_memory.append(dt)

        # Prepare dataset and dataloader for exemplar mean calculation
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )

        # Extract vectors and calculate exemplar mean
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            self._class_means[class_idx, :] = mean / np.linalg.norm(mean)

        # Concatenate the reduced data and targets memory once at the end
        self._data_memory = np.concatenate(reduced_data_memory)
        self._targets_memory = np.concatenate(reduced_targets_memory)

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
    
        # Loop through classes that need exemplar construction
        for class_idx in range(self._known_classes, self._total_classes):
        # Fetch data and corresponding targets for the current class
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
        
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = self._normalize(vectors)
            class_mean = np.mean(vectors, axis=0)

            # Select exemplars for the class
            selected_exemplars, exemplar_vectors = self._select_exemplars(vectors, data, class_mean, m)

            # Append selected exemplars to memory
            self._append_to_memory(selected_exemplars, class_idx, m)

            # Recompute class mean after exemplar selection
            self._recompute_class_mean(data_manager, selected_exemplars, class_idx, m)

    def _select_exemplars(self, vectors, data, class_mean, m):
        selected_exemplars = []
        exemplar_vectors = []  # [n, feature_dim]

        for k in range(1, m + 1):
            S = np.sum(exemplar_vectors, axis=0) if exemplar_vectors else np.zeros_like(vectors[0])
            mu_p = (vectors + S) / k  # [n, feature_dim] sum of all vectors + previously selected
            i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))  # Find closest to mean

            selected_exemplars.append(np.array(data[i]))  # Avoid passing by inference
            exemplar_vectors.append(np.array(vectors[i]))

            vectors = np.delete(vectors, i, axis=0)  # Remove the selected exemplar from consideration
            data = np.delete(data, i, axis=0)  # Remove corresponding data

        return np.array(selected_exemplars), np.array(exemplar_vectors)

    def _append_to_memory(self, selected_exemplars, class_idx, m):
        exemplar_targets = np.full(m, class_idx)  # Create an array of the same target for selected exemplars

        self._data_memory = (
            np.concatenate((self._data_memory, selected_exemplars))
        if len(self._data_memory) != 0
        else selected_exemplars
        )
        self._targets_memory = (
            np.concatenate((self._targets_memory, exemplar_targets))
            if len(self._targets_memory) != 0
            else exemplar_targets
        )

    def _recompute_class_mean(self, data_manager, selected_exemplars, class_idx, m):
        # Fetch dataset containing the newly selected exemplars
        idx_dataset = data_manager.get_dataset(
            [],
            source="train",
            mode="test",
            appendent=(selected_exemplars, np.full(m, class_idx)),
        )

        idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        vectors, _ = self._extract_vectors(idx_loader)
        vectors = self._normalize_vectors(vectors)
        mean = np.mean(vectors, axis=0)
        mean = mean / np.linalg.norm(mean)

        # Store the new class mean
        self._class_means[class_idx, :] = mean
    
    def _construct_exemplar_unified(self, data_manager, m):
        logging.info("Constructing exemplars for new classes...({} per classes)".format(m))
        _class_means = np.zeros((self._total_classes, self.feature_dim))

    # Helper function to calculate mean for given data and targets
        def compute_class_mean(data, targets):
            class_dset = data_manager.get_dataset([], source="train", mode="test", appendent=(data, targets))
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            return np.mean(vectors, axis=0) / np.linalg.norm(np.mean(vectors, axis=0))

        # Calculate the means of old classes with the newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = self._data_memory[mask], self._targets_memory[mask]
            _class_means[class_idx, :] = compute_class_mean(class_data, class_targets)

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
            np.arange(class_idx, class_idx + 1), source="train", mode="test", ret_data=True)
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

        # Select exemplars
            selected_exemplars, exemplar_vectors = [], []
            for k in range(1, m + 1):
                S = np.sum(exemplar_vectors, axis=0) if exemplar_vectors else np.zeros_like(vectors[0])
                mu_p = (vectors + S) / k  # Adjusted mean with selected exemplars
                i = np.argmin(np.linalg.norm(class_mean - mu_p, axis=1))  # Find closest vector to class mean

                selected_exemplars.append(np.array(data[i]))
                exemplar_vectors.append(np.array(vectors[i]))

                vectors = np.delete(vectors, i, axis=0)
                data = np.delete(data, i, axis=0)

        # Store the selected exemplars and their corresponding targets
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)

            self._data_memory = np.concatenate([self._data_memory, selected_exemplars]) if self._data_memory.size else selected_exemplars
            self._targets_memory = np.concatenate([self._targets_memory, exemplar_targets]) if self._targets_memory.size else exemplar_targets

        # Recompute the mean based on selected exemplars
            _class_means[class_idx, :] = compute_class_mean(selected_exemplars, exemplar_targets)

        self._class_means = _class_means

        
    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)


