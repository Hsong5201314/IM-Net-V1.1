import os
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
from tqdm import tqdm


# ================= 1. Strict and High-Speed Negative Sampling Dataset =================
class RecDataset(Dataset):
    def __init__(self, interaction_data, train_dict, num_items, neg_count=1):
        self.data = interaction_data
        self.train_dict = train_dict
        self.num_items = num_items
        self.neg_count = neg_count

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, pos_item = self.data[idx]
        pos_set = self.train_dict[user]
        neg_items = []
        max_tries = self.neg_count * 100
        tries = 0
        while len(neg_items) < self.neg_count and tries < max_tries:
            neg = random.randint(0, self.num_items - 1)
            if neg not in pos_set:
                neg_items.append(neg)
            tries += 1
        # Fill insufficient samples randomly (possible low-probability duplicates)
        while len(neg_items) < self.neg_count:
            neg_items.append(random.randint(0, self.num_items - 1))
        return user, pos_item, neg_items

def collate_fn(batch):
    users, pos_items, neg_items_list = zip(*batch)
    # Expand multiple negative samples for each instance
    expanded_users = []
    expanded_pos = []
    expanded_neg = []
    for u, pos, negs in zip(users, pos_items, neg_items_list):
        for neg in negs:
            expanded_users.append(u)
            expanded_pos.append(pos)
            expanded_neg.append(neg)
    return torch.LongTensor(expanded_users), torch.LongTensor(expanded_pos), torch.LongTensor(expanded_neg)

# ================= 2. Core Data Processor =================
class DataProcessor:
    def __init__(self, path, dataset_type='yelp', batch_size=2048):
        # Core Enhancement: Check if the 'path' parameter is a dictionary for compatibility
        if isinstance(path, dict):
            config = path
            # Core Enhancement: Extract parameters from the dictionary, use default values if not present
            self.path = config.get('data_path', './data/yelp_processed_for_meta')
            self.dataset_type = config.get('dataset', 'yelp')
            self.batch_size = config.get('batch_size', 2048)
            self.dual_aux = config.get('dual_aux', False)  # Flag to enable dual auxiliary tasks
            self.simulation_conflict = config.get('simulation_conflict', False)
        else:
            # Maintain the original string parameter passing logic
            self.path = path
            self.dataset_type = dataset_type
            self.batch_size = batch_size
            self.dual_aux = False  # Disabled by default for string paths
            self.simulation_conflict = False

        if self.dataset_type == 'amazon' and 'data/' not in self.path:
            self.path = './data/amazon_books_processedDataV3'
        elif self.dataset_type == 'yelp' and 'data/' not in self.path:
            self.path = './data/yelp_processed_for_meta'

        print(f"[INFO] Loading {self.dataset_type.upper()} dataset from {self.path}...")


        # 1. Preload all files
        self.train_data = self._load_interactions(os.path.join(self.path, "train.txt"))
        test_data = self._load_interactions(os.path.join(self.path, "test.txt"))

        meta_val_file = os.path.join(self.path, "meta_val.txt")
        if os.path.exists(meta_val_file):
            self.meta_val_data = self._load_interactions(meta_val_file)
            print(f"[INFO] Loaded {len(self.meta_val_data)} interactions for Meta-Validation.")
        else:
            self.meta_val_data = []
            print(f"[WARNING] {meta_val_file} not found. Fallback to train data for Meta-Validation.")

        # ================= Calculate global maximum IDs to prevent out-of-bounds errors =================
        all_data = self.train_data + test_data + self.meta_val_data
        self.n_users, self.n_items = self._get_counts(all_data)
        print(f"[INFO] Global Graph Size - Users: {self.n_users}, Items: {self.n_items}")

        # 2. Construct dictionary
        self.train_dict = self._build_user_item_dict(self.train_data, as_set=True)
        self.test_dict = self._build_user_item_dict(test_data, as_set=False)

        # --- Dynamically set the number of training negative samples based on dataset type (20 for Amazon, 10 for Yelp)---
        if self.dataset_type == 'amazon':
            self.train_neg_count = 20
        else:
            self.train_neg_count = 10

        # 3. Create PyTorch DataLoaders (accelerated with multi-threading + pinned memory)
        self.train_loader = DataLoader(
            RecDataset(self.train_data, self.train_dict, self.n_items, neg_count=self.train_neg_count),
            batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
            collate_fn=collate_fn
        )

        if len(self.meta_val_data) > 0:
            self.meta_val_loader = DataLoader(
                RecDataset(self.meta_val_data, self.train_dict, self.n_items, neg_count=1),
                batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True,
                collate_fn=collate_fn
            )
        else:
            self.meta_val_loader = None

        # 4. Construct the normalized adjacency matrix required by LightGCN
        self.norm_adj = self._build_norm_adj()

        # 5. Construct sampling pools for auxiliary tasks
        self._prepare_auxiliary_sampler()

        # Determine whether to generate secondary auxiliary edges based on the dual_aux flag
        if self.dual_aux:
            self._prepare_user_auxiliary()
            self._prepare_item_auxiliary()
        else:
            # Ensure these attributes exist even if not generated, preventing errors in subsequent sampling functions
            self.user_user_links = np.empty((0, 2), dtype=int)
            self.item_item_links = np.empty((0, 2), dtype=int)

    def _prepare_auxiliary_sampler(self):
        """Prepare social edges for Yelp and item co-occurrence associations for Amazon"""
        if self.dataset_type == 'yelp':
            social_file = os.path.join(self.path, "social.txt")
            if os.path.exists(social_file):
                # ================= Manually parse the adjacency list formatted social.txt =================
                social_links = []
                try:
                    with open(social_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) > 1:
                                u1 = int(parts[0])
                                for u2_str in parts[1:]:
                                    social_links.append([u1, int(u2_str)])

                    raw_links = np.array(social_links, dtype=int)
                    # Strictly filter out-of-bounds nodes
                    valid_mask = (raw_links[:, 0] < self.n_users) & (raw_links[:, 1] < self.n_users)
                    self.aux_links = raw_links[valid_mask]
                    print(f"[INFO] Yelp Auxiliary Task: Loaded {len(self.aux_links)} valid User-User social links.")
                except Exception as e:
                    print(f"[WARNING] Social file load failed: {e}. Using empty auxiliary links.")
                    self.aux_links = np.empty((0, 2), dtype=int)
            else:
                print("[WARNING] Yelp social.txt not found.")
                self.aux_links = np.empty((0, 2), dtype=int)
        else:
            # Keep the Amazon logic unchanged
            self.users_with_multi_interact = [
                u for u, items in self.train_dict.items() if len(items) >= 2
            ]
            self.aux_links = np.empty((0, 2), dtype=int)
            print(
                f"[INFO] Amazon Auxiliary Task: Found {len(self.users_with_multi_interact)} users for Item Co-occurrence sampling.")

    def _prepare_user_auxiliary(self):
        """Construct user-user co-occurrence edges (based on the number of co-purchased items) - Efficient implementation using inverted indexes"""
        print("[INFO] Building User-User co-occurrence edges (efficient inverted index)...")
        threshold = 3 if self.dataset_type == 'amazon' else 2

        # 1. Build an inverted index mapping items to user lists
        item_to_users = defaultdict(list)
        for user, items in self.train_dict.items():
            for item in items:
                item_to_users[item].append(user)

        # 2. Aggregate user pairs by item and count co-occurrence frequencies
        user_pair_count = defaultdict(int)
        for users in item_to_users.values():
            if len(users) < 2:
                continue
            # Generate pairwise combinations of all users associated with the same item
            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    u1, u2 = users[i], users[j]
                    # Maintain order and avoid duplicates (u1,u2) / (u2,u1)
                    if u1 > u2:
                        u1, u2 = u2, u1
                    user_pair_count[(u1, u2)] += 1

        # 3. Filter user pairs exceeding the threshold
        user_links = [[u1, u2] for (u1, u2), cnt in user_pair_count.items() if cnt >= threshold]

        self.user_user_links = np.array(user_links) if user_links else np.empty((0, 2), dtype=int)
        print(f"[INFO] Generated {len(self.user_user_links)} User-User co-occurrence edges (efficient).")

    def _prepare_item_auxiliary(self):
        """Construct item-item co-occurrence edges (based on the number of co-purchasing users)"""
        print("[INFO] Building Item-Item co-occurrence edges...")
        # First build the item->users inverted index
        item_to_users = defaultdict(set)
        for user, items in self.train_dict.items():
            for item in items:
                item_to_users[item].add(user)
        item_list = list(item_to_users.keys())
        item_links = []
        threshold = 3 if self.dataset_type == 'amazon' else 2
        for i in range(len(item_list)):
            i1 = item_list[i]
            users1 = item_to_users[i1]
            for j in range(i + 1, len(item_list)):
                i2 = item_list[j]
                users2 = item_to_users[i2]
                common = len(users1 & users2)
                if common >= threshold:
                    item_links.append([i1, i2])
        self.item_item_links = np.array(item_links) if item_links else np.empty((0, 2), dtype=int)
        print(f"[INFO] Generated {len(self.item_item_links)} Item-Item co-occurrence edges.")


    def sample_aux_links(self, batch_size):
        """
        Dynamically sample edges for auxiliary tasks
        Return: node1_tensor, node2_tensor
        """
        if self.dataset_type == 'amazon':
            # Item-Item 共现采样
            if len(self.users_with_multi_interact) == 0:
                return torch.randint(0, self.n_items, (batch_size,)), torch.randint(0, self.n_items, (batch_size,))

            sampled_users = np.random.choice(self.users_with_multi_interact, batch_size, replace=True)
            node1_list, node2_list = [], []
            for u in sampled_users:
                items = list(self.train_dict[u])
                # Randomly sample two distinct books purchased by the user
                i1, i2 = np.random.choice(items, 2, replace=False)
                node1_list.append(i1)
                node2_list.append(i2)
            return torch.LongTensor(node1_list), torch.LongTensor(node2_list)

        else:
            # Yelp User-User social sampling
            if len(self.aux_links) > 0:
                indices = np.random.randint(0, len(self.aux_links), size=batch_size)
                sampled_links = self.aux_links[indices]
                node1 = torch.LongTensor(sampled_links[:, 0])
                node2 = torch.LongTensor(sampled_links[:, 1])
                return node1, node2
            else:
                # Fallback: Random sampling
                return torch.randint(0, self.n_users, (batch_size,)), torch.randint(0, self.n_users, (batch_size,))

    def sample_user_aux_links(self, batch_size):
        """Uniformly sample user-user edges"""
        if len(self.user_user_links) == 0:
            # If unavailable, randomly sample user nodes as fallback
            return torch.randint(0, self.n_users, (batch_size,)), torch.randint(0, self.n_users, (batch_size,))
        indices = np.random.randint(0, len(self.user_user_links), size=batch_size)
        edges = self.user_user_links[indices]
        return torch.LongTensor(edges[:, 0]), torch.LongTensor(edges[:, 1])

    def sample_item_aux_links(self, batch_size):
        """Uniformly sample item-item edges"""
        if len(self.item_item_links) == 0:
            # fallback: Randomly sample items
            return torch.randint(0, self.n_items, (batch_size,)), torch.randint(0, self.n_items, (batch_size,))
        indices = np.random.randint(0, len(self.item_item_links), size=batch_size)
        edges = self.item_item_links[indices]
        return torch.LongTensor(edges[:, 0]), torch.LongTensor(edges[:, 1])

    # ================= Basic Utility Functions =================

    def _load_interactions(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    user = int(parts[0])
                    for item in parts[1:]:
                        data.append([user, int(item)])
        return data

    def _build_user_item_dict(self, data, as_set=False):
        ui_dict = defaultdict(set if as_set else list)
        for u, i in data:
            if as_set:
                ui_dict[u].add(i)
            else:
                ui_dict[u].append(i)
        return ui_dict

    def _get_counts(self, data):
        users = [d[0] for d in data]
        items = [d[1] for d in data]
        return max(users) + 1, max(items) + 1

    def _build_norm_adj(self):
        """Construct the Laplacian matrix (Ã) of LightGCN"""
        print("[INFO] Building normalized adjacency matrix for Graph Models...")
        n_nodes = self.n_users + self.n_items

        users = [d[0] for d in self.train_data]
        items = [d[1] for d in self.train_data]

        row = np.array(users + [i + self.n_users for i in items])
        col = np.array([i + self.n_users for i in items] + users)
        data = np.ones(len(row), dtype=np.float32)

        adj_mat = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv = np.zeros_like(rowsum)
        valid_idx = rowsum > 0
        d_inv[valid_idx] = np.power(rowsum[valid_idx], -0.5)
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocoo()

        indices = torch.LongTensor(np.vstack((norm_adj.row, norm_adj.col)))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)

        print("[INFO] Graph construction done!")
        return torch.sparse_coo_tensor(indices, values, shape)