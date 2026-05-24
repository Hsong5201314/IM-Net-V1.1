import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import copy


class IMNetTrainer:

    def __init__(self, data_processor, model, config, device):
        self.dp = data_processor
        self.device = device
        self.config = config
        self.model = model

        self.model_name = config.get('model_name', 'LightGCN')
        self.is_yelp = (self.config.get('dataset') == 'yelp')
        self.is_amazon = (self.config.get('dataset') == 'amazon')
        self.is_lightgcn_meta = (self.model_name == 'LightGCN')
        self.mode = config.get('mode', 'meta')
        self.is_random_dynamic = (self.mode == 'random_dynamic')

        # ================= Dynamic meta-learning control =================
        self.meta_active = False
        self.meta_cool_down = 0
        self.consecutive_decreases = 0
        self.best_metric = 0.0
        self.best_model_state = None
        self.pre_intervention_state = None
        if self.is_amazon and self.mode == 'meta':
            self.meta_active = True
            self.pre_intervention_state = copy.deepcopy(self.model.state_dict())
            print("[Amazon] Meta-learning activated from the start.")
        self.meta_intervention_steps = 0
        self.meta_max_steps = config.get('meta_max_steps', 5)
        self.meta_recovery_threshold = config.get('meta_recovery_threshold', 0.001)
        self.meta_decrease_window = config.get('meta_decrease_window', 3)
        # ================= Disable lookahead =================
        self.disable_lookahead = config.get('disable_lookahead', False)
        # =================Dynamic mode switching (reactive for the first N rounds, then lookahead mode) =================
        self.disable_lookahead_original = self.disable_lookahead  # Back up original configurations
        self.current_disable_lookahead = self.disable_lookahead  # Currently adopted mode
        self.lookahead_switch_epoch = config.get('lookahead_switch_epoch', None)
        if self.lookahead_switch_epoch is not None:
            print(f"[Dynamic Mode] Reactive before epoch {self.lookahead_switch_epoch}, then switch to Proactive.")

        # ================= Loss explosion detection =================
        self.avg_loss_before_meta = None
        # ================= Minimum trigger parameter =================
        self.last_best_update_epoch = 0
        self.force_meta_patience = config.get('force_meta_patience', 8)
        self.force_meta_start_epoch = config.get('force_meta_start_epoch', 50)
        self.force_meta_epoch = config.get('force_meta_epoch', None)
        # Early stopping parameters
        self.early_stop_patience = config.get('early_stop_patience', 0)
        self.no_improve_epochs = 0
        # ================= HVP step size control =================
        self.hvp_eps = config.get('hvp_eps', 0.01)  # Base step size
        self.hvp_max_eps = config.get('hvp_max_eps', 5e-4)  # Maximum allowable step size
        # ================= Dedicated stability parameters for lookahead mode =================
        self.proactive_hvp_eps = config.get('proactive_hvp_eps', self.hvp_eps)
        self.proactive_hvp_eps_original = self.proactive_hvp_eps
        self.proactive_hvp_max_eps = config.get('proactive_hvp_max_eps', self.hvp_max_eps)
        self.proactive_meta_loss_weight = config.get('proactive_meta_loss_weight',
                                                     config.get('meta_loss_weight', 1.0))
        self.adaptive_epsilon = config.get('adaptive_epsilon', True)
        self.epsilon_base = config.get('epsilon_base', 1e-5)
        self.multi_step_lookahead = config.get('multi_step_lookahead', 1)  # 1: Single step, >1: Multi-step

        self.proactive_loss_explosion_ratio = config.get('proactive_loss_explosion_ratio', 10.0)


        # ================= EMA (Exponential Moving Average) =================
        self.use_ema = config.get('use_ema', False)  # Default is False, no impact on original operation
        if self.use_ema:
            self.ema_decay = config.get('ema_decay', 0.999)
            self.ema_model = copy.deepcopy(model)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad_(False)
            print(f"[EMA] EMA enabled with decay={self.ema_decay}")

        # ================= Gradient accumulation support =================
        self.grad_accumulation_steps = config.get('grad_accumulation_steps', 1)
        if self.grad_accumulation_steps > 1:
            print(f"[GradAccum] Gradient accumulation enabled with {self.grad_accumulation_steps} steps")

        # ================= Gradient Warmup =================
        self.use_warmup = config.get('use_warmup', False)
        self.warmup_epochs = config.get('warmup_epochs', 0)
        self.initial_lr = config.get('lr', 0.001)
        if self.use_warmup and self.warmup_epochs > 0:
            print(f"[Warmup] Learning rate warmup enabled for {self.warmup_epochs} epochs")
            self.current_warmup_epoch = 0

        # Initialize IMNet
        from core.imnet import IMNet
        self.imnet = IMNet(num_tasks=4).to(device)

        # Check necessary configuration items
        if 'grad_clip_norm' not in self.config:
            self.config['grad_clip_norm'] = 1.0
            print(f"[Init] Set default grad_clip_norm to 1.0")

        if 'loss_explosion_threshold' not in self.config:
            self.config['loss_explosion_threshold'] = 3.0
            print(f"[Init] Set default loss_explosion_threshold to 3.0")

        # Fixed meta-validation batch (optional)
        if self.config.get('fixed_meta_val_batch', False):
            try:
                self.fixed_meta_val_batch = next(iter(self.dp.meta_val_loader))
                print("[Expert Info] Using fixed meta-validation batch.")
            except StopIteration:
                print("[WARNING] Could not create fixed meta-val batch, fallback to random sampling.")
                self.fixed_meta_val_batch = None
        else:
            self.fixed_meta_val_batch = None

        # Optimizer
        wd = config.get('wd', 1e-4)
        if self.model_name == 'NCF':
            self.optimizer_model = optim.AdamW(self.model.parameters(), lr=config['lr'], weight_decay=wd)
        else:
            self.optimizer_model = optim.AdamW(self.model.parameters(), lr=config['lr'])

        if self.is_amazon:
            default_meta_lr = 5e-4
        else:
            default_meta_lr = 1e-8
        self.optimizer_meta = optim.AdamW(self.imnet.parameters(), lr=config.get('meta_lr', default_meta_lr))

        # Select scheduler based on whether it is lookahead mode
        if not self.disable_lookahead:
            # Lookahead mode: Use cosine annealing restart scheduler to help escape local optima
            self.scheduler_model = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer_model, T_0=50, T_mult=2, eta_min=1e-5
            )
            self.scheduler_meta = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer_meta, T_0=50, T_mult=2, eta_min=1e-7
            )
        else:
            # Non-lookahead mode (reactive, Static, etc.): Use ReduceLROnPlateau
            if self.is_lightgcn_meta and self.is_yelp:
                self.scheduler_model = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_model, mode='max',
                                                                            factor=0.5, patience=12)
                self.scheduler_meta = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_meta, mode='max', factor=0.5,
                                                                           patience=8)
            else:
                self.scheduler_model = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_model, mode='max',
                                                                            factor=0.5, patience=8)
                self.scheduler_meta = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_meta, mode='max', factor=0.5,
                                                                           patience=5)

        # Meta-validation loader (initial placeholder)
        if hasattr(self.dp, 'meta_val_loader') and self.dp.meta_val_loader is not None:
            self.meta_val_loader = iter(self.dp.meta_val_loader)
            print("[Expert Info] Using provided meta_val_loader.")
        else:
            print("[WARNING] meta_val_loader is None. Attempting to create a proxy Meta-Val set from train_loader.")
            all_train_batches = list(self.dp.train_loader)
            val_size = max(1, len(all_train_batches) // 10)
            self.meta_val_samples = all_train_batches[-val_size:]
            self.meta_val_loader = iter(self.meta_val_samples)
            print(f"[Critical Fix] Created a proxy Meta-Val set of {len(self.meta_val_samples)} batches.")

        # Cache training batch list for dynamic meta-validation sampling (used after meta-learning is activated)
        self._train_batches_cache = None
        self.meta_update_freq = config.get('meta_update_freq', 10)
        self.meta_batch_counter = 0


        # ===== Validation loss cache =====
        self.cached_val_loss = None
        self.cached_val_step = -100
        self.val_cache_freq = config.get('val_cache_freq', 10)

        self.use_fast_validation = config.get('fast_validation', True)
        if self.use_fast_validation:
            # Reduce validation batch quantity
            self.val_batch_count = 1
            # Validation loss cache
            self.cached_val_loss = None
            self.cached_val_step = -100
            self.val_cache_freq = config.get('val_cache_freq', 10)
            # Multi-step validation loss cache
            self.cached_first_val_loss = None
            self.cached_last_val_loss = None
            print(
                f"[Optimization] Fast validation: {self.val_batch_count} batch(es), cache every {self.val_cache_freq} steps")
        else:
            self.val_batch_count = 5
            self.cached_val_loss = None
            self.val_cache_freq = 1
            self.cached_first_val_loss = None
            self.cached_last_val_loss = None



    # ================= EMA update method =================
    def update_ema(self, decay=None):
        """Update EMA model parameters (effective only when enabled)"""
        if not self.use_ema:
            return
        if decay is None:
            decay = self.ema_decay
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)

    # ================= Evaluation method using the EMA model =================
    @torch.no_grad()
    def evaluate_with_ema(self, top_k=20):
        """Evaluate using the EMA model (only when enabled)"""
        if not self.use_ema:
            return self.evaluate(top_k)

        self.ema_model.eval()
        from main_gpu import get_metrics
        adj = self.dp.norm_adj.to(self.device) if hasattr(self.dp, 'norm_adj') else None
        recall, ndcg = get_metrics(
            self.ema_model, self.config['model_name'], adj,
            self.dp.train_dict, self.dp.test_dict, self.device, top_k=top_k
        )
        return recall, ndcg

    # ================= Learning rate warmup method =================
    def apply_warmup(self, epoch):
        """Apply learning rate warmup"""
        if not self.use_warmup or self.warmup_epochs == 0:
            return
        if epoch <= self.warmup_epochs:
            warmup_factor = epoch / self.warmup_epochs
            current_lr = self.initial_lr * warmup_factor
            for param_group in self.optimizer_model.param_groups:
                param_group['lr'] = current_lr
            if epoch != self.current_warmup_epoch:
                self.current_warmup_epoch = epoch
                print(f"[Warmup] Epoch {epoch}: LR = {current_lr:.6f}")

    # ================= Original method =================
    def _nash_mtl_gradient(self, task_grads_flat, max_iter=20, lr=0.1):
        """
        Nash-MTL: Nash Bargaining Gradient Aggregation
        task_grads_flat : list of 1D tensors, each flatten gradient of a task.
        Returns: (combined_gradient_1d, task_weights)
        """
        num_tasks = len(task_grads_flat)
        # Construct Gram matrix
        G = torch.zeros(num_tasks, num_tasks, device=task_grads_flat[0].device)
        for i in range(num_tasks):
            for j in range(num_tasks):
                G[i, j] = torch.dot(task_grads_flat[i], task_grads_flat[j])
        # Solve via projected gradient method λ
        λ = torch.ones(num_tasks, device=G.device) / num_tasks
        for _ in range(max_iter):
            grad = torch.mv(G, λ)
            λ_new = λ - lr * grad
            # Project onto the simplex
            λ_new = torch.clamp(λ_new, min=0)
            λ_new = λ_new / (λ_new.sum() + 1e-8)
            λ = λ_new
        combined = sum(λ[i] * task_grads_flat[i] for i in range(num_tasks))
        return combined, λ

    def _compute_losses_meta(self, batch):

        users = self._to_tensor(batch[0])
        pos_items = self._to_tensor(batch[1])
        neg_items = self._to_tensor(batch[2])

        is_amazon = self.is_amazon
        is_yelp = self.is_yelp
        model_name = self.model_name

        # ================= NCF-specific branch =================
        if model_name == 'NCF':
            pos_scores, neg_scores = self.model(users, pos_items, neg_items)

            # BPR loss
            bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

            # L2 regularization (not divided by batch size)
            reg_loss = sum(p.norm(2).pow(2) for p in self.model.parameters() if p.requires_grad)
            reg_loss = reg_loss * self.config.get('wd', 1e-4)

            main_loss = bpr_loss + reg_loss

            # Auxiliary loss (using embedding inner product)
            aux_loss = torch.tensor(0.0, device=self.device)
            aux_loss2 = torch.tensor(0.0, device=self.device)

            # If auxiliary links exist, add embedding similarity loss
            if hasattr(self.dp, 'aux_links') and self.dp.aux_links is not None and len(self.dp.aux_links) > 0:
                u_emb, i_emb = self.model.get_all_embeddings()
                batch_size_aux = min(users.shape[0], len(self.dp.aux_links))
                idx = np.random.choice(len(self.dp.aux_links), batch_size_aux, replace=False)
                node1 = torch.tensor(self.dp.aux_links[idx, 0], dtype=torch.long, device=self.device)
                node2 = torch.tensor(self.dp.aux_links[idx, 1], dtype=torch.long, device=self.device)

                if is_amazon:
                    node1 = node1 % i_emb.size(0)
                    node2 = node2 % i_emb.size(0)
                    emb1, emb2 = i_emb[node1], i_emb[node2]
                else:
                    node1 = node1 % u_emb.size(0)
                    node2 = node2 % u_emb.size(0)
                    emb1, emb2 = u_emb[node1], u_emb[node2]

                aux_loss = -F.logsigmoid(torch.sum(emb1 * emb2, dim=1)).mean()

            cl_loss = torch.tensor(0.0, device=self.device)
            avg_score_diff = (pos_scores - neg_scores).mean()

            return torch.stack([main_loss, aux_loss, aux_loss2, cl_loss]), avg_score_diff

        # ================= LightGCN/SimGCL branch =================
        adj = self.dp.norm_adj.to(self.device) if hasattr(self.dp,
                                                          'norm_adj') and self.dp.norm_adj is not None else None

        # Dataset-adaptive hyperparameters
        if self.is_lightgcn_meta and is_yelp:
            cl_temp = self.config.get('cl_temp_yelp', 0.3)
            margin = self.config.get('margin_yelp', 0.05)
            reg_val = self.config.get('reg_val_yelp', 5e-4)
            cl_eps = self.config.get('cl_eps_yelp', 0.05)
        elif self.is_lightgcn_meta and is_amazon:
            cl_temp = self.config.get('cl_temp_amazon', 0.07)
            margin = self.config.get('margin_amazon', 0.15)
            reg_val = self.config.get('reg_val_amazon', 2e-3)
            cl_eps = self.config.get('cl_eps_amazon', 0.15)
        else:
            cl_temp = self.config.get('cl_temp', 0.2)
            margin = 0.0
            reg_val = self.config.get('wd', 1e-4)
            cl_eps = self.config.get('cl_eps', 0.1)

        # Obtain embeddings
        if model_name == 'SimGCL':
            u_emb, i_emb = self.model.get_all_embeddings(adj, perturbed=False)
            u_v1, i_v1 = self.model.get_all_embeddings(adj, perturbed=True)
            u_v2, i_v2 = self.model.get_all_embeddings(adj, perturbed=True)
        else:
            # LightGCN
            u_emb, i_emb = self.model.get_all_embeddings(adj)
            eps = cl_eps
            u_v1, i_v1 = u_emb + torch.randn_like(u_emb) * eps, i_emb + torch.randn_like(i_emb) * eps
            u_v2, i_v2 = u_emb + torch.randn_like(u_emb) * eps, i_emb + torch.randn_like(i_emb) * eps

        # Dropout when meta-learning is activated
        if self.meta_active and self.model.training:
            if self.is_amazon:
                u_emb = F.dropout(u_emb, p=0.4, training=True)
                i_emb = F.dropout(i_emb, p=0.4, training=True)
                noise_std = 0.05
                u_emb = u_emb + torch.randn_like(u_emb) * noise_std
                i_emb = i_emb + torch.randn_like(i_emb) * noise_std
            else:
                u_emb = F.dropout(u_emb, p=0.2, training=True)
                i_emb = F.dropout(i_emb, p=0.2, training=True)

        # contrastive loss
        unique_users, unique_items = torch.unique(users), torch.unique(pos_items)
        u_cl_loss = self.info_nce(u_v1[unique_users], u_v2[unique_users], cl_temp)
        i_cl_loss = self.info_nce(i_v1[unique_items], i_v2[unique_items], cl_temp)
        cl_loss = u_cl_loss + i_cl_loss

        # BPR loss
        batch_u, batch_pos, batch_neg = u_emb[users], i_emb[pos_items], i_emb[neg_items]
        pos_scores = torch.sum(batch_u * batch_pos, dim=1)
        neg_scores = torch.sum(batch_u * batch_neg, dim=1)

        if self.is_lightgcn_meta:
            bpr_loss = -F.logsigmoid(pos_scores - neg_scores - margin).mean()
        else:
            bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        reg_loss = (1 / 2) * (batch_u.norm(2).pow(2) + batch_pos.norm(2).pow(2) + batch_neg.norm(2).pow(2)) / float(
            users.shape[0])
        main_loss = bpr_loss + reg_val * reg_loss

        # Conflict simulation branch
        if hasattr(self.dp, 'simulation_conflict') and self.dp.simulation_conflict:
            conflict_scale = self.config.get('conflict_scale', 1.0)
            aux_loss = -F.logsigmoid(neg_scores - pos_scores).mean() * conflict_scale
            aux_loss2 = torch.tensor(0.0, device=self.device)
            cl_loss = torch.tensor(0.0, device=self.device)
            avg_score_diff = (pos_scores - neg_scores).mean()
            return torch.stack([main_loss, aux_loss, aux_loss2, cl_loss]), avg_score_diff

        # Auxiliary loss 1
        aux_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self.dp, 'aux_links') and self.dp.aux_links is not None and len(self.dp.aux_links) > 0:
            batch_size_aux = users.shape[0]
            idx = np.random.choice(len(self.dp.aux_links), batch_size_aux, replace=True)
            node1 = torch.tensor(self.dp.aux_links[idx, 0], dtype=torch.long, device=self.device)
            node2 = torch.tensor(self.dp.aux_links[idx, 1], dtype=torch.long, device=self.device)
            max_node_idx_u = u_emb.size(0)
            max_node_idx_i = i_emb.size(0)
            if is_amazon:
                node1 = node1 % max_node_idx_i
                node2 = node2 % max_node_idx_i
                emb1, emb2 = i_emb[node1], i_emb[node2]
            else:
                node1 = node1 % max_node_idx_u
                node2 = node2 % max_node_idx_u
                emb1, emb2 = u_emb[node1], u_emb[node2]
            aux_loss = -F.logsigmoid(torch.sum(emb1 * emb2, dim=1)).mean()

        # Auxiliary loss 2
        aux_loss2 = torch.tensor(0.0, device=self.device)
        if self.is_yelp and hasattr(self.dp, 'item_item_links') and len(self.dp.item_item_links) > 0:
            node1, node2 = self.dp.sample_item_aux_links(users.shape[0])
            node1, node2 = node1.to(self.device), node2.to(self.device)
            emb1, emb2 = i_emb[node1], i_emb[node2]
            aux_loss2 = -F.logsigmoid(torch.sum(emb1 * emb2, dim=1)).mean()
        elif self.is_amazon and hasattr(self.dp, 'user_user_links') and len(self.dp.user_user_links) > 0:
            node1, node2 = self.dp.sample_user_aux_links(users.shape[0])
            node1, node2 = node1.to(self.device), node2.to(self.device)
            emb1, emb2 = u_emb[node1], u_emb[node2]
            aux_loss2 = -F.logsigmoid(torch.sum(emb1 * emb2, dim=1)).mean()

        avg_score_diff = (pos_scores - neg_scores).mean()
        return torch.stack([main_loss, aux_loss, aux_loss2, cl_loss]), avg_score_diff

    def _compute_losses(self, batch):
        if hasattr(self.dp, 'simulation_conflict') and self.dp.simulation_conflict:

            users = self._to_tensor(batch[0])
            pos_items = self._to_tensor(batch[1])
            neg_items = self._to_tensor(batch[2])

            is_amazon = (self.config.get('dataset') == 'amazon')
            model_name = self.config.get('model_name', 'LightGCN')
            adj = self.dp.norm_adj.to(self.device) if hasattr(self.dp,
                                                              'norm_adj') and self.dp.norm_adj is not None else None

            if model_name == 'SimGCL':
                u_emb, i_emb = self.model.get_all_embeddings(adj, perturbed=False)
            elif model_name == 'NCF':
                u_emb, i_emb = self.model.get_all_embeddings()
            else:
                u_emb, i_emb = self.model.get_all_embeddings(adj)

            batch_u = u_emb[users]
            batch_pos = i_emb[pos_items]
            batch_neg = i_emb[neg_items]
            pos_scores = torch.sum(batch_u * batch_pos, dim=1)
            neg_scores = torch.sum(batch_u * batch_neg, dim=1)
            bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            reg_loss = (1 / 2) * (batch_u.norm(2).pow(2) + batch_pos.norm(2).pow(2) + batch_neg.norm(2).pow(2)) / float(
                users.shape[0])
            main_loss = bpr_loss + self.config.get('wd', 1e-4) * reg_loss
            conflict_scale = self.config.get('conflict_scale', 1.0)
            aux_loss = -F.logsigmoid(neg_scores - pos_scores).mean() * conflict_scale

            cl_loss = torch.tensor(0.0, device=self.device)
            return torch.stack([main_loss, aux_loss, cl_loss])
        else:

            users = self._to_tensor(batch[0])
            pos_items = self._to_tensor(batch[1])
            neg_items = self._to_tensor(batch[2])

            is_amazon = (self.config.get('dataset') == 'amazon')
            model_name = self.config.get('model_name', 'LightGCN')
            adj = self.dp.norm_adj.to(self.device) if hasattr(self.dp,
                                                              'norm_adj') and self.dp.norm_adj is not None else None
            if model_name == 'SimGCL':
                u_emb, i_emb = self.model.get_all_embeddings(adj, perturbed=False)
            elif model_name == 'NCF':
                u_emb, i_emb = self.model.get_all_embeddings()
            else:
                u_emb, i_emb = self.model.get_all_embeddings(adj)

            cl_loss = torch.tensor(0.0, device=self.device)
            if model_name == 'SimGCL':
                u_v1, i_v1 = self.model.get_all_embeddings(adj, perturbed=True)
                u_v2, i_v2 = self.model.get_all_embeddings(adj, perturbed=True)
                cl_temp = self.config.get('cl_temp', 0.2)
                unique_users = torch.unique(users)
                unique_items = torch.unique(pos_items)
                u_cl_loss = self.info_nce(u_v1[unique_users], u_v2[unique_users], cl_temp)
                i_cl_loss = self.info_nce(i_v1[unique_items], i_v2[unique_items], cl_temp)
                cl_loss = u_cl_loss + i_cl_loss

            batch_u, batch_pos, batch_neg = u_emb[users], i_emb[pos_items], i_emb[neg_items]
            pos_scores = torch.sum(batch_u * batch_pos, dim=1)
            neg_scores = torch.sum(batch_u * batch_neg, dim=1)
            bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            reg_loss = (1 / 2) * (batch_u.norm(2).pow(2) + batch_pos.norm(2).pow(2) + batch_neg.norm(2).pow(2)) / float(
                users.shape[0])
            main_loss = bpr_loss + self.config.get('wd', 1e-4) * reg_loss

            node1, node2 = self.dp.sample_aux_links(users.shape[0])
            node1, node2 = node1.to(self.device), node2.to(self.device)
            emb1, emb2 = (i_emb[node1], i_emb[node2]) if is_amazon else (u_emb[node1], u_emb[node2])
            aux_loss = -F.logsigmoid(torch.sum(emb1 * emb2, dim=1)).mean()

            return torch.stack([main_loss, aux_loss, cl_loss])

    def info_nce(self, z1, z2, temp):
        z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
        pos_sim = torch.sum(z1 * z2, dim=-1)
        sim_matrix = torch.matmul(z1, z2.t())
        matrix_exp = torch.exp(sim_matrix / temp)
        pos_exp = torch.exp(pos_sim / temp)
        return -torch.log(pos_exp / (matrix_exp.sum(dim=-1) + 1e-8)).mean()

    def train_epoch(self, epoch, mode='meta'):
        # Apply Learning Rate Warmup
        self.apply_warmup(epoch)

        self.model.train()

        # Dynamically increase perturbation in later training stages (only for look-ahead mode and when meta-learning is activated)
        if not self.disable_lookahead and self.meta_active:
            total_epochs = self.config.get('epochs', 400)
            if epoch > 0.8 * total_epochs:
                new_eps = min(self.proactive_hvp_eps_original * 2, 1e-3)
                if self.proactive_hvp_eps < new_eps:
                    self.proactive_hvp_eps = new_eps
                    print(f"[Dynamic] Increased proactive_hvp_eps to {self.proactive_hvp_eps:.2e} at epoch {epoch}")

        # Dynamic Switching Logic
        if self.lookahead_switch_epoch is not None:
            if epoch >= self.lookahead_switch_epoch:
                self.current_disable_lookahead = not self.disable_lookahead_original
            else:
                self.current_disable_lookahead = self.disable_lookahead_original
        else:
            self.current_disable_lookahead = self.disable_lookahead

        total_loss = 0
        pbar = tqdm(self.dp.train_loader, desc=f"Epoch {epoch}", leave=False)

        if mode == 'gradnorm' and not hasattr(self, 'gradnorm_weights'):
            self.gradnorm_weights = nn.Parameter(torch.ones(3, device=self.device))
            self.gradnorm_optimizer = optim.Adam([self.gradnorm_weights], lr=0.01)
            self.initial_losses = None

        # Gradient Accumulation Counter
        accumulation_counter = 0

        for batch_idx, batch in enumerate(pbar):
            # Do not perform zero_grad for every batch during gradient accumulation
            if accumulation_counter == 0:
                self.optimizer_model.zero_grad()

            if mode == 'fixed_weights':
                # Calculate the original training loss first
                train_losses_original, _ = self._compute_losses_meta(batch)
                # Fixed weight mode (true meta-learner-free baseline)
                fixed_weights = self.config.get('fixed_weights', [1.0, 0.1, 0.05, 0.02])
                fixed_weights = torch.tensor(fixed_weights, device=self.device)

                # Ensure the number of weights matches
                num_losses = len(train_losses_original)
                if len(fixed_weights) > num_losses:
                    fixed_weights = fixed_weights[:num_losses]
                elif len(fixed_weights) < num_losses:
                    # Pad zeros for insufficient parts
                    padding = torch.zeros(num_losses - len(fixed_weights), device=self.device)
                    fixed_weights = torch.cat([fixed_weights, padding])

                model_loss = torch.sum(fixed_weights * train_losses_original)
                final_loss = model_loss

                pbar.set_postfix({
                    'Phase': 'Fixed',
                    'wM': f"{fixed_weights[0].item():.2f}",
                    'wA1': f"{fixed_weights[1].item():.2f}",
                    'wA2': f"{fixed_weights[2].item():.2f}" if num_losses > 2 else 'N/A',
                    'wC': f"{fixed_weights[3].item():.2f}" if num_losses > 3 else 'N/A'
                })

                # Gradient backpropagation and optimization (same as Random Dynamic)
                if self.grad_accumulation_steps > 1:
                    final_loss = final_loss / self.grad_accumulation_steps
                    final_loss.backward()
                    accumulation_counter += 1
                    if accumulation_counter % self.grad_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer_model.step()
                        self.update_ema()
                        accumulation_counter = 0
                else:
                    final_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer_model.step()
                    self.update_ema()

                total_loss += model_loss.item()
                continue  # Skip the subsequent meta-learning logic

            elif mode == 'meta' or mode == 'random_dynamic':
                # Dynamic learning rate adjustment
                if self.is_lightgcn_meta and self.is_yelp and epoch > 30:
                    for param_group in self.optimizer_model.param_groups:
                        if param_group['lr'] > 0.0001:
                            param_group['lr'] = max(0.0001, param_group['lr'] * 0.95)

                # Obtain meta validation batches
                try:
                    val_batch = next(self.meta_val_loader)
                except StopIteration:
                    loader_source = self.meta_val_samples if hasattr(self, 'meta_val_samples') else self.dp.train_loader
                    self.meta_val_loader = iter(loader_source)
                    val_batch = next(self.meta_val_loader)

                # Compute original training loss
                train_losses_original, train_diff_orig = self._compute_losses_meta(batch)

                # Random Dynamic branch
                if self.is_random_dynamic:
                    w_main = random.uniform(0.01, 0.99)
                    w_aux = 1.0 - w_main
                    model_loss = w_main * train_losses_original[0] + w_aux * train_losses_original[1]
                    final_loss = model_loss
                    pbar.set_postfix({'Phase': 'Random', 'wM': f"{w_main:.2f}", 'wA': f"{w_aux:.2f}"})

                    # Gradient accumulation processing
                    if self.grad_accumulation_steps > 1:
                        final_loss = final_loss / self.grad_accumulation_steps
                        final_loss.backward()
                        accumulation_counter += 1
                        if accumulation_counter % self.grad_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.optimizer_model.step()
                            self.update_ema()
                            accumulation_counter = 0
                    else:
                        final_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer_model.step()
                        self.update_ema()

                    total_loss += model_loss.item()
                    continue

                # Warmup phase
                if not self.meta_active:
                    # Assign a higher initial weight to the auxiliary task (aux_loss2, index 2) to encourage the utilization of auxiliary information
                    weights = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
                    final_loss = torch.sum(weights * train_losses_original)
                    model_loss = final_loss
                    with torch.no_grad():
                        raw_w_obs, const_w_obs = self.imnet(train_losses_original.detach())
                    pbar.set_postfix(
                        {"Phase": "Warmup", "loss": f"{final_loss.item():.4f}", "rW_M": f"{raw_w_obs[0].item():.2f}"})

                    if self.grad_accumulation_steps > 1:
                        final_loss = final_loss / self.grad_accumulation_steps
                        final_loss.backward()
                        accumulation_counter += 1
                        if accumulation_counter % self.grad_accumulation_steps == 0:
                            if self.is_amazon and self.meta_active:
                                clip_norm = 0.5
                            elif self.is_lightgcn_meta and self.is_yelp:
                                clip_norm = 1.5
                            else:
                                clip_norm = 3.0
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
                            self.optimizer_model.step()
                            self.update_ema()
                            accumulation_counter = 0
                    else:
                        final_loss.backward()
                        if self.is_amazon and self.meta_active:
                            clip_norm = 0.5
                        elif self.is_lightgcn_meta and self.is_yelp:
                            clip_norm = 1.5
                        else:
                            clip_norm = 3.0
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
                        self.optimizer_model.step()
                        self.update_ema()

                    total_loss += model_loss.item()
                    continue

                # Logic after meta-learning activation (keep original, but add EMA update)
                if not hasattr(self, 'fixed_val_batches'):
                    self.fixed_val_batches = []
                    if hasattr(self.dp, 'meta_val_loader') and self.dp.meta_val_loader is not None:
                        for i, b in enumerate(self.dp.meta_val_loader):
                            if i >= 1:
                                break
                            self.fixed_val_batches.append(b)
                    else:
                        if self._train_batches_cache is None:
                            self._train_batches_cache = list(self.dp.train_loader)
                        self.fixed_val_batches = random.sample(self._train_batches_cache,
                                                               min(5, len(self._train_batches_cache)))
                    print(
                        f"[Meta] Using {len(self.fixed_val_batches)} fixed validation batches for meta-loss averaging.")

                self.meta_batch_counter += 1
                do_full_meta = (self.meta_batch_counter % self.meta_update_freq == 0)

                if self.meta_active and self.meta_intervention_steps <= 3:
                    do_full_meta = False

                if do_full_meta and not self.current_disable_lookahead:
                    # Multi-step Proactive Meta-learning integrated with Nash-MTL
                    raw_weights, weights = self.imnet(train_losses_original.detach())

                    if (self.is_yelp or self.is_amazon) and not self.current_disable_lookahead and self.meta_active:
                        weights = torch.sigmoid(weights) * 0.9 + 0.05
                    else:
                        weights = torch.sigmoid(weights) * 0.99 + 0.005

                    # Original validation loss (using cache, updated every val_cache_freq batches)
                    if self.meta_batch_counter - self.cached_val_step >= self.val_cache_freq:
                        val_loss_before = 0.0
                        for v_batch in self.fixed_val_batches:
                            v_losses, _ = self._compute_losses_meta(v_batch)
                            val_loss_before += v_losses[0]
                        val_loss_before /= len(self.fixed_val_batches)
                        self.cached_val_loss = val_loss_before
                        self.cached_val_step = self.meta_batch_counter
                    else:
                        val_loss_before = self.cached_val_loss

                    T = self.multi_step_lookahead
                    original_params = [p.detach().clone() for p in self.model.parameters()]
                    current_params = [p.clone() for p in self.model.parameters()]
                    cumulative_val_loss = 0.0

                    for step in range(T):
                        with torch.no_grad():
                            for param, curr in zip(self.model.parameters(), current_params):
                                param.data.copy_(curr)

                        train_losses_current, _ = self._compute_losses_meta(batch)
                        num_tasks = len(train_losses_current)

                        task_grads_flat = []
                        for i in range(num_tasks):
                            loss_i = train_losses_current[i]
                            grad_i = torch.autograd.grad(loss_i, self.model.parameters(),
                                                         retain_graph=True, allow_unused=True)
                            grad_i = [g if g is not None else torch.zeros_like(p)
                                      for g, p in zip(grad_i, self.model.parameters())]
                            flat_grad = torch.cat([g.flatten() for g in grad_i])
                            task_grads_flat.append(flat_grad)

                        combined_grad_flat, λ = self._nash_mtl_gradient(task_grads_flat, max_iter=20, lr=0.1)

                        offset = 0
                        combined_grads = []
                        for p in self.model.parameters():
                            numel = p.numel()
                            g = combined_grad_flat[offset:offset + numel].view(p.shape)
                            combined_grads.append(g)
                            offset += numel

                        grad_norm = torch.norm(combined_grad_flat, 2)
                        if self.adaptive_epsilon:
                            curvature_scale = 1.0 / (grad_norm + 1e-8)
                            curvature_scale = torch.clamp(curvature_scale, 0.1, 10.0)
                            base_eps = self.epsilon_base * curvature_scale
                            base_eps = torch.clamp(base_eps, min=1e-8, max=1e-4)
                        else:
                            # Simplify the Amazon branch: directly use proactive_hvp_eps as the base step size
                            if self.is_amazon and self.meta_active:
                                base_eps = self.proactive_hvp_eps
                                base_eps = torch.clamp(base_eps, min=1e-6, max=1e-3)
                            elif self.is_yelp and not self.disable_lookahead and self.meta_active:
                                decay = 0.85 ** (self.meta_intervention_steps + step)
                                max_eps = 5e-7
                                effective_eps = 5e-6
                                base_eps = min(effective_eps * decay / (grad_norm + 1e-8), max_eps)
                                base_eps = torch.clamp(base_eps, min=1e-8, max=max_eps)
                            else:
                                base_eps = min(self.proactive_hvp_eps / (grad_norm + 1e-8), 1e-5)
                                base_eps = torch.clamp(base_eps, min=1e-6, max=1e-5)
                        step_scale = 1.0 / (step + 1)
                        safe_eps = base_eps * step_scale




                        new_params = []
                        for param, grad in zip(current_params, combined_grads):
                            new_param = param - safe_eps * grad
                            new_params.append(new_param)
                        current_params = new_params

                        # Compute validation loss after current step (using temporary parameters)
                        with torch.no_grad():
                            for param, curr in zip(self.model.parameters(), current_params):
                                param.data.copy_(curr)

                        # Use cache: compute accurately only at the first and last steps, use estimation for intermediate steps
                        if step == 0 or step == T - 1:
                            val_loss_step = 0.0
                            for v_batch in self.fixed_val_batches:
                                v_losses, _ = self._compute_losses_meta(v_batch)
                                val_loss_step += v_losses[0]
                            val_loss_step /= len(self.fixed_val_batches)

                            # Cache the loss of the first and last steps
                            if step == 0:
                                self.cached_first_val_loss = val_loss_step
                            elif step == T - 1:
                                self.cached_last_val_loss = val_loss_step
                        else:
                            # Estimate intermediate steps via linear interpolation
                            progress = step / T
                            # Use the value of the first step temporarily if the last step value is unavailable
                            if self.cached_last_val_loss is None:
                                val_loss_step = self.cached_first_val_loss
                            else:
                                val_loss_step = self.cached_first_val_loss * (
                                            1 - progress) + self.cached_last_val_loss * progress

                        cumulative_val_loss += val_loss_step

                    # Finally roll back to the original parameters
                    with torch.no_grad():
                        for param, orig in zip(self.model.parameters(), original_params):
                            param.data.copy_(orig)

                    # Final perturbed validation loss (adopting optimized computation method)
                    if T > 1 and hasattr(self, 'cached_first_val_loss') and hasattr(self, 'cached_last_val_loss'):
                        # Use the average of the first and last steps
                        val_loss_after = (self.cached_first_val_loss + self.cached_last_val_loss) / 2
                    else:
                        val_loss_after = cumulative_val_loss / T


                    meta_loss = (val_loss_after - val_loss_before) / (val_loss_before + 1e-8)
                    meta_loss = torch.clamp(meta_loss, min=-1.0, max=1.0)

                    if meta_loss.item() > self.config.get('loss_explosion_threshold', 3.0):
                        print(f"[Warning] Meta loss explosion: {meta_loss.item():.2f}")
                        self.config['meta_loss_weight'] *= 0.5
                        continue

                    model_loss = torch.sum(weights * train_losses_original)

                    if not hasattr(self, 'model_loss_ema'):
                        self.model_loss_ema = model_loss.detach()
                        self.meta_loss_ema = meta_loss.detach()
                    self.model_loss_ema = 0.99 * self.model_loss_ema + 0.01 * model_loss.detach()
                    self.meta_loss_ema = 0.99 * self.meta_loss_ema + 0.01 * meta_loss.detach()
                    adaptive_weight = (self.model_loss_ema / (self.meta_loss_ema + 1e-8)).clamp(0.1, 5.0)

                    meta_warmup_epochs = self.config.get('meta_warmup_epochs', 3)
                    warmup_factor = min(1.0, self.meta_intervention_steps / meta_warmup_epochs)
                    current_meta_weight = self.proactive_meta_loss_weight * warmup_factor

                    final_loss = model_loss + current_meta_weight * adaptive_weight * meta_loss

                    pbar.set_postfix({
                        'Phase': 'Proactive-Meta+Nash',
                        'W_M': f"{weights[0].item():.2f}",
                        'W_A': f"{weights[1].item():.2f}",
                        'W_C': f"{weights[2].item():.2f}",
                        'ValLoss': f"{val_loss_after.item():.4f}"
                    })

                    self.optimizer_meta.zero_grad()
                    self.optimizer_model.zero_grad()

                    if self.grad_accumulation_steps > 1:
                        final_loss = final_loss / self.grad_accumulation_steps
                        final_loss.backward()
                        accumulation_counter += 1
                        if accumulation_counter % self.grad_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(self.imnet.parameters(), max_norm=1.0)
                            if self.is_amazon and self.meta_active:
                                clip_norm = 1.0
                            elif self.is_lightgcn_meta and self.is_yelp:
                                clip_norm = 1.5
                            else:
                                clip_norm = 3.0
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
                            self.optimizer_model.step()
                            self.optimizer_meta.step()
                            self.update_ema()
                            accumulation_counter = 0
                    else:
                        final_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.imnet.parameters(), max_norm=1.0)
                        if self.is_amazon and self.meta_active:
                            clip_norm = 1.0
                        elif self.is_lightgcn_meta and self.is_yelp:
                            clip_norm = 1.5
                        else:
                            clip_norm = 3.0
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
                        self.optimizer_model.step()
                        self.optimizer_meta.step()
                        self.update_ema()

                    total_loss += model_loss.item()

                else:
                    # Fast update branch
                    train_losses_detached = [l.detach() for l in train_losses_original]
                    if self.current_disable_lookahead:
                        raw_weights, weights = self.imnet(train_losses_original.detach())
                        if (self.is_yelp or self.is_amazon) and not self.current_disable_lookahead and self.meta_active:
                            weights = torch.sigmoid(weights) * 0.8 + 0.1
                        else:
                            weights = torch.sigmoid(weights) * 0.99 + 0.005
                        model_loss = torch.sum(weights * train_losses_original)
                        pbar.set_postfix({
                            'Phase': 'Reactive-Meta',
                            'W_M': f"{weights[0].item():.2f}",
                            'W_A1': f"{weights[1].item():.2f}",
                            'W_A2': f"{weights[2].item():.2f}",
                            'W_C': f"{weights[3].item():.2f}",
                        })
                    else:
                        weights = self.imnet(train_losses_original.detach())[1]
                        if (self.is_yelp or self.is_amazon) and not self.current_disable_lookahead and self.meta_active:
                            weights = torch.sigmoid(weights) * 0.8 + 0.1
                        else:
                            weights = torch.sigmoid(weights) * 0.99 + 0.005
                        model_loss = torch.sum(weights * train_losses_original)
                        pbar.set_postfix({
                            'Phase': 'Fast-Meta',
                            'W_M': f"{weights[0].item():.2f}",
                            'W_A': f"{weights[1].item():.2f}",
                            'W_C': f"{weights[2].item():.2f}"
                        })

                    self.optimizer_meta.zero_grad()
                    self.optimizer_model.zero_grad()

                    if self.grad_accumulation_steps > 1:
                        model_loss = model_loss / self.grad_accumulation_steps
                        model_loss.backward()
                        accumulation_counter += 1
                        if accumulation_counter % self.grad_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(self.imnet.parameters(), max_norm=1.0)
                            if self.current_disable_lookahead:
                                clip_norm = 3.0
                            else:
                                clip_norm = 1.0 if self.is_amazon and self.meta_active else 3.0
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
                            self.optimizer_model.step()
                            self.optimizer_meta.step()
                            self.update_ema()
                            accumulation_counter = 0
                    else:
                        model_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.imnet.parameters(), max_norm=1.0)
                        if self.current_disable_lookahead:
                            clip_norm = 3.0
                        else:
                            clip_norm = 1.0 if self.is_amazon and self.meta_active else 3.0
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
                        self.optimizer_model.step()
                        self.optimizer_meta.step()
                        self.update_ema()

                    total_loss += model_loss.item()

            # Other mode branches (add EMA and gradient accumulation)
            elif mode == 'scalarization':
                losses = self._compute_losses(batch)
                weights = torch.tensor(self.config.get('static_weights', [1.0, 0.1, 0.05])).to(self.device)
                final_loss = torch.sum(weights * losses)
                pbar.set_postfix({'Loss': f"{final_loss.item():.4f}"})

                if self.grad_accumulation_steps > 1:
                    final_loss = final_loss / self.grad_accumulation_steps
                    final_loss.backward()
                    accumulation_counter += 1
                    if accumulation_counter % self.grad_accumulation_steps == 0:
                        self.optimizer_model.step()
                        self.update_ema()
                        accumulation_counter = 0
                else:
                    final_loss.backward()
                    self.optimizer_model.step()
                    self.update_ema()

                total_loss += final_loss.item()

            elif mode == 'pcgrad':
                losses = self._compute_losses(batch)
                loss_val = self._step_pcgrad(losses, epoch)
                total_loss += loss_val

            elif mode == 'mgda':
                losses = self._compute_losses(batch)
                loss_val = self._step_mgda(losses, epoch)
                total_loss += loss_val

            elif mode == 'gradnorm':
                losses = self._compute_losses(batch)
                loss_val = self._step_gradnorm(losses)
                total_loss += loss_val

            elif mode == 'single':
                if self.config.get('model_name') == 'SimGCL':
                    losses = self._compute_losses(batch)
                    cl_lambda = self.config.get('cl_lambda', 0.1)
                    final_loss = losses[0] + cl_lambda * losses[2]
                else:
                    users = self._to_tensor(batch[0])
                    pos_items = self._to_tensor(batch[1])
                    neg_items = self._to_tensor(batch[2])

                    if self.model_name == 'NCF':
                        u_emb, i_emb = self.model.get_all_embeddings()
                    else:
                        adj = self.dp.norm_adj.to(self.device) if hasattr(self.dp, 'norm_adj') else None
                        u_emb, i_emb = self.model.get_all_embeddings(adj)
                    batch_u = u_emb[users]
                    batch_pos = i_emb[pos_items]
                    batch_neg = i_emb[neg_items]
                    pos_scores = torch.sum(batch_u * batch_pos, dim=1)
                    neg_scores = torch.sum(batch_u * batch_neg, dim=1)
                    bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                    reg_loss = (1 / 2) * (
                            batch_u.norm(2).pow(2) + batch_pos.norm(2).pow(2) + batch_neg.norm(2).pow(2)) / float(
                        users.shape[0])
                    final_loss = bpr_loss + self.config.get('wd', 1e-4) * reg_loss
                pbar.set_postfix({'Loss': f"{final_loss.item():.4f}"})

                if self.grad_accumulation_steps > 1:
                    final_loss = final_loss / self.grad_accumulation_steps
                    final_loss.backward()
                    accumulation_counter += 1
                    if accumulation_counter % self.grad_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                        self.optimizer_model.step()
                        self.update_ema()
                        accumulation_counter = 0
                else:
                    final_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer_model.step()
                    self.update_ema()

                total_loss += final_loss.item()

        # Process remaining gradient accumulation
        if accumulation_counter > 0:
            if self.is_amazon and self.meta_active:
                clip_norm = 0.5
            elif self.is_lightgcn_meta and self.is_yelp:
                clip_norm = 1.5
            else:
                clip_norm = 3.0
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
            self.optimizer_model.step()
            self.update_ema()
            accumulation_counter = 0

        return total_loss / len(self.dp.train_loader)

    def update_lr(self, current_ndcg, epoch):
        """
        Learning rate scheduling + dynamic meta-learning triggering logic (based on consecutive NDCG drops)
        """
        # ===== GradNorm mode: completely skip meta-learning related logic =====
        if self.mode == 'gradnorm':
            # Only perform basic learning rate scheduling (if needed) and best model recording
            if current_ndcg > self.best_metric:
                self.best_metric = current_ndcg
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            # If a scheduler is available, perform scheduling as needed (optional)
            if hasattr(self, 'scheduler_model') and self.scheduler_model is not None:
                if isinstance(self.scheduler_model, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler_model.step(current_ndcg)
                else:
                    self.scheduler_model.step()
            return

        # ===== Fixed Weights mode: do not use meta-learning, but retain basic learning rate scheduling =====
        if self.mode == 'fixed_weights':
            if current_ndcg > self.best_metric:
                self.best_metric = current_ndcg
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            if isinstance(self.scheduler_model, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler_model.step(current_ndcg)
            else:
                self.scheduler_model.step()
            return

        # ===== Random Dynamic mode: completely disable meta-learning =====
        if self.is_random_dynamic:
            if current_ndcg > self.best_metric:
                self.best_metric = current_ndcg
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            if isinstance(self.scheduler_model, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler_model.step(current_ndcg)
            else:
                self.scheduler_model.step()
            return

        # ===== Static mode =====
        if self.mode == 'single':
            self.scheduler_model.step(current_ndcg)
            if current_ndcg > self.best_metric:
                self.best_metric = current_ndcg
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            return

        # Update historical best metrics
        if current_ndcg > self.best_metric:
            self.best_metric = current_ndcg
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            self.consecutive_decreases = 0
            self.last_best_update_epoch = epoch

        # Learning rate scheduling
        if self.is_lightgcn_meta and self.is_yelp:
            if not hasattr(self, '_last_ndcg'):
                self._last_ndcg = current_ndcg
            else:
                if current_ndcg < self._last_ndcg * (1 - self.config.get('lr_drop_threshold', 0.02)):
                    self.scheduler_model.step(current_ndcg)
                    if self.meta_active:
                        self.scheduler_meta.step(current_ndcg)
                    print(f"[LightGCN+Meta-Yelp] Adjusting learning rates")
                self._last_ndcg = current_ndcg
        else:
            if isinstance(self.scheduler_model, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler_model.step(current_ndcg)
            else:
                self.scheduler_model.step()
            if self.meta_active:
                if isinstance(self.scheduler_meta, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler_meta.step(current_ndcg)
                else:
                    self.scheduler_meta.step()

        # Meta-learning triggering logic
        if not self.meta_active and self.meta_cool_down == 0:
            if current_ndcg < self.best_metric - self.meta_recovery_threshold:
                self.consecutive_decreases += 1
            else:
                self.consecutive_decreases = 0

            if self.consecutive_decreases >= self.meta_decrease_window:
                print(
                    f"\n[Meta Trigger] NDCG decreased {self.consecutive_decreases} times consecutively. Activating meta-learning...")
                self.meta_active = True
                self.meta_intervention_steps = 0
                self.pre_intervention_state = copy.deepcopy(self.model.state_dict())
                self.last_best_update_epoch = epoch
                lr_factor = 0.5 if self.is_amazon else 0.8
                for param_group in self.optimizer_model.param_groups:
                    param_group['lr'] *= lr_factor
                    print(f"[Meta Activation] Model LR reduced to {param_group['lr']:.6f}")

        # Cooling period decay
        if self.meta_cool_down > 0:
            self.meta_cool_down -= 1

        # Force activation by absolute epoch
        if not self.meta_active and self.meta_cool_down == 0 and self.force_meta_epoch is not None:
            if epoch >= self.force_meta_epoch:
                print(
                    f"\n[Meta Force] Epoch {epoch} reached force_meta_epoch={self.force_meta_epoch}. Activating meta-learning immediately.")
                self.meta_active = True
                self.meta_intervention_steps = 0
                self.pre_intervention_state = copy.deepcopy(self.model.state_dict())
                self.last_best_update_epoch = epoch
                lr_factor = 0.95 if self.is_amazon else 0.8
                for param_group in self.optimizer_model.param_groups:
                    param_group['lr'] *= lr_factor
                    print(f"[Meta Activation] Model LR reduced to {param_group['lr']:.6f}")
                self.force_meta_epoch = None

        # Guaranteed trigger
        if not self.meta_active and self.meta_cool_down == 0 and epoch >= self.force_meta_start_epoch:
            if epoch - self.last_best_update_epoch >= self.force_meta_patience:
                print(
                    f"\n[Meta Force] No best improvement for {epoch - self.last_best_update_epoch} epochs (>= {self.force_meta_patience}). Forcing meta-learning.")
                self.meta_active = True
                self.last_best_update_epoch = epoch
                self.meta_intervention_steps = 0
                self.pre_intervention_state = copy.deepcopy(self.model.state_dict())
                lr_factor = 0.95 if self.is_amazon else 0.8
                for param_group in self.optimizer_model.param_groups:
                    param_group['lr'] *= lr_factor
                    print(f"[Meta Activation] Model LR reduced to {param_group['lr']:.6f}")

    @torch.no_grad()
    def evaluate(self, top_k=20):
        self.model.eval()
        from main_gpu import get_metrics
        adj = self.dp.norm_adj.to(self.device) if hasattr(self.dp, 'norm_adj') else None
        recall, ndcg = get_metrics(
            self.model, self.config['model_name'], adj,
            self.dp.train_dict, self.dp.test_dict, self.device, top_k=top_k
        )
        return recall, ndcg

    def _get_random_weights(self):
        weights = torch.rand(3, device=self.device)
        return weights / (weights.sum() + 1e-8)

    def train(self):
        epochs = self.config.get('epochs', 200)
        top_k = self.config.get('top_k', 20)
        eval_step = self.config.get('eval_freq', 1)
        best_recall, best_ndcg, best_epoch = 0.0, 0.0, 0
        print(
            f"[Trainer Info] Start training {self.model_name} on {self.config.get('dataset')} in '{self.mode}' mode for {epochs} epochs.")

        if 'seed' in self.config:
            print(f"[Trainer Info] Using fixed seed: {self.config['seed']} (set by caller)")

        loss_window = []

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(epoch=epoch, mode=self.mode)

            loss_window.append(loss)
            if len(loss_window) > 5:
                loss_window.pop(0)
            avg_loss = np.mean(loss_window) if loss_window else loss

            if epoch % eval_step == 0 or epoch == 1:
                # Select evaluation method based on whether EMA is enabled
                if self.use_ema and epoch > self.warmup_epochs:
                    recall, ndcg = self.evaluate_with_ema(top_k=top_k)
                else:
                    recall, ndcg = self.evaluate(top_k=top_k)
                self.update_lr(ndcg, epoch)

                # Meta-learning intervention state machine
                if self.meta_active:
                    self.meta_intervention_steps += 1

                    explosion_detected = False
                    if self.avg_loss_before_meta is not None:
                        if self.is_amazon:
                            if loss < self.avg_loss_before_meta * 0.7:
                                self.avg_loss_before_meta = 0.98 * self.avg_loss_before_meta + 0.02 * loss
                            explosion_ratio = self.proactive_loss_explosion_ratio
                        elif self.is_yelp and not self.disable_lookahead and self.meta_active:
                            if loss < self.avg_loss_before_meta * 0.7:
                                self.avg_loss_before_meta = 0.98 * self.avg_loss_before_meta + 0.02 * loss
                            explosion_ratio = self.proactive_loss_explosion_ratio
                        else:
                            if loss < self.avg_loss_before_meta * 0.9:
                                self.avg_loss_before_meta = 0.9 * self.avg_loss_before_meta + 0.1 * loss
                            explosion_ratio = self.proactive_loss_explosion_ratio
                        if loss > self.avg_loss_before_meta * explosion_ratio:
                            print(
                                f"[Meta] Loss explosion detected (current={loss:.4f}, baseline={self.avg_loss_before_meta:.4f}, ratio={explosion_ratio:.1f}x). Immediate rollback.")
                            self.model.load_state_dict(self.pre_intervention_state)
                            self.meta_active = False
                            self.meta_cool_down = 15
                            print(f"[Meta] Cooling down for {self.meta_cool_down} epochs.")
                            self.avg_loss_before_meta = None
                            explosion_detected = True

                            if ndcg > best_ndcg:
                                best_recall, best_ndcg, best_epoch = recall, ndcg, epoch
                                print(
                                    f"[Best Updated] Epoch {epoch:03d} | Loss: {loss:.4f} | R@{top_k}: {best_recall:.4f} | N@{top_k}: {best_ndcg:.4f}  <-- New Best!")
                            else:
                                print(
                                    f"Epoch {epoch:03d} | Loss: {loss:.4f} | R@{top_k}: {recall:.4f} | N@{top_k}: {ndcg:.4f} | (Best N: {best_ndcg:.4f})")
                            continue

                    if explosion_detected:
                        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} (Exploded, rolled back)")
                        continue

                    if ndcg > self.best_metric:
                        self.best_metric = ndcg
                        self.best_model_state = copy.deepcopy(self.model.state_dict())
                        self.consecutive_decreases = 0
                        print(f"[Meta] Intervention effective: new best NDCG = {ndcg:.4f}")

                    if self.meta_intervention_steps >= self.meta_max_steps:
                        if ndcg <= self.best_metric:
                            print(
                                f"[Meta] No improvement after {self.meta_max_steps} steps. Rolling back to pre-intervention model.")
                            self.model.load_state_dict(self.pre_intervention_state)
                        else:
                            print(f"[Meta] Intervention succeeded! Keeping the improved model.")

                        self.meta_active = False
                        self.meta_cool_down = 8
                        self.avg_loss_before_meta = None
                        print(f"[Meta] Cooling down for {self.meta_cool_down} epochs.")

                # Normal best model update
                if ndcg > best_ndcg:
                    best_recall, best_ndcg, best_epoch = recall, ndcg, epoch
                    print(
                        f"[Best Updated] Epoch {epoch:03d} | Loss: {loss:.4f} | R@{top_k}: {best_recall:.4f} | N@{top_k}: {best_ndcg:.4f}  <-- New Best!")
                    if self.early_stop_patience > 0:
                        self.no_improve_epochs = 0
                else:
                    print(
                        f"Epoch {epoch:03d} | Loss: {loss:.4f} | R@{top_k}: {recall:.4f} | N@{top_k}: {ndcg:.4f} | (Best N: {best_ndcg:.4f})")
                    if self.early_stop_patience > 0:
                        self.no_improve_epochs += 1
                        if self.no_improve_epochs >= self.early_stop_patience:
                            print(
                                f"[Early Stop] No improvement for {self.early_stop_patience} epochs. Stopping training.")
                            break
            else:
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | [Eval Skipped]")

            if self.meta_active and self.meta_intervention_steps == 1 and self.avg_loss_before_meta is None:
                baseline_loss = min(loss_window) if loss_window else avg_loss
                self.avg_loss_before_meta = baseline_loss
                print(
                    f"[Meta] Recorded baseline loss for explosion detection: {self.avg_loss_before_meta:.4f} (min of recent {len(loss_window)} epochs)")

        return best_recall, best_ndcg

    def _step_pcgrad(self, losses, epoch):
        num_tasks = len(losses)
        task_grads = []
        for i in range(num_tasks):
            grad = torch.autograd.grad(losses[i], self.model.parameters(), retain_graph=(i < num_tasks - 1),
                                       allow_unused=True)
            grad = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad, self.model.parameters())]
            task_grads.append(torch.cat([g.flatten() for g in grad]))

        # PCGrad projection
        modified_grads = [g.clone() for g in task_grads]
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i == j: continue
                dot = torch.dot(modified_grads[i], task_grads[j])
                if dot < 0:
                    modified_grads[i] -= (dot / (torch.dot(task_grads[j], task_grads[j]) + 1e-8)) * task_grads[j]
        total_grad = sum(modified_grads)

        self.optimizer_model.zero_grad()
        offset = 0
        for param in self.model.parameters():
            numel = param.numel()
            param.grad = total_grad[offset:offset + numel].view(param.shape).clone()
            offset += numel
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip_norm', 1.0))
        self.optimizer_model.step()
        self.update_ema()
        return sum(l.item() for l in losses)

    def _step_mgda(self, losses, epoch):
        num_tasks = len(losses)
        task_grads = []
        for i in range(num_tasks):
            grad = torch.autograd.grad(losses[i], self.model.parameters(), retain_graph=(i < num_tasks - 1),
                                       allow_unused=True)
            grad = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad, self.model.parameters())]
            task_grads.append(torch.cat([g.flatten() for g in grad]))
        avg_grad = sum(task_grads) / num_tasks
        self.optimizer_model.zero_grad()
        offset = 0
        for param in self.model.parameters():
            numel = param.numel()
            param.grad = avg_grad[offset:offset + numel].view(param.shape).clone()
            offset += numel
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip_norm', 1.0))
        self.optimizer_model.step()
        self.update_ema()
        return sum(l.item() for l in losses)

    def _step_gradnorm(self, losses):
        if self.initial_losses is None:
            self.initial_losses = [l.item() for l in losses]
        else:
            for i in range(len(losses)):
                self.initial_losses[i] = 0.99 * self.initial_losses[i] + 0.01 * losses[i].item()

        num_tasks = len(losses)

        # ----- Ensure gradnorm_weights dims match task count -----
        if not hasattr(self, 'gradnorm_weights') or len(self.gradnorm_weights) != num_tasks:
            self.gradnorm_weights = nn.Parameter(torch.ones(num_tasks, device=self.device))
            self.gradnorm_optimizer = optim.Adam([self.gradnorm_weights], lr=0.01)

        # 1. Compute weighted loss for model update
        weighted_losses = [self.gradnorm_weights[i] * losses[i] for i in range(num_tasks)]
        total_loss = sum(weighted_losses)

        self.optimizer_model.zero_grad()
        total_loss.backward(retain_graph=True)  # Retain computation graph for GradNorm

        # ========== Select shared parameters (first trainable parameter) ==========
        shared_param = None
        for param in self.model.parameters():
            if param.requires_grad:
                shared_param = param
                break

        if shared_param is None:
            print("[ERROR] No trainable parameters found in model!")
            # Fallback to standard training
            self.optimizer_model.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip_norm', 1.0))
            self.optimizer_model.step()
            self.update_ema()
            return total_loss.item()

        grad_norms = []
        valid_grad_count = 0

        for i in range(num_tasks):
            # Compute gradients of each task w.r.t. shared parameters (with computation graph)
            grad = torch.autograd.grad(
                weighted_losses[i],
                shared_param,
                retain_graph=True,
                create_graph=True,  # Enable gradient history for grad tensor
                allow_unused=True
            )[0]

            if grad is not None:
                grad_norm = torch.norm(grad, 2)
                grad_norms.append(grad_norm)
                if grad_norm > 0:
                    valid_grad_count += 1
            else:
                grad_norms.append(torch.tensor(0.0, device=self.device))

        grad_norms = torch.stack(grad_norms)

        if valid_grad_count == 0:
            print("[WARNING] No valid gradients found, skipping GradNorm update")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip_norm', 1.0))
            self.optimizer_model.step()
            self.update_ema()
            return total_loss.item()

        # ========== Compute target norm for GradNorm==========
        with torch.no_grad():
            # Average gradient norm (as a constant, detached to avoid extra gradients)
            mean_norm = grad_norms.mean().detach()

            # Loss ratio (as tensor, detached to prevent gradients flowing into main model)
            loss_ratios = torch.tensor(
                [losses[i].item() / (self.initial_losses[i] + 1e-8) for i in range(num_tasks)],
                device=self.device
            ).detach()

            inverse_train_rate = loss_ratios / (loss_ratios.mean() + 1e-8)
            alpha = 1.0
            targets = mean_norm * (inverse_train_rate ** alpha)

        # Gradient loss (grad_norms has gradient history, targets are detached → grad_loss is differentiable)
        grad_loss = torch.sum(torch.abs(grad_norms - targets))

        # Update GradNorm weights (weight optimizer)
        self.gradnorm_optimizer.zero_grad()
        grad_loss.backward()
        self.gradnorm_optimizer.step()

        # Normalize GradNorm weights (keep weight scale stable)
        with torch.no_grad():
            self.gradnorm_weights.data = self.gradnorm_weights.data / (
                        self.gradnorm_weights.data.sum() + 1e-8) * num_tasks

        # Update main model parameters
        # Note: total_loss has been backpropagated, gradients are stored in model params, call step() directly
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip_norm', 1.0))
        self.optimizer_model.step()
        self.update_ema()

        return total_loss.item()

    def _to_tensor(self, x):
        if torch.is_tensor(x):
            return x.to(self.device).view(-1)
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return torch.empty(0, device=self.device)
            if torch.is_tensor(x[0]):
                if x[0].dim() == 0:
                    x = torch.stack(x)
                else:
                    x = torch.cat(x)
            else:
                x = torch.tensor(x)
            return x.to(self.device).view(-1)
        return torch.tensor([x]).to(self.device).view(-1)