import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64, n_layers=3):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        # Core: Initial Embeddings for Users and Items
        self.embedding_user = nn.Embedding(num_users, embed_dim)
        self.embedding_item = nn.Embedding(num_items, embed_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def get_all_embeddings(self, graph, perturbed=False, eps=0.1):
        """
         perturbed=True: Introduce data augmentation for contrastive learning:
         After each layer propagation, inject uniform noise normalized by L2 norm into the embedding.
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(graph, all_emb)

            # Graph structure-level perturbation(Data Augmentation for CL)
            if perturbed and self.training:
                random_noise = torch.rand_like(all_emb).to(all_emb.device) * 2 - 1
                all_emb = all_emb + torch.sign(all_emb) * F.normalize(random_noise, dim=-1) * eps

            embs.append(all_emb)

        # Adopt mean pooling to fuse multi-layer receptive fields
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def forward(self, users, pos_items, neg_items, graph):
        # Compatible with the legacy main task forward propagation
        all_users, all_items = self.get_all_embeddings(graph)
        return all_users[users], all_items[pos_items], all_items[neg_items]

class NCF(nn.Module):
    """ Standard Neural Collaborative Filtering (NCF) with configurable MLP layers """

    def __init__(self, num_users, num_items, embed_dim=64, n_layers=3, mlp_hidden_ratio=2):
        """
        Args:
            num_users: Number of users
            num_items: Number of items
            embed_dim: Embedding dimension
            n_layers: Number of hidden layers in MLP (excluding input and output layers)
            mlp_hidden_ratio: Reduction ratio of hidden units per layer relative to the input dimension (halved layer by layer)
        """
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        # Build dynamic MLP
        input_dim = embed_dim * 2
        layers = []
        hidden_dim = input_dim // mlp_hidden_ratio
        for i in range(n_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim * 2, hidden_dim))
            layers.append(nn.ReLU())
            # Halve layer by layer (optional)
            input_dim = hidden_dim
            hidden_dim = max(hidden_dim // 2, 16)
        # Final layer outputs 1-dimensional score
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_all_embeddings(self, graph=None):
        return self.user_embedding.weight, self.item_embedding.weight

    def forward(self, users, pos_items, neg_items=None, return_embs=False, graph=None):
        u_emb = self.user_embedding(users)
        pos_i_emb = self.item_embedding(pos_items)
        pos_concat = torch.cat([u_emb, pos_i_emb], dim=-1)
        pos_scores = self.mlp(pos_concat).squeeze(-1)

        neg_scores = None
        neg_i_emb = None
        if neg_items is not None:
            neg_i_emb = self.item_embedding(neg_items)
            neg_concat = torch.cat([u_emb, neg_i_emb], dim=-1)
            neg_scores = self.mlp(neg_concat).squeeze(-1)

        if return_embs:
            return pos_scores, neg_scores, u_emb, pos_i_emb, neg_i_emb
        else:
            return pos_scores, neg_scores


class SimGCL(nn.Module):
    """ Simple Graph Contrastive Learning (SimGCL). """

    def __init__(self, num_users, num_items, embed_dim=64, n_layers=3, eps=0.1):
        super(SimGCL, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = n_layers
        self.eps = eps
        self.embedding = nn.Embedding(num_users + num_items, embed_dim)

        nn.init.normal_(self.embedding.weight, std=0.1)

    def get_all_embeddings(self, graph, perturbed=False):
        all_emb = self.embedding.weight
        embs = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(graph, all_emb)

            if perturbed and self.training:
                noise = torch.rand_like(all_emb).to(all_emb.device)

                # Core: Equivariant Noise Injection
                all_emb = all_emb + torch.sign(all_emb) * F.normalize(noise, dim=-1) * self.eps

            embs.append(all_emb)

        # Final representation: average of all layers
        final_emb = torch.stack(embs, dim=1).mean(dim=1)
        user_emb, item_emb = torch.split(final_emb, [self.num_users, self.num_items])
        return user_emb, item_emb

    def forward(self, users, pos_items, neg_items, graph):
        # No noise is added during training of the main task (BPR Loss)
        u_g, i_g = self.get_all_embeddings(graph, perturbed=False)
        return u_g[users], i_g[pos_items], i_g[neg_items]

class HINE(nn.Module):
    """
    Heterogeneous Information Network Embedding (HINE).
    Simplified implementation: Introduce a layer-wise attention mechanism on top of LightGCN to simulate heterogeneous propagation.
    """
    def __init__(self, num_users, num_items, embed_dim=64, n_layers=3):
        super(HINE, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        # Layer-wise weights: Simulate the importance of different hop counts (i.e., heterogeneous path lengths)
        self.layer_weights = nn.Parameter(torch.ones(n_layers + 1))

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def get_all_embeddings(self, graph):
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(graph, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        # Core modification: Use learnable layer weights for weighted aggregation instead of simple averaging
        all_embeddings = torch.stack(all_embeddings, dim=1)
        weights = F.softmax(self.layer_weights, dim=0)
        final_embeddings = torch.sum(all_embeddings * weights.unsqueeze(0).unsqueeze(-1), dim=1)

        u_g, i_g = torch.split(final_embeddings, [self.num_users, self.num_items])
        return u_g, i_g

    def forward(self, users, pos_items, neg_items, graph):
        u_g, i_g = self.get_all_embeddings(graph)
        return u_g[users], i_g[pos_items], i_g[neg_items]
