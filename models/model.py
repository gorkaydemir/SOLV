import warnings

import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as f
from torch.nn import init
import random
import timm

from sklearn.cluster import AgglomerativeClustering

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, residual=False, layer_order="none"):
        super().__init__()
        self.residual = residual
        self.layer_order = layer_order
        if residual:
            assert input_dim == output_dim

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)

        if layer_order in ["pre", "post"]:
            self.norm = nn.LayerNorm(input_dim)
        else:
            assert layer_order == "none"

    def forward(self, x):
        input = x

        if self.layer_order == "pre":
            x = self.norm(x)

        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.dropout(x)

        if self.residual:
            x = x + input
        if self.layer_order == "post":
            x = self.norm(x)

        return x
    
class Visual_Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.resize_to = args.resize_to
        self.feat_res = [size // args.patch_size for size in args.resize_to]
        self.token_drop_ratio = args.token_drop_ratio
        self.token_num = args.token_num
        self.reduced_token_num = int(self.token_num * (1.0 - self.token_drop_ratio))

        self.encoder = args.encoder

        self.model = self.load_model(args)


    def load_model(self, args):
        assert args.resize_to[0] % args.patch_size == 0
        assert args.resize_to[1] % args.patch_size == 0
        
        if args.encoder == "dino-vitb-8":
            model = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
        elif args.encoder == "dino-vitb-16":
            model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
        elif args.encoder == "dinov2-vitb-14":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        elif args.encoder == "sup-vitb-16":
            model = timm.create_model("vit_base_patch16_224", pretrained=True, img_size=(args.resize_to[0], args.resize_to[1]))
        else:
            assert False

        for p in model.parameters():
            p.requires_grad = False

        # wget https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth
        # wget https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
        # wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
        
        return model
    
    @torch.no_grad()
    def forward(self, x, get_gt=False):
        # :arg x:  (B, F, 3, H, W)
        # 
        # :return x:  (B, token * token_drop_ratio, 768)
        # :return rand_indices:  (B, token * token_drop_ratio)

        self.model.eval()

        x = torch.flatten(x, start_dim=0, end_dim=1)
        if self.encoder.startswith("dinov2-"):
            x = self.model.prepare_tokens_with_masks(x)
        elif self.encoder.startswith("sup-"):
            x = self.model.patch_embed(x)
            x = self.model._pos_embed(x)
        else:
            x = self.model.prepare_tokens(x)

        B = x.shape[0]

        if self.token_drop_ratio != 0.0 and not get_gt:
            rand_indices = [torch.randperm(self.token_num, device=x.device)[:self.reduced_token_num] for _ in range(B)]
            rand_indices = torch.vstack(rand_indices).sort()[0] + 1
            # add cls token
            rand_indices = torch.cat([torch.zeros(B, 1, device=x.device, dtype=torch.long), rand_indices], dim=-1)  # (B, N' + 1)
            
            x = torch.gather(x, dim=1, index=rand_indices.unsqueeze(dim=-1).repeat(1, 1, 768))  # (B, N' + 1, 768)
            rand_indices = rand_indices[:, 1:] - 1                                              # (B, N')
        else:
            rand_indices = [torch.randperm(self.token_num, device=x.device) for _ in range(B)]
            rand_indices = torch.vstack(rand_indices).sort()[0]

        for blk in self.model.blocks:
            x = blk(x)
        x = x[:, 1:]

        assert x.shape[0] == B
        if not get_gt: 
            assert x.shape[1] == self.reduced_token_num
        else: 
            assert x.shape[1] == self.token_num
        assert x.shape[2] == 768

        return x, rand_indices

class Spatial_Binder(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()

        self.num_slots = args.num_slots
        self.scale = args.slot_dim ** -0.5
        self.iters = args.slot_att_iter
        self.slot_dim = args.slot_dim

        self.res_h = args.resize_to[0] // args.patch_size
        self.res_w = args.resize_to[1] // args.patch_size
        self.N = int(self.res_h * self.res_w)
        self.N_frame = args.N

        # === abs_grid ===
        self.sigma = 5
        xs = torch.linspace(-1, 1, steps=self.res_w)                                                # (C_x)
        ys = torch.linspace(-1, 1, steps=self.res_h)                                                # (C_y)

        xs, ys = torch.meshgrid(xs, ys, indexing='xy')                                              # (C_x, C_y), (C_x, C_y)
        xs = xs.reshape(1, 1, -1, 1)                                                                # (1, 1, C_x * C_y, 1)
        ys = ys.reshape(1, 1, -1, 1)                                                                # (1, 1, C_x * C_y, 1)
        self.abs_grid = nn.Parameter(torch.cat([xs, ys], dim=-1), requires_grad=True)               # (1, 1, N, 2)
        assert self.abs_grid.shape[2] == self.N

        self.h = nn.Linear(2, self.slot_dim)
        # === === ===

        # === Slot related ===
        self.slots = nn.Parameter(torch.Tensor(1, self.num_slots, self.slot_dim))

        self.S_s = nn.Parameter(torch.Tensor(1, self.num_slots, 1, 2))  # (1, S, 1, 2)
        self.S_p = nn.Parameter(torch.Tensor(1, self.num_slots, 1, 2))  # (1, S, 1, 2)

        init.xavier_uniform_(self.slots)
        init.normal_(self.S_s, mean=0., std=.02)
        init.normal_(self.S_p, mean=0., std=.02)
        # === === ===

        # === Slot Attention related ===
        self.Q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.norm = nn.LayerNorm(self.slot_dim)
        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.mlp = MLP(self.slot_dim, 4*self.slot_dim, self.slot_dim,
                       residual=True, layer_order="pre")
        # === === ===

        # === Query & Key & Value ===
        self.K = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.V = nn.Linear(self.slot_dim, self.slot_dim, bias=False)

        self.g = nn.Linear(2, self.slot_dim)
        self.f = nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.slot_dim, self.slot_dim))
        # === === ===

        # Note: starts and ends with LayerNorm
        self.initial_mlp = nn.Sequential(nn.LayerNorm(input_dim),
                                         nn.Linear(input_dim, input_dim),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(input_dim, self.slot_dim),
                                         nn.LayerNorm(self.slot_dim))

        self.final_layer = nn.Linear(self.slot_dim, self.slot_dim)

    def get_rel_grid(self, attn, token_indices):
        # :arg attn: (B, S, N')
        #
        # :return: (B * S, N, D_slot)

        B, S = attn.shape[:2]
        attn = attn.unsqueeze(dim=2)                                            # (B, S, 1, N')

        abs_grid_org = self.abs_grid.expand(B, S, self.N, 2)                    # (B, S, N, 2)
        abs_grid = torch.cat([abs_grid_org[i, :, token_indices[i]]\
            .unsqueeze(dim=0) for i in range(B)], dim=0)                        # (B, S, N', 2)
        
        S_p = torch.einsum('bsjd,bsij->bsd', abs_grid, attn)                    # (B, S, N', 2) x (B, S, 1, N') -> (B, S, 2)
        S_p = S_p.unsqueeze(dim=2)                                              # (B, S, 1, 2)

        values_ss = torch.pow(abs_grid - S_p, 2)                                # (B, S, N', 2)
        S_s = torch.einsum('bsjd,bsij->bsd', values_ss, attn)                   # (B, S, N', 2) x (B, S, 1, N') -> (B, S, 2)
        S_s = torch.sqrt(S_s)                                                   # (B, S, 2)
        S_s = S_s.unsqueeze(dim=2)                                              # (B, S, 1, 2)

        rel_grid = (abs_grid_org - S_p) / (S_s * self.sigma)                    # (B, S, N, 2)
        rel_grid = rel_grid.reshape(B * S, self.res_h, self.res_w, 2)           # (B * S, H, W, 2)
        rel_grid = rel_grid.reshape(B * S, -1, 2)                               # (B * S, N // 16, 2)

        rel_grid = self.h(rel_grid)                                             # (B * S, N // 16, D_slot)

        return rel_grid


    def forward(self, inputs, token_indices):
        # :arg inputs:              (B * F, N', D)
        # :arg token_indices:       (B * F, N')
        #
        # :return slots:            (B * F, S, D_slot)
        # :return attn:             (B, S, N')
        # :return all_attn:         (B, F, S, N')

        B, N, D = inputs.shape
        S = self.num_slots
        D_slot = self.slot_dim
        epsilon = 1e-8

        slots = self.slots.expand(B, S, D_slot)                     # (B * F, S, D_slot)
        inputs = self.initial_mlp(inputs).unsqueeze(dim=1)          # (B * F, 1, N', D_slot)
        inputs = inputs.expand(B, S, N, D_slot)                     # (B * F, S, N', D_slot)

        abs_grid_org = self.abs_grid.expand(B, S, self.N, 2)        # (B * F, S, N, 2)
        abs_grid = torch.cat([abs_grid_org[i, :, token_indices[i]]\
            .unsqueeze(dim=0) for i in range(B)], dim=0)            # (B * F, S, N', 2)

        assert torch.sum(torch.isnan(abs_grid)) == 0, f"token_indices[0, -5:]: {token_indices[0, -5:]}"

        S_s = self.S_s.expand(B, S, 1, 2)                           # (B * F, S, 1, 2)
        S_p = self.S_p.expand(B, S, 1, 2)                           # (B * F, S, 1, 2)

        for t in range(self.iters + 1):

            assert torch.sum(torch.isnan(slots)) == 0, f"Iteration {t}"
            assert torch.sum(torch.isnan(S_s)) == 0, f"Iteration {t}"
            assert torch.sum(torch.isnan(S_p)) == 0, f"Iteration {t}"
            
            slots_prev = slots
            slots = self.norm(slots)

            # === key and value calculation using rel_grid ===
            rel_grid = (abs_grid - S_p) / (S_s * self.sigma)        # (B * F, S, N', 2)
            k = self.f(self.K(inputs) + self.g(rel_grid))           # (B * F, S, N', D_slot)
            v = self.f(self.V(inputs) + self.g(rel_grid))           # (B * F, S, N', D_slot)

            # === Calculate attention ===
            q = self.Q(slots).unsqueeze(dim=-1)                     # (B * F, S, D_slot, 1)

            dots = torch.einsum('bsdi,bsjd->bsj', q, k)             # (B * F, S, D_slot, 1) x (B * F, S, N', D_slot) -> (B, S, N')
            dots *=  self.scale                                     # (B * F, S, N')
            attn = dots.softmax(dim=1) + epsilon                    # (B * F, S, N')


            # === Weighted mean ===
            attn = attn / attn.sum(dim=-1, keepdim=True)            # (B * F, S, N')
            attn = attn.unsqueeze(dim=2)                            # (B * F, S, 1, N')
            updates = torch.einsum('bsjd,bsij->bsd', v, attn)       # (B * F, S, N', D_slot) x (B * F, S, 1, N') -> (B * F, S, D_slot)

            # === Update S_p and S_s ===
            S_p = torch.einsum('bsjd,bsij->bsd', abs_grid, attn)    # (B * F, S, N', 2) x (B * F, S, 1, N') -> (B * F, S, 2)
            S_p = S_p.unsqueeze(dim=2)                              # (B * F, S, 1, 2)

            values_ss = torch.pow(abs_grid - S_p, 2)                # (B * F, S, N', 2)
            S_s = torch.einsum('bsjd,bsij->bsd', values_ss, attn)   # (B * F, S, N', 2) x (B * F, S, 1, N') -> (B * F, S, 2)
            S_s = torch.sqrt(S_s)                                   # (B * F, S, 2)
            S_s = S_s.unsqueeze(dim=2)                              # (B * F, S, 1, 2)

            # === Update ===
            if t != self.iters:
                slots = self.gru(
                    updates.reshape(-1, self.slot_dim),
                    slots_prev.reshape(-1, self.slot_dim))

                slots = slots.reshape(B, -1, self.slot_dim)
                slots = self.mlp(slots)

        slots = self.final_layer(slots)                                          # (B * F, S, D_slot)

        N_sub = rel_grid.shape[2]
        F = 2 * self.N_frame + 1
        attn = attn.reshape(-1, F, S, N_sub)                                      # (B, F, S, N')
        attn = attn[:, self.N_frame]                                              # (B, S, N')

        return slots, attn


class Temporal_Binder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.slot_dim = args.slot_dim
        self.num_slots = args.num_slots
        self.N = args.N
        self.F = 2 * args.N + 1

        encoder_layer = nn.TransformerEncoderLayer(self.slot_dim, nhead=8, dim_feedforward=4*self.slot_dim, batch_first=True)
        self.slot_transformer = nn.TransformerEncoder(encoder_layer, 3)

        self.pos_embed_temporal = nn.Parameter(torch.Tensor(1, self.F, 1, self.slot_dim))
        init.normal_(self.pos_embed_temporal, mean=0., std=.02)

    def forward(self, slots, mask):
        # :arg slots: (B * F, S, D_slot)
        # :arg mask: (B, F)
        #
        # :return: (B * F, S, D_slot)

        _, S, D_slot = slots.shape

        slots = slots.view(-1, self.F, S, D_slot)                       # (B, F, S, D_slot)
        slots = slots + self.pos_embed_temporal.expand(slots.shape)

        B = slots.shape[0]

        slots = slots.permute(0, 2, 1, 3)                                   # (B, S, F, D_slot)
        slots = torch.flatten(slots, start_dim=0, end_dim=1)                # (B * S, F, D_slot)

        mask = torch.logical_not(mask.to(torch.bool))                       # (B, F)
        mask = mask.repeat_interleave(S, dim=0)                             # (B * S, F)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            slots = self.slot_transformer(slots, src_key_padding_mask=mask) # (B * S, F, D_slot)

        # unblocked_slot_num = (torch.mean(mask.float(), dim=0) != 1).sum().long()
        unblocked_slot_num = self.F
        
        slots = slots.view(B, S, unblocked_slot_num, D_slot)                # (B, S, F, D_slot)
        slots = slots.permute(0, 2, 1, 3)                                   # (B, F, S, D_slot)

        slot_t = slots[:, self.N]                                           # (B, S, D_slot)

        return slot_t, slots

class Slot_Merger(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.num_slots = args.num_slots
        self.slot_dim = args.slot_dim
        self.cluster_drop_p = 0.1
        self.slot_merge_coeff = args.slot_merge_coeff
        self.epsilon = 1e-4

    def forward(self, slots, patch_attn):
        # :arg slots: (B, S, D_slot)
        # :arg patch_attn: (B, S, N')
        #
        # :return new_slots: (B, S, D_slot)
        # :return new_patch_attn: (B, S, N')
        # :return slot_nums: (B)

        assert torch.sum(torch.isnan(slots)) == 0
        if patch_attn is not None:
            assert torch.sum(torch.isnan(patch_attn)) == 0

        B, S = slots.shape[:2]
        slots_np = slots.detach().cpu().numpy()     # (B, S, D_slot)

        clusters = np.zeros((B, S))
        
        for i in range(B):
            # do not cluster
            if random.random() < self.cluster_drop_p:
                clusters[i] = np.arange(self.num_slots)
            else:
                AC = AgglomerativeClustering(n_clusters=None, metric="cosine", compute_full_tree=True, distance_threshold=self.slot_merge_coeff, linkage="complete")
                AC.fit(slots_np[i])
                clusters[i] = AC.labels_
                if clusters[i].max() <= 1:
                    clusters[i] = np.arange(self.num_slots)

        clusters = torch.from_numpy(clusters).long().to(slots.device)               # (B, S)

        I = torch.eye(self.num_slots).to(slots.device)                                               # (S, S)
        clusters = I[clusters].transpose(1, 2).to(slots.device).float()             # (B, S, S); such as [[0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
        slot_assignment_nums = torch.sum(clusters, dim=-1, keepdim=True)            # (B, S, 1); how many slot assigned to each, such as [[2], [1], [1], [0]]
        slot_nums = (slot_assignment_nums[:, :, 0] > 0).sum(dim=-1)                 # (B); number of reduced slots for each item, such as [2, 3, 1, 2]

        new_slots = torch.einsum('bij,bjd->bid', clusters, slots)                   # (B, S, D)
        new_slots = new_slots / (slot_assignment_nums + 1e-8)                       # (B, S, D)

        new_patch_attn = None
        if patch_attn is not None:
            new_patch_attn = torch.einsum('bij,bjd->bid', clusters, patch_attn) + 1e-8       # (B, S, N')
            new_patch_attn = new_patch_attn / (new_patch_attn.sum(dim=-1, keepdim=True))     # (B, S, N')

        return new_slots, new_patch_attn, slot_nums

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # === Token calculations ===
        slot_dim = args.slot_dim
        hidden_dim = 1024

        # === MLP Based Decoder ===
        self.layer1 = nn.Linear(slot_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.layer5 = nn.Linear(hidden_dim, 768 + 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, slot_maps):
        # :arg slot_maps: (B * S, token // 16, D_slot)

        slot_maps = self.relu(self.layer1(slot_maps))    # (B * S, token, D_hidden)
        slot_maps = self.relu(self.layer2(slot_maps))    # (B * S, token, D_hidden)
        slot_maps = self.relu(self.layer3(slot_maps))    # (B * S, token, D_hidden)
        slot_maps = self.relu(self.layer4(slot_maps))    # (B * S, token, D_hidden)

        slot_maps = self.layer5(slot_maps)               # (B * S, token, 768 + 1)

        return slot_maps

class SOLV(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.slot_dim = args.slot_dim
        self.slot_num = args.num_slots
        self.token_num = args.token_num
        self.merge_slots = args.merge_slots

        self.s_bind = Spatial_Binder(args, input_dim=768)
        self.t_bind = Temporal_Binder(args)
        self.merge = Slot_Merger(args)

        self.dec = Decoder(args)
        self.pos_dec = nn.Parameter(torch.Tensor(1, self.token_num, self.slot_dim))
        init.normal_(self.pos_dec, mean=0., std=.02)

    def sbd_slots(self, slots):
        # slots.shape =  (B * S, D_slot)

        slots = slots.view(-1, 1, self.slot_dim)
        # slots = slots.unsqueeze(1)                      # (B * S, 1, D_slot)
        slots = slots.tile(1, self.token_num, 1)

        pos_embed = self.pos_dec.expand(slots.shape)
        slots = slots + pos_embed                                   # (B * S, token // 16, D_slot)

        return slots
    
    
    def reconstruct_feature_map(self, rec, slot_nums):
        # :arg rec: (B * S, token // 16, D_slot)
        # :arg slot_nums: (B)

        B = rec.shape[0] // self.slot_num
        token = rec.shape[1]

        unstacked = rec.reshape([B, -1] + list(rec.shape[1:]))      # (B, S, token, 768 + 1)
        channels, masks = torch.split(unstacked, [768, 1], dim=-1)  # (B, S, token, 768), (B, S, token, 1)

        reconstruction_return = torch.zeros(B, token, 768, device=rec.device)
        mask_return = torch.zeros(B, self.slot_num, token, device=rec.device)

        for i in range(B):
            slot_num = slot_nums[i]
            channels_i = channels[i, :slot_num]                         # (S', token, 768 + 1)
            masks_i = masks[i, :slot_num]                               # (S', token, 1)
            masks_i = f.softmax(masks_i, dim=0)                         # (S', token, 1)

            reconstruction = torch.sum(channels_i * masks_i, dim=0)     # (token, 768)
            reconstruction_return[i] = reconstruction

            mask_return[i, :slot_num] = masks_i.squeeze(dim=-1)

        assert torch.sum(torch.isnan(reconstruction_return)) == 0
        assert torch.sum(torch.isnan(mask_return)) == 0

        reconstruction_dict = {"rec": reconstruction_return,            # (B, token, 768)
                                "mask": mask_return}                    # (B, S, token)
        return reconstruction_dict
    
    def forward(self, frames, mask, token_indices):

        # :arg frames: (B * F, token, D)
        # :arg mask:   (B, F)

        B_F, token, _ = frames.shape
        B, F = mask.shape
        assert B * F == B_F

        # === === Get Slots === ===
        slots, attn = self.s_bind(frames, token_indices)              # (B * F, S, D_slot), (B, S, token')
        assert torch.sum(torch.isnan(slots)) == 0
        assert torch.sum(torch.isnan(attn)) == 0

        slots, all_slots = self.t_bind(slots, mask)                   # (B, S, D_slot), (B, F, S, D_slot)
        assert torch.sum(torch.isnan(slots)) == 0
        assert torch.sum(torch.isnan(all_slots)) == 0
        # === === === === ===

        # === === Slot merging === ===
        slot_nums = torch.ones(B, device=frames.device, dtype=torch.long) * self.slot_num
        if self.merge_slots:
            slots, attn, slot_nums = self.merge(slots, attn)                             # (B, S, D_slot), (B, S, token'), (B)
    
        rel_grid = self.s_bind.get_rel_grid(attn, token_indices)
            

        slot_maps = self.sbd_slots(slots) + rel_grid

        assert torch.sum(torch.isnan(slots)) == 0
        assert torch.sum(torch.isnan(rel_grid)) == 0
        assert torch.sum(torch.isnan(slot_maps)) == 0

        out = self.dec(slot_maps)

        assert torch.sum(torch.isnan(out)) == 0

        reconstruction = self.reconstruct_feature_map(out, slot_nums)

        reconstruction["slots"] = slots
        reconstruction["all_slots"] = all_slots
        reconstruction["slot_nums"] = slot_nums

        return reconstruction
    
