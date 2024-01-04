import os
import sys
import math
import random

import numpy as np

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import shutil
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import linear_sum_assignment

import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score

from datasets import ytvis19



# === ================ ===
# === Training Related ===

def restart_from_checkpoint(args, run_variables, **kwargs):

    checkpoint_path = args.checkpoint_path

    assert checkpoint_path is not None
    assert os.path.exists(checkpoint_path)

    # open checkpoint file
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint with msg {}".format(key, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint".format(key))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint".format(key))
        else:
            print("=> key '{}' not found in checkpoint".format(key))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def get_scheduler(args, optimizer, train_loader):
    T_max = len(train_loader) * args.num_epochs
    warmup_steps = int(T_max * 0.05)
    steps = T_max - warmup_steps
    gamma = math.exp(math.log(0.5) / (steps // 3))

    linear_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, total_iters=warmup_steps)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linear_scheduler, scheduler], milestones=[warmup_steps])
    return scheduler

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def get_dataloaders(args):
    if args.dataset == "ytvis19":
        train_dataset = ytvis19.YTVIS_train(args)
        val_dataset = ytvis19.YTVIS_val(args)

    elif args.dataset == "davis17":
        pass
    else:
        print("Not available dataset")


    train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=args.gpus, rank=args.gpu, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size // args.gpus,
        num_workers=5,      # cpu per gpu
        drop_last=True,
        pin_memory=True,
    )

    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
        batch_size=1,
        shuffle=False, 
        num_workers=5, 
        drop_last=False, 
        pin_memory=True)
     
    return train_dataloader, val_dataloader


# === Evaluation Related ===

def pairwise_cos_sim(a, b):
    n1 = a.shape[0]
    n2 = b.shape[0]

    a_norm = a / (a.norm(dim=1) + 1e-8)[:, None]
    b_norm = b / (b.norm(dim=1)+ 1e-8)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0,1))
    
    assert res.shape == (n1, n2)
    return res

def bipartiate_match_video(slots, slot_nums, masks):
    # :arg slots: (sum(slot_nums), D_slot)
    # :arg slot_nums: (#frames)
    # :arg masks: (#frames, H_target, W_target)
    
    F, H, W = masks.shape
    
    slot_acc = 0
    prev_assignment = torch.arange(slot_nums.max(), device=slot_nums.device).long()
    
    for t in range(1, F):
        slot_num_t = slot_nums[t - 1]
        slot_num_t_1 = slot_nums[t]
        
        slots_t = slots[slot_acc: slot_acc + slot_num_t]
        slot_acc = slot_acc + slot_num_t
        
        slots_t_1 = slots[slot_acc: slot_acc + slot_num_t_1]
        
        similarity_matrix = pairwise_cos_sim(slots_t, slots_t_1)
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix.cpu().numpy())
        ab = list(zip(row_ind, col_ind))
        ab_sorted = sorted(ab, key=lambda x: x[0])

        prev_indices = [x[0] for x in ab_sorted]
        indices = [x[1] for x in ab_sorted]
        
        
        if len(indices) < slot_num_t_1:
            assignments = torch.argmax(similarity_matrix, dim=0).tolist()
            
            for j in range(slot_num_t_1):
                if j not in col_ind:
                    prev_indices.append(assignments[j])
                    indices.append(j)

        """
        print(f"Frame {t}")
        print(f"prev_indices: {prev_indices}")
        print(f"indices: {indices}")
        print(f"prev_assignment: {prev_assignment}\n")
        """
        # assert len(indices) == slot_num_t_1
        
        prev_assignment_new = prev_assignment.clone()
        new_mask = torch.zeros(masks[t].shape, device=masks.device, dtype=torch.long)
        
        for i in range(len(indices)):
            t_id = prev_assignment[prev_indices[i]]
            new_mask[masks[t] == indices[i]] = t_id
            
            prev_assignment_new[indices[i]] = t_id
            
        prev_assignment = prev_assignment_new
        masks[t] = new_mask
        
    return masks


# === Evaluation Related ===

def get_iou(pred, gt, pred_class, gt_class):
    """
    Calculates IoU per frame for objects with given ID
    If union is 0, frame IoU is set to -1
    
    :arg pred: (#frames, H, W)
    :arg gt: (#frames, H, W)
    :arg pred_class: int
    :arg gt_class: int
    
    :return IoUs: (#frames)
    :return mIoU: (#frames)
    """
    
    # target.shape = (#frames, H, W)
    # source.shape = (#frames, H, W)
    pred = (pred == pred_class)
    gt = (gt == gt_class)
    
    # intersection.shape = (#frames)
    # union.shape = (#frames)
    intersection = torch.logical_and(pred, gt).sum(dim=-1).sum(dim=-1)
    union = torch.logical_or(pred, gt).sum(dim=-1).sum(dim=-1)
    
    available_frames = torch.where(union != 0)[0]
    assert len(available_frames) > 0, "GT mask has non-assigned label"
    
    
    # IoU will be -1 for NA frames
    intersection[union == 0] = -1
    union[union == 0] = 1
    
    IoUs = intersection / union
    mIoU = torch.mean(IoUs[IoUs != -1])
    
    return IoUs, mIoU

def swap_indices(pred, row_ind, col_ind, bg_class):
    """
    Changes row_ind of pred to col_ind
    Sets remaining to bg_class
    
    :arg pred: (#frames, H, W)
    :arg row_ind: (#ids)
    :arg col_ind: (#ids)
    :arg bg_class: int
    
    :return: (#frames, H, W)
    """
    
    new_pred = torch.ones(pred.shape, device=pred.device, dtype=torch.long) * bg_class
    
    matching_num = len(row_ind)
    for i in range(matching_num):
        pred_ID = row_ind[i]
        gt_ID = col_ind[i]
        
        new_pred[pred == pred_ID] = gt_ID
        
    return new_pred


# matches object masks (not bg)
def hungarian_matching(pred, gt, bg_class=0):
    """
    Matches region IDs, leaving bg. 
    Unassigned regions are all set to bg
    Returns prediction mask with matched region IDs
    
    :arg pred: (#frames, H, W)
    :arg gt: (#frames, H, W)
    :arg bg_class: int

    :return: (#frames)
    """
    
    # get all region IDs
    pred_classes = torch.unique(pred)
    gt_classes = torch.unique(gt)
    
    # delete BG from gt_classes:
    gt_classes = gt_classes[gt_classes != bg_class]
    
    gt_class_num = len(gt_classes)
    pred_class_num = len(pred_classes)
    if pred_class_num < gt_class_num:
        pred_class_num = gt_class_num
    
    # discard bg
    miou_res = torch.zeros(pred_class_num, gt_class_num, device=pred.device)
    for i, gt_ID in enumerate(gt_classes):
        for j, pred_ID in enumerate(pred_classes):
            miou_res[j, i] = get_iou(pred, gt, pred_ID, gt_ID)[1]

    all_metrics = miou_res.cpu().numpy()
    
    # row_ind -> col_ind
    # pred ID -> gt ID (without BG)
    row_ind, col_ind = linear_sum_assignment(-all_metrics)
    # col_ind[gt_classes.cpu() > bg_class] += 1

    # TODO: parametrize here:
    col_ind += 1
    
    new_pred = swap_indices(pred, row_ind, col_ind, bg_class)
    
    return new_pred


def get_ari(prediction_masks, gt_masks, bg_class):
    """
    :args prediction_masks: predicted slot mask, (#frames, H, W)
    :args gt_masks: gt instance mask, (#frames, H, W)
    """

    # prediction_masks.shape = (#frames, H * W)
    # gt_masks.shape = (#frames, H * W)
    prediction_masks = torch.flatten(prediction_masks, start_dim=1, end_dim=-1).cpu().numpy().astype(int)
    gt_masks = torch.flatten(gt_masks, start_dim=1, end_dim=-1).cpu().numpy().astype(int)

    assert prediction_masks.shape == gt_masks.shape, f"prediction_masks.shape: {prediction_masks.shape} gt_masks.shape: {gt_masks.shape}"

    frame_num = gt_masks.shape[0]

    # fg_indices.shape = (#frames, H * W)
    fg_indices = np.not_equal(bg_class, gt_masks)

    rand_scores = []
    for frame_idx in range(frame_num):
        fg_indices_frame = fg_indices[frame_idx]

        if fg_indices_frame.sum() == 0:
            continue

        pred = prediction_masks[frame_idx][fg_indices_frame]
        gt = gt_masks[frame_idx][fg_indices_frame]

        rand_scores.append(adjusted_rand_score(gt, pred))
    
    if len(rand_scores) == 0:
        ari = None
    else:
        ari = sum(rand_scores) / len(rand_scores)
    return ari



class Evaluator:
    def __init__(self, bg_class=0):
        self.reset()
        self.bg_class = bg_class

    def reset(self):
        self.object_numbers = []
        self.mious = []
        self.fg_aris = []

    def calculate_miou(self, pred, gt):
        # get all region IDs
        pred_classes = torch.unique(pred).sort()[0]
        gt_classes = torch.unique(gt).sort()[0]

        assert len(pred_classes) <= len(gt_classes) + 1, f"pred_classes: {pred_classes}, gt_classes: {gt_classes}"

        mious = []
        class_num = len(gt_classes)

        for cls in range(class_num):
            if gt_classes[cls] == self.bg_class:
                continue
            
            _, miou = get_iou(pred, gt, gt_classes[cls], gt_classes[cls])
            mious.append(miou)

        self.mious.extend(mious)


    def calculate_fg_ari(self, pred, gt):
        # get all region IDs
        pred_classes = torch.unique(pred)
        gt_classes = torch.unique(gt)

        assert len(pred_classes) <= len(gt_classes) + 1, f"pred_classes: {pred_classes}, gt_classes: {gt_classes}"

        aris = get_ari(pred, gt, self.bg_class)
        if aris is not None:
            self.fg_aris.append(aris)

    def update(self, prediction_masks, gt_masks):
        """
        :args prediction_masks: predicted slot mask, (#frames, H, W)
        :args gt_masks: gt instance mask, (#frames, H, W)
        """

        object_num = int(torch.max(gt_masks))
        self.object_numbers.append(object_num)

        assert gt_masks.shape == prediction_masks.shape, f"gt_masks.shape: {gt_masks.shape}, prediction_masks.shape: {prediction_masks.shape}"
        
        prediction_masks = hungarian_matching(prediction_masks, gt_masks, self.bg_class)

        self.calculate_miou(prediction_masks, gt_masks)
        self.calculate_fg_ari(prediction_masks, gt_masks)

        miou, _ = self.get_results(reset=False)

        return miou

    def get_results(self, reset=True):

        miou = sum(self.mious) / len(self.mious)
        fg_ari = sum(self.fg_aris) / len(self.fg_aris)

        if reset:
            self.reset()
        
        return miou, fg_ari
    


def get_writer(args):
    writer_path = os.path.join(args.model_save_path, "writer.log")
    if os.path.exists(writer_path):
        shutil.rmtree(writer_path)

    comment = f"lr: {args.learning_rate:.5f} bs: {args.batch_size}"
    writer = SummaryWriter(log_dir=writer_path, comment=comment)

    return writer

# === ===================== ===
# ===  Distributed Settings ===

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    # From https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L467C1-L499C42

    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])

    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = args.gpus
        args.gpu = args.rank % torch.cuda.device_count()

    # launched naively with `python train.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, 'env://'), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    # From https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L452C1-L464C30
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def fix_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)