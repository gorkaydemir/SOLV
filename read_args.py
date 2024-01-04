import sys
import argparse

import torch

# hpc: {73y59kPu.M,F,Tj (OLD)
# hpc: qKMj\2HtK"XnE:[4 (OLD)

# hpc: T3WqohKgGkDr18w
# git: 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

def set_remaining_args(args):
    args.gpus = torch.cuda.device_count()

    args.patch_size = int(args.encoder.split("-")[-1])
    args.token_num = (args.resize_to[0] * args.resize_to[1]) // (args.patch_size ** 2)


def print_args(args):
    if args.validate:
        print("====== Validation ======")
        print(f"use_checkpoint: {args.use_checkpoint}")
        print(f"checkpoint_path: {args.checkpoint_path}\n")

    else:
        print("====== Training ======")
        print(f"training name: {args.model_save_path.split('/')[-1]}\n")

        print(f"dataset: {args.dataset}")
        print(f"train_splits: {args.train_splits}")
        print(f"resize_to: {args.resize_to}\n")

        print(f"encoder: {args.encoder}\n")

        print(f"num_slots: {args.num_slots}")
        print(f"slot_att_iter: {args.slot_att_iter}")
        print(f"slot_dim: {args.slot_dim}")
        print(f"merge_slots: {args.merge_slots}")
        print(f"slot_merge_coeff: {args.slot_merge_coeff}\n")

        print(f"N: {args.N}\n")

        print(f"finetuning: {args.finetuning}")
        print(f"token_drop_ratio: {args.token_drop_ratio}")
        print(f"learning_rate: {args.learning_rate}")
        print(f"batch_size: {args.batch_size}")
        print(f"num_epochs: {args.num_epochs}")
    print("====== ======= ======\n")

def get_args():
    parser = argparse.ArgumentParser("SOLV")

    # === Data Related Parameters ===
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="ytvis19", choices=["davis17", "ytvis19"])
    parser.add_argument('--train_splits', nargs='+', type=str, default=["train", "valid", "test"])
    parser.add_argument('--resize_to',  nargs='+', type=int, default=[336, 504])

    # === ViT Related Parameters ===
    parser.add_argument('--encoder', type=str, default="dinov2-vitb-14", 
                        choices=["dinov2-vitb-14", "dino-vitb-16", "dino-vitb-8", "sup-vitb-16"])

    # === Slot Attention Related Parameters ===
    parser.add_argument('--num_slots', type=int, default=8)
    parser.add_argument('--slot_att_iter', type=int, default=3)
    parser.add_argument('--slot_dim', type=int, default=128)
    parser.add_argument('--merge_slots', type=bool, default=True)
    parser.add_argument('--slot_merge_coeff', type=float, default=0.12)

    # === Model Related Parameters ===
    parser.add_argument('--N', type=int, default=2)

    # === Training Related Parameters ===
    parser.add_argument('--finetuning', action="store_true")
    parser.add_argument('--token_drop_ratio', type=float, default=0.5)
    
    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_epochs', type=int, default=180)

    # === Misc ===
    parser.add_argument('--validate', action="store_true")
    parser.add_argument('--use_checkpoint', action="store_true")
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--validation_epoch', type=int, default=1)

    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--model_save_path', type=str, required=True)

    args = parser.parse_args()

    set_remaining_args(args)

    return args