import os
# import random
import numpy as np
import torch
import json
import random
from torch.utils import data
from glob import glob
from PIL import Image


import torchvision.transforms as T
import torch.nn.functional as F

# === Transformation Utils ===

def remove_borders(image, borders=None):
    image = np.array(image)                  # (H, W, C)    
    
    rows = np.mean(image, axis=(1, 2))       # (H)
    columns = np.mean(image, axis=(0, 2))    # (W)
    
    if borders is None:
        borders = {}
        top = 0
        for i in rows:
            if i < 1: top += 1
            else: break

        bottom = 0
        for i in rows[::-1]:
            if i < 1: bottom += 1
            else: break

        left = 0
        for i in columns:
            if i < 1: left += 1
            else: break

        right = 0
        for i in columns[::-1]:
            if i < 1: right += 1
            else: break

        borders["top"] = top
        borders["bottom"] = bottom
        borders["left"] = left
        borders["right"] = right

    else:
        bottom = borders["bottom"]
        top = borders["top"]
        right = borders["right"]
        left = borders["left"]

    if bottom == 0: image = image[top:]
    else: image = image[top:-bottom]
    
    if right == 0: image = image[:, left:]
    else: image = image[:, left:-right]
        
    image = Image.fromarray(np.uint8(image))

    return image, borders

class To_One_Hot(object):
    def __init__(self, max_obj_n, shuffle):
        self.max_obj_n = max_obj_n
        self.shuffle = shuffle

    def __call__(self, mask, obj_list=None):
        new_mask = np.zeros((self.max_obj_n, *mask.shape), np.uint8)

        if not obj_list:
            obj_list = list()
            obj_max = mask.max() + 1
            for i in range(1, obj_max):
                tmp = (mask == i).astype(np.uint8)
                if tmp.max() > 0:
                    obj_list.append(i)

            if self.shuffle:
                random.shuffle(obj_list)
            obj_list = obj_list[:self.max_obj_n - 1]

        for i in range(len(obj_list)):
            new_mask[i + 1] = (mask == obj_list[i]).astype(np.uint8)
        new_mask[0] = 1 - np.sum(new_mask, axis=0)

        return torch.from_numpy(new_mask), obj_list

    def __repr__(self):
        return self.__class__.__name__ + '(max_obj_n={})'.format(self.max_obj_n)


# === Dataset Classes ===
class YTVIS_train(data.Dataset):
    def __init__(self, args):
        self.root = args.root
        self.train_splits = args.train_splits


        self.N = args.N
        self.relative_orders = list(range(-self.N, self.N + 1))
        
        self.resize_to = args.resize_to

        self.patch_size = args.patch_size
        self.token_num = args.token_num
        
        # === Get Video Names and Lengths ===
        self.dataset_list = []
        self.video_lengths = []
        self.split_name = []

        # === Train Set ===
        if "train" in self.train_splits:
            train_imset_path = os.path.join(self.root, "train", "train.json")
            train_imset = json.load(open(train_imset_path))


            for j, video_dict in enumerate(train_imset["videos"]):

                video_name = video_dict["file_names"][0].split("/")[0]
                self.dataset_list.append(video_name)

                frame_num = video_dict["length"]
                self.video_lengths.append(frame_num)

                self.split_name.append("train")
        
        # === Val Set ===
        if "valid" in self.train_splits:
            val_imset_path = os.path.join(self.root, "valid", "valid.json")
            val_imset = json.load(open(val_imset_path))

            for video_dict in val_imset["videos"]:
                video_name = video_dict["file_names"][0].split("/")[0]
                self.dataset_list.append(video_name)

                frame_num = video_dict["length"]
                self.video_lengths.append(frame_num)
                
                self.split_name.append("valid")

        # === Test Set ===
        if "test" in self.train_splits:
            test_imset_path = os.path.join(self.root, "test", "test.json")
            test_imset = json.load(open(test_imset_path))

            for video_dict in test_imset["videos"]:
                video_name = video_dict["file_names"][0].split("/")[0]
                self.dataset_list.append(video_name)

                frame_num = video_dict["length"]
                self.video_lengths.append(frame_num)
                
                self.split_name.append("test")

        self.create_idx_frame_mapping()

        # === Transformations ===
        self.resize = T.Resize(self.resize_to)
        self.resize_nn = T.Resize(self.resize_to, T.InterpolationMode.NEAREST)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    def __len__(self):
        return sum(self.video_lengths)

    def transform(self, image):

        image, _ = remove_borders(image)
        image = self.resize(image)
        image = self.to_tensor(image)
        image = self.normalize(image)

        return image

    def create_idx_frame_mapping(self):
        self.mapping = []

        for video_idx, video_length in enumerate(self.video_lengths):
            video_name = self.dataset_list[video_idx]
            split_name = self.split_name[video_idx]
            for video_frame_idx in range(video_length):
                self.mapping.append((video_name, video_frame_idx, split_name))

    def get_rgb(self, idx):
        video_name, frame_idx, split_name = self.mapping[idx]
        img_dir = os.path.join(self.root, split_name, "JPEGImages", video_name)
        img_list = sorted(glob(os.path.join(img_dir, "*.jpg")), key=lambda x: int(x.split("/")[-1].split(".")[0]))
        frame_num = len(img_list)

        input_frames = torch.zeros((2 * self.N + 1, 3, self.resize_to[0], self.resize_to[1]), dtype=torch.float)
        mask = torch.ones(2 * self.N + 1)

        for i, frame_order in enumerate(self.relative_orders):
            frame_idx_real = frame_idx + frame_order

            if frame_idx_real < 0 or frame_idx_real >= frame_num:
                mask[i] = 0
                continue

            frame = Image.open(img_list[frame_idx_real]).convert('RGB')

            frame = self.transform(frame)
            input_frames[i] = frame
        
        return input_frames, mask

    def __getitem__(self, idx):
        """
        :return:
            input_features: RGB frames [t-N, ..., t+N]
                            in shape (2*N + 1, 3, H, W)

            frame_masks: Mask for input_features indicating if frame is available
                            in shape (2*N + 1)
        """

        input_frames, frame_masks = self.get_rgb(idx)             # (2N + 1, 3, H, W), (2N + 1)

        return input_frames, frame_masks



class YTVIS_val(data.Dataset):
    def __init__(self, args):
        """
        Currently, validation class uses trainset labels and data
        """
        self.root = args.root

        imset_path = os.path.join(self.root, "train", "train.json")
        imset = json.load(open(imset_path))

        self.N = args.N
        self.relative_orders = list(range(-self.N, self.N + 1))
        self.resize_to = args.resize_to
        self.patch_size = args.patch_size
        self.token_num = args.token_num

        
        # === Get Video Names and Lengths ===
        self.dataset_list = []
        for video_dict in imset["videos"][600:900]:
            video_name = video_dict["file_names"][0].split("/")[0]
            self.dataset_list.append(video_name)

        # === Transformations ===
        self.resize = T.Resize(self.resize_to)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.to_one_hot = To_One_Hot(20, shuffle=False)

    def __len__(self):
        return len(self.dataset_list)

    def transform(self, image, borders):

        image, borders = remove_borders(image, borders)
        image = self.resize(image)
        image = self.to_tensor(image)
        image = self.normalize(image)

        return image, borders

    def get_rgb(self, video_name):
        img_dir = os.path.join(self.root, "train", "JPEGImages", video_name)
        img_list = sorted(glob(os.path.join(img_dir, "*.jpg")), key=lambda x: int(x.split("/")[-1].split(".")[0]))
        frame_num = len(img_list)
        
        borders = None
        input_frames = torch.zeros(frame_num, 3, self.resize_to[0], self.resize_to[1], dtype=torch.float)
        for i in range(frame_num):
            frame = Image.open(img_list[i]).convert('RGB')
            frame, borders_i = self.transform(frame, borders)
            input_frames[i] = frame
            if borders is None:
                borders = borders_i

        model_input = torch.zeros(frame_num + 2 * self.N, 3, self.resize_to[0], self.resize_to[1], dtype=torch.float)
        input_masks = torch.ones(frame_num + 2 * self.N)
        
        for frame_idx in range(frame_num + 2 * self.N):
            
            frame_idx_real = frame_idx - self.N

            if frame_idx_real < 0 or frame_idx_real >= frame_num:
                input_masks[frame_idx] = 0
                continue

            model_input[frame_idx] = input_frames[frame_idx_real]

        assert (input_masks == 0).sum() == 2 * self.N
        assert input_masks[:self.N].sum() == 0
        assert input_masks[-self.N:].sum() == 0

        return model_input, input_masks, borders

    def get_gt_masks(self, video_name, border):
        mask_path = os.path.join(self.root, "train", "Annotations", video_name)
        mask_list = sorted(glob(os.path.join(mask_path, "*.png")), key=lambda x: int(x.split("/")[-1].split(".")[0]))
        frame_num = len(mask_list)

        first_mask = Image.open(mask_list[0]).convert('P')
        first_mask_np = np.array(first_mask, np.uint8)
        H, W = first_mask_np.shape
        obj_n = first_mask_np.max() + 1

        bottom = border["bottom"]
        top = border["top"]
        right = border["right"]
        left = border["left"]

        H = H - bottom - top
        W = W - right - left
        masks = torch.zeros(frame_num, 20, H, W, dtype=torch.float)
        for i in range(frame_num):
            mask = Image.open(mask_list[i]).convert('P')
            mask = np.array(mask, np.uint8)

            # Discard borders 
            if bottom == 0: mask = mask[top:]
            else: mask = mask[top:-bottom]

            if right == 0: mask = mask[:, left:]
            else: mask = mask[:, left:-right]

            if i == 0:
                mask, obj_list = self.to_one_hot(mask)
                obj_n = len(obj_list) + 1
            else:
                mask, _ = self.to_one_hot(mask, obj_list)

            masks[i] = mask

        return masks, obj_n

    def __getitem__(self, idx):
        """
        :return:
            input_frames: (#frames + 2N, 3, H, W)
            frame_masks: (#frames + 2N)
            mask: (#frames, #objects, H, W)
        """

        video_name = self.dataset_list[idx]

        # === DINO Features of Frames ===
        input_frames, frame_masks, border = self.get_rgb(video_name)             # (#frames + 2N, 3, H, W), (#frames + 2N)
        mask, obj_n = self.get_gt_masks(video_name, border)
        mask = mask[:, :obj_n]
        mask = torch.argmax(mask, dim=1)

        return input_frames, frame_masks, mask