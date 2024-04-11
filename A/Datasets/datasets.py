from torch.utils.data import Dataset
import os
import cv2
import random
import torch
from tqdm import tqdm
from .dataset_utils import simul_transform



class DIV2K(Dataset):
    def __init__(self, cfg, mode):
        super(DIV2K, self).__init__()
        self.cfg = cfg
        self.mode = mode

        # Locate the path of the dataset and list images
        
        if self.mode == "train":
            lr_names = os.listdir(cfg.LR_TRAIN_PATH)
            hr_names = os.listdir(cfg.HR_TRAIN_PATH)
            lr_paths = [os.path.join(cfg.LR_TRAIN_PATH, hr_name[:-4]+"x4.png") for hr_name in hr_names]
            hr_paths = [os.path.join(cfg.HR_TRAIN_PATH, hr_name) for hr_name in hr_names]
    
        else:
            lr_names = os.listdir(cfg.LR_VAL_PATH)
            hr_names = os.listdir(cfg.HR_VAL_PATH)
            lr_paths = [os.path.join(cfg.LR_VAL_PATH, hr_name[:-4]+"x4.png") for hr_name in hr_names]
            hr_paths = [os.path.join(cfg.HR_VAL_PATH, hr_name) for hr_name in hr_names]


        # Load, preprocess, and store images into memery to save I/O time
        print("Loading lr images for " + "training" if self.mode == "train" else "validation")
        self.lr_list = [self._load_and_preprocess_img(lr_paths[index]) for index in tqdm(range(len(lr_paths)))]
        print("Loading hr images for " + "training" if self.mode == "train" else "validation")
        self.hr_list = [self._load_and_preprocess_img(hr_paths[index]) for index in tqdm(range(len(hr_paths)))]

    def _load_and_preprocess_img(self, img_path):
        img = cv2.imread(img_path)
        # BGR->RGB HWC->CHW
        img = img[:, :, (2, 1, 0)].transpose((2, 0, 1))

        return img

    def __getitem__(self, ind):
        lr_img = self.lr_list[ind]
        hr_img = self.hr_list[ind]

        # Get height and width of the lr image
        _, hl, wl = lr_img.shape

        # The size of the cropped image
        crop_size = self.cfg.PATCH_SIZE // self.cfg.SCALE

        # Where to start cropping
        h_start = random.randint(0, hl - crop_size)
        w_start = random.randint(0, wl - crop_size)

        # Crop lr and hr image patches
        lr_crop = lr_img[:, h_start:h_start + crop_size, w_start:w_start + crop_size].copy()
        hr_crop = hr_img[:, h_start * self.cfg.SCALE: (h_start + crop_size) * self.cfg.SCALE,
                  w_start * self.cfg.SCALE: (w_start + crop_size) * self.cfg.SCALE].copy()

        # Perform data augmentation
        lr_crop, hr_crop = simul_transform(lr_crop, hr_crop, flip=True, rotation=True)

        # Normalize the data and convert them into tensors
        lr_t = (torch.from_numpy(lr_crop).to(dtype=torch.float32) / 255) 
        hr_t = (torch.from_numpy(hr_crop).to(dtype=torch.float32) / 255) 
        return lr_t, hr_t

    def __len__(self):
        return len(self.lr_list)


class BenchmarkDataset(Dataset):
    def __init__(self, cfg, name, length):
        """
        :param cfg: Configuration object
        :param name: Name of the dataset
        """
        super(BenchmarkDataset, self).__init__()
        self.cfg = cfg
        self.length = length

        # Locate the path of the dataset and list images
        if name != "DIV2K":
            test_path = self.cfg.TEST_PATH
            hr_test_path = os.path.join(test_path, name, "HR")
            lr_test_path = os.path.join(test_path, name, "LR_bicubic", "X4")
        else:
            hr_test_path = cfg.HR_VAL_PATH
            lr_test_path = cfg.LR_VAL_PATH
        
        lr_names = os.listdir(lr_test_path)
        hr_names = os.listdir(hr_test_path)
        
        self.lr_paths = [os.path.join(lr_test_path, hr_name[:-4]+"x4.png") for hr_name in hr_names]
        self.hr_paths = [os.path.join(hr_test_path, hr_name) for hr_name in hr_names]
        # Load, preprocess, and store images into memery to save I/O time
        print("Loading lr images for testing")
        self.lr_list = [self._load_and_preprocess_img(self.lr_paths[index], hr=False)
                        for index in tqdm(range(len(self.lr_paths)))]
        print("Loading hr images for testing")
        self.hr_list = [self._load_and_preprocess_img(self.hr_paths[index], hr=True)
                        for index in tqdm(range(len(self.hr_paths)))]

    def _load_and_preprocess_img(self, img_path, hr=True):
        img = cv2.imread(img_path)
        crop_size = self.cfg.PATCH_SIZE if hr else self.cfg.PATCH_SIZE // self.cfg.SCALE
        h, w, _ = img.shape
        crop_flag = False

        # Resize the image if its size is smaller than that of a single patch
        if h < crop_size:
            h = crop_size
            crop_flag = True
        if w < crop_size:
            w = crop_size
            crop_flag = True
        if crop_flag:
            img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        # BGR->RGB HWC->CHW
        img = img[:, :, (2, 1, 0)].transpose((2, 0, 1))

        return img

    def __getitem__(self, ind):
        # The length of the dataset may exceed its actual volume,
        # use a remainder to determine which image to retrieve
        img_ind = ind % (len(self.lr_list))

        lr_img = self.lr_list[img_ind]
        hr_img = self.hr_list[img_ind]
        path_pair = (self.lr_paths, self.hr_paths)
        # Get height and width of the lr image
        _, hl, wl = lr_img.shape

        # The size of the cropped image
        crop_size = self.cfg.PATCH_SIZE // self.cfg.SCALE

        # Where to start cropping
        h_start = random.randint(0, hl - crop_size)
        w_start = random.randint(0, wl - crop_size)

        # Crop lr and hr image patches
        lr_crop = lr_img[:, h_start:h_start + crop_size, w_start:w_start + crop_size].copy()
        hr_crop = hr_img[:, h_start * self.cfg.SCALE: (h_start + crop_size) * self.cfg.SCALE,
                  w_start * self.cfg.SCALE: (w_start + crop_size) * self.cfg.SCALE].copy()
        crop_pos = (h_start, w_start)
        # Normalize the data and convert them into tensors
        lr_t = (torch.from_numpy(lr_crop).to(dtype=torch.float32) / 255) 
        hr_t = (torch.from_numpy(hr_crop).to(dtype=torch.float32) / 255) 
        return lr_t, hr_t, path_pair, crop_pos

    def __len__(self):
        return self.length


