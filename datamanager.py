#!/usr/bin/env python


# Adapted for the paper "From Patches to Objects: Exploiting Spatial
# Reasoning for Better Visual Representations"
# Based on:


# The MIT License (MIT)
# Copyright (c) 2020 Massimiliano Patacchiola
# Paper: "Self-Supervised Relational Reasoning for Representation Learning", M. Patacchiola & A. Storkey, NeurIPS 2020
# GitHub: https://github.com/mpatacchiola/self-supervised-relational-reasoning

# Data manager that returns transformations and samplers for each method/dataset.
# If a new method is included it should be added to the "DataManager" class.
# The dataset classes with prefix "Multi" are overriding the original dataset class
# to allow multi-sampling of more images in parallel (required by our method).

import os
import sys
import random
import pickle

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class PatchTinyImageFolder(dset.ImageFolder):
    def __init__(self, repeat_augmentations, patch_size,patch_count,additive, **kwds):
        super().__init__(**kwds)
        self.repeat_augmentations = repeat_augmentations
        # Calculation of the target size for the image
        # Resizing the image once and then sampling patches from it is faster than resizing each patch
        self.size = int((64/patch_size)*64)
        self.patch_size = patch_size
        print("Target size for image:", self.size)
        self.resizing = transforms.Compose([transforms.Resize(size=(self.size, self.size)),transforms.ToTensor()])
        self.to_tensor =transforms.Compose([transforms.ToTensor()])

        # Patch augmentations are missing the random crop and horizontal flip from normal augmentations
        normalize = transforms.Normalize((0.1307,), (0.3081,)) #tiny imagenet
        color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        self.patch_count = patch_count
        self.patch_transformations = transforms.Compose([rnd_color_jitter, rnd_gray, normalize])


        if additive:
            self.mode = "additive"
            print("Patch mode set to additive")
        else:
            self.mode = "not_additive"
            print("Patch mode set to standard")

    def patch_transform(self,image):
        if self.mode == "additive":
            # Sampling the positions of the patches, resample until the patches are not overlapping
            positions = random.sample(range(0,64-self.patch_size), self.patch_count*2)
            while abs(positions[0]-positions[1]) < self.patch_size and abs(positions[2]-positions[3]) < self.patch_size:
                positions = random.sample(range(0,64-self.patch_size), self.patch_count*2)

            target_list = list()
            res_images = list()
            image = self.to_tensor(image)
            for patch in range(self.patch_count):
                target_list.append(np.array([positions[patch*2]/(64-self.patch_size), positions[patch*2+1]/(64-self.patch_size),0]))
                res_im =image[:,positions[patch*2]:positions[patch*2]+self.patch_size,positions[patch*2+1]:positions[patch*2+1]+self.patch_size]
                base_im =torch.zeros((3,64,64))
                new_pos = random.sample(range(0,64-self.patch_size), 2)
                nres_im = self.patch_transformations(res_im)
                base_im[:,new_pos[0]:new_pos[0]+self.patch_size,new_pos[1]:new_pos[1]+self.patch_size] = nres_im
                res_images.append(base_im)
            return res_images, target_list
        
        # Resizing operation is only necessary for the standard implementation
        image = self.resizing(image)
        positions = random.sample(range(0,self.size-64), self.patch_count*2)
        while abs(positions[0]-positions[1]) < 64 and abs(positions[2]-positions[3]) < 64:
            positions = random.sample(range(0,self.size-64), self.patch_count*2)
        
        target_list = list()
        res_images = list()

        for patch in range(self.patch_count):
            target_list.append(np.array([positions[patch*2]/128, positions[patch*2+1]/128,0]))
            res_im =image[:,positions[patch*2]:positions[patch*2]+64,positions[patch*2+1]:positions[patch*2+1]+64]
            res_images.append(self.patch_transformations(res_im))
        return res_images, target_list
                
    def __getitem__(self, index):
        img_path, target = self.imgs[index]
        pic = Image.open(img_path).convert("RGB")
        img = torch.from_numpy(np.array(pic, np.uint8, copy=True))
            
        img_list = list()
        patches, patch_targets = self.patch_transform(pic)
        img_list.append(patches)
        if self.transform is not None:
            for _ in range(self.repeat_augmentations):
               img_transformed = self.transform(pic.copy())
               img_list.append(img_transformed)
        else:
            img_list = None
           
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img_list, patch_targets




class EvalPatchTinyImageFolder(dset.ImageFolder):
    def __init__(self, patch_size,patch_count, **kwds):
        super().__init__(**kwds)
        normalize = transforms.Normalize((0.1307,), (0.3081,)) #tiny imagenet
        self.patch_count =patch_count
        self.size = int((64/patch_size)*64)
        self.resize = transforms.Compose([transforms.Resize(self.size)])
        self.transforms = transforms.Compose([transforms.ToTensor(), normalize])
        self.patch_size = patch_size
        print(self.transform)
            
    def __getitem__(self, index):
        img_path, target = self.imgs[index]
        pic = Image.open(img_path).convert("RGB")

        # This is the standard mode used for additive patch use
        if self.patch_count == 0:
            img_transformed = self.transform(pic.copy())
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img_transformed, target    
        
        img_transformed = self.transform(pic.copy())
        img_transformed = self.resize(img_transformed)

        step_size = int((self.size-64)/2) -1
        patches = img_transformed.unfold(1, 64, step_size).unfold(2, 64, step_size)
        patches = patches.reshape(3,9,64,64)
        
        patches = patches.clone().permute(1,0,2,3)        
        img_list = list()

        if self.patch_count == 1:
            img_list.append(patches[4].unsqueeze(0))
        elif self.patch_count == 3:
            img_list.append(patches[3:6])
        elif self.patch_count == 5:
            img_list.append(torch.stack([patches[i] for i in [1, 3, 4, 5, 7]]))
        elif self.patch_count == 7:
            selected_patches = [patches[i] for i in [1, 3, 4, 5, 7]]
            remaining_patches = [patches[i] for i in [0, 2, 6, 8]]
            random_indices = torch.randperm(len(remaining_patches))[:2]
            random_patches = [remaining_patches[i] for i in random_indices]
            selected_patches.extend(random_patches)
            img_list.append(torch.stack(selected_patches))
        elif self.patch_count == 9:
            img_list.append(patches)

        # Transformation without resizing
        img_transformed = self.transform(pic.copy())

        img_list.append(img_transformed)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_list, target    



class SlimImageFolder(dset.ImageFolder):
    def __init__(self, **kwds):
        super().__init__(**kwds)


class MultiSlimImageFolder(dset.ImageFolder):
    def __init__(self, repeat_augmentations, **kwds):
        super().__init__(**kwds)
        self.repeat_augmentations = repeat_augmentations
            
    def __getitem__(self, index):
        img_path, target = self.imgs[index]
        pic = Image.open(img_path).convert("RGB")
        img = torch.from_numpy(np.array(pic, np.uint8, copy=False))
            
        img_list = list()
        if self.transform is not None:
            for _ in range(self.repeat_augmentations):
               img_transformed = self.transform(pic.copy())
               img_list.append(img_transformed)
        else:
            img_list = None
           
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img_list, target


class DataManager():
    def __init__(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _check(self, dataset):
        datasets_list = ["cifar10", "stl10", "cifar100", 
                         "supercifar100", "tiny", "slim"]
        if(dataset not in datasets_list):
            raise Exception("[ERROR] The dataset " + str(dataset) + " is not supported!")            
        if(dataset=="slim"):
            if(os.path.isdir("./data/SlimageNet64/train")==False):
                raise Exception("[ERROR] The train data of SlimageNet64 has not been found in ./data/SlimageNet64/train \n"
                            + "1) Download the dataset from: https://zenodo.org/record/3672132 \n"
                            + "2) Uncompress the dataset in ./data/SlimageNet64  \n"
                            + "3) Place training images in /train and test images in /test")
        elif(dataset=="tiny"):
            if(os.path.isdir("./data/tiny-imagenet-200/train")==False):                            
                raise Exception("[ERROR] The train data of TinyImagenet has not been found in ./data/tiny-imagenet-200/train \n"
                            + "1) Download the dataset \n"
                            + "2) Uncompress the dataset in ./data/tiny-imagenet-200  \n"
                            + "3) Place training images in /train and test images in /test")                         

    def get_num_classes(self, dataset):
        self._check(dataset)
        if(dataset=="cifar10"): return 10
        elif(dataset=="stl10"): return 10
        elif(dataset=="supercifar100"): return 20
        elif(dataset=="cifar100"): return 100
        elif(dataset=="tiny"): return 200
        elif(dataset=="slim"): return 1000

    def get_train_transforms(self, method, dataset):
        """Returns the training torchvision transformations for each dataset/method.
           If a new method or dataset is added, this file should by modified
           accordingly.
        Args:
          method: The name of the method.
          dataset: The name of the dataset.
        Returns:
          train_transform: An object of type torchvision.transforms.
        """
        self._check(dataset)
        if(dataset=="cifar10"): 
            normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
            side = 32; padding = 4; cutout=0.25
        elif(dataset=="stl10"): 
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            side = 96; padding = 12; cutout=0.111
        elif(dataset=="cifar100" or dataset=="supercifar100"): 
            normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            side = 32; padding = 4; cutout=0.0625
        elif(dataset=="tiny"):
            #Image-Net --> mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            normalize = transforms.Normalize(mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081])
            side = 64; padding = 8; cutout=0.111
        elif(dataset=="slim"):
            #Image-Net --> mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
            side = 64; padding = 8

        if(method=="relationnet" or method=="simclr" or method=="patchbased"):
            color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=0.2)
            rnd_resizedcrop = transforms.RandomResizedCrop(size=side, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333))
            rnd_hflip = transforms.RandomHorizontalFlip(p=0.5)
            #rnd_rot = transforms.RandomRotation(10., resample=2),
            train_transform = transforms.Compose([rnd_resizedcrop, rnd_hflip,
                                                  rnd_color_jitter, rnd_gray, transforms.ToTensor(), normalize])
        elif(method=="deepinfomax"):
            train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        elif(method=="standard" or method=="rotationnet" or method=="deepcluster"):
            train_transform = transforms.Compose([transforms.RandomCrop(side, padding=padding), 
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(), normalize])
        elif(method=="finetune"):
            rnd_affine = transforms.RandomApply([transforms.RandomAffine(18, scale=(0.9, 1.1),
                                                                         translate=(0.1, 0.1), shear=10,
                                                                         resample=Image.BILINEAR, fillcolor=0)], p=0.5)
            train_transform = transforms.Compose([#transforms.RandomCrop(side, padding=padding),
                                                  transforms.RandomHorizontalFlip(),
                                                  rnd_affine,
                                                  transforms.ToTensor(), normalize,
                                                  #transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))]) #pytorch default
                                                  transforms.RandomErasing(p=0.5, scale=(cutout, cutout), ratio=(1.0, 1.0))])
        elif(method=="lineval"):
            train_transform = transforms.Compose([transforms.ToTensor(), normalize])
            """
            # This configuration was used to test the effect of augmentations on the lineval to diminish the effect of
            # overfitting for large representations:
            rnd_affine = transforms.RandomApply([transforms.RandomAffine(18, scale=(0.9, 1.1),
                                                                         translate=(0.1, 0.1), shear=10,
                                                                         resample=Image.BILINEAR, fillcolor=0)], p=0.5)
            train_transform = transforms.Compose([rnd_affine,
                                                  transforms.ToTensor(), normalize,

                                                  transforms.RandomErasing(p=0.5, scale=(cutout, cutout), ratio=(1.0, 1.0))])"""
        else:
            raise Exception("[ERROR] The method " + str(method) + " is not supported!")

        return train_transform

    def get_train_loader(self, dataset, data_type, additive, data_size, train_transform, repeat_augmentations, num_workers=8, drop_last=False, patch_count = 0, patch_size = 0):
        """Returns the training loader for each dataset/method.
           If a new method or dataset is added, this method should by modified
           accordingly.
        Args:
          method: The name of the method.
          dataset: The name of the dataset.
          data_type: The type of data multi (multiple images in parallel),
            single (one image at the time), unsupervised (used in STL10 to load 
            the unlabeled data split).
          data_size: the mini-batch size.
          train_transform: the transformations used by the sampler, they
            should be returned by the method get_train_transforms().
          repeat_augmentations: repeat the augmentations on the same image
            for the specified number of times (needed by RelationNet and SimCLR).
          num_workers: the total number of parallel workers for the samples.
          drop_last: it drops the last sample if the mini-batch cannot be
             aggregated, necessary for methods like DeepInfomax.            
        Returns:
          train_loader: The loader that can be used a training time.
          train_set: The train set (used in DeepCluster)
        """
        self._check(dataset)
        from torch.utils.data.dataset import Subset
        if(data_type=="multi"):
        #Used for: Relational reasoning, SimCLR
            if(dataset=="cifar10"): 
                train_set = MultiCIFAR10(repeat_augmentations, root="./data", train=True, transform=train_transform, download=True)
            elif(dataset=="stl10"):
                train_set = MultiSTL10(repeat_augmentations, root="./data", split="unlabeled", transform=train_transform, download=True)
            elif(dataset=="cifar100"): 
                train_set = MultiCIFAR100(repeat_augmentations, root="./data", train=True, transform=train_transform, download=True)
            elif(dataset=="tiny"): 
                train_set = MultiTinyImageFolder(repeat_augmentations, root="./data/tiny-imagenet-200/train", transform=train_transform)
            elif(dataset=="slim"): 
                train_set = MultiSlimImageFolder(repeat_augmentations, root="./data/SlimageNet64/train", transform=train_transform)
        elif(data_type=="single"):
        #Used for: deepinfomax, rotationnet, standard, lineval, finetune, deepcluster
            if(dataset=="cifar10"): 
                train_set = dset.CIFAR10("data", train=True, transform=train_transform, download=True)
            elif(dataset=="stl10"):
                train_set = dset.STL10(root="./data", split="train", transform=train_transform, download=True)
            elif(dataset=="supercifar100"):
                train_set = SuperCIFAR100("data", train=True, transform=train_transform, download=True)
            elif(dataset=="cifar100"):
                train_set = dset.CIFAR100("data", train=True, transform=train_transform, download=True)
            elif(dataset=="tiny"): 
                train_set = TinyImageFolder(root="./data/tiny-imagenet-200/train", transform=train_transform)
            elif(dataset=="slim"): 
                train_set = SlimImageFolder(root="./data/SlimageNet64/train", transform=train_transform)
        elif(data_type=="patch"):
        #Used for: patchnet
            if(dataset=="tiny"): 
                train_set = PatchTinyImageFolder(repeat_augmentations,patch_size=patch_size,patch_count=patch_count,
                    root="./data/tiny-imagenet-200/train", transform=train_transform, additive=additive)
            if(dataset=="cifar10"): 
                train_set = PatchMultiCIFAR10(repeat_augmentations,patch_size=patch_size,patch_count=patch_count,
                    root="./data", transform=train_transform, download=False, additive=additive)
            if(dataset=="cifar100"): 
                train_set = PatchMultiCIFAR100(repeat_augmentations,patch_size=patch_size,patch_count=patch_count,
                    root="./data", transform=train_transform, download=False, additive=additive)
            if(dataset=="stl10"): 
                train_set = PatchMultiSTL10(repeat_augmentations,patch_size=patch_size,patch_count=patch_count,
                    root="./data",split="unlabeled", transform=train_transform, download=False, additive=additive)
        elif(data_type=="patch_eval"):
        #Used for: patchnet
            if(dataset=="tiny"): 
                print("calling tiny patch train loader")
                train_set = EvalPatchTinyImageFolder(patch_size,patch_count,root="./data/tiny-imagenet-200/train", transform=train_transform)
            if(dataset=="cifar10"): 
                print("calling cifar10 patch train loader")
                train_set = EvalPatchCifar10ImageFolder(patch_size,patch_count,root="./data", transform=train_transform)
            if(dataset=="cifar100"): 
                print("calling cifar100 patch train loader")
                train_set = EvalPatchCifar100ImageFolder(patch_size,patch_count,root="./data", transform=train_transform)
            if(dataset=="supercifar100"): 
                print("calling cifar100 patch train loader")
                train_set = EvalPatchSuperCIFAR100(patch_size=patch_size,patch_count=patch_count,root="./data", train=True, download=False, transform=train_transform)
            if(dataset=="stl10"): 
                print("calling stl10 patch train loader")
                train_set = EvalPatchSTL10ImageFolder(patch_size,patch_count,split="train",root="./data", download=False, transform=train_transform)

        elif(data_type=="unsupervised"):
            if(dataset=="stl10"):
                train_set = dset.STL10(root="data", split="unlabeled", transform=train_transform, download=True)
        else:
            raise Exception("[ERROR] The type " + str(data_type) + " is not supported!")

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=data_size, shuffle=True, 
                                                   num_workers=num_workers, pin_memory=True, drop_last=drop_last)        
        return train_loader, train_set


    def get_test_loader(self, dataset, data_size, num_workers=8, patch_count = 0, patch_size = 0):
        # Patch count here means the number of patches added to each image.
        print("Started getting testloader")
        self._check(dataset)        
        if(dataset=="cifar10"): 
            normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            test_set = dset.CIFAR10("./data", train=False, transform=test_transform, download=True)
            if patch_size > 0 :
                print("Setting up patchbased testloader")
                test_set = EvalPatchCifar10ImageFolder(patch_size = patch_size, train=False, patch_count=patch_count,root="./data", transform=test_transform)
        elif(dataset=="stl10"): 
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            test_set = dset.STL10(root="./data", split="test", transform=test_transform, download=False)
            if patch_size > 0 :
                print("Setting up patchbased testloader")
                test_set = EvalPatchSTL10ImageFolder(patch_size = patch_size, split="test", patch_count=patch_count,root="./data", transform=test_transform)
        elif(dataset=="supercifar100"): 
            normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            test_set = SuperCIFAR100("./data", train=False, transform=test_transform, download=True)
            if patch_size > 0 :
                print("Setting up patchbased testloader")
                test_set = EvalPatchSuperCIFAR100(patch_size = patch_size, train=False, patch_count=patch_count,root="./data", transform=test_transform)
        elif(dataset=="cifar100"): 
            normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            test_set = dset.CIFAR100("./data", train=False, transform=test_transform, download=True)
            if patch_size > 0 :
                print("Setting up patchbased testloader")
                test_set = EvalPatchCifar100ImageFolder(patch_size = patch_size,train=False,patch_count=patch_count,root="./data", transform=test_transform)
        elif(dataset=="tiny"):
            print("Creating tiny dataloader")
            normalize = transforms.Normalize(mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.3081, 0.3081])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            if patch_size > 0 :
                print("Setting up patchbased testloader")
                test_set = EvalPatchTinyImageFolder(patch_size = patch_size,patch_count=patch_count,root="./data/tiny-imagenet-200/val/images", transform=test_transform)
            else:
                print("Setting up standard test loader") 
                test_set = TinyImageFolder(root="./data/tiny-imagenet-200/val/images", transform=test_transform)
        elif(dataset=="slim"):
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])
            test_set = SlimImageFolder(root="./data/SlimageNet64/test", transform=test_transform)
        print("no if-else clause triggered")
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=data_size, shuffle=False, 
                                                   num_workers=num_workers, pin_memory=True)
        return test_loader

