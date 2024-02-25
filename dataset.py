'''
Use code from here to get data from CelebA/CelebA-HQ datasets
Datasets should have format images/all_images and a file with all attributes
'''
import torch, os, torchvision
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json, argparse

import torchvision.utils as vutils
import random


class CelebA(Dataset):
    def __init__(self, filename = "../data/celeba", split = "train", transforms = T.ToTensor(), selected_attr = list(range(40)), landmarks = False, extras = False):
        super().__init__()
    
        self.features = []
        self.labels = []
        self.landmarks = landmarks
        self.extras = extras
        self.extras_l = []
        # Open file and get basic characteristics
        annotations_filename = os.path.join(filename, "list_attr_celeba.txt")
        annotations = open(annotations_filename, "r").read().splitlines()
        if self.landmarks: land_anno = open(os.path.join(filename, "list_landmarks_align_celeba.txt"), "r").read().splitlines()
        self.size = int(annotations[0])
        self.transform = transforms
        
        if split == "train": img_range = range(1, int((self.size+1)*0.7)+1)
        elif split == "test": img_range = range(int((self.size+1)*0.7)+1, int((self.size+1)*0.9)+1)
        elif split == "val": img_range = range(int((self.size+1)*0.9)+1, self.size+1)
        else: raise NotImplementedError(
              f"Only options for dataset are \"train\", \"test\", \"val\". You entered \"{split}\".")       
        self.land_labels = []
        # Start building real characteristics
        for i in img_range:
            words_in_line = annotations[i+1].split(" ")
            filename_img = words_in_line[0]
            
            # Turn labels from -1 and 1 strings into 0 and 1 ints and then add to list
            label = [(int(w) + 1) // 2 for w in words_in_line[1:] if w] 
            self.labels.append(torch.Tensor(label)[selected_attr])
            
            if self.extras: self.extras_l.append(torch.Tensor(label)[extras])
            # Get filename and add it to list  
            # It is extremely slow to just add the image here, better to let dataloader parallelize          
            img = os.path.join(filename, f"images/{filename_img}")
            self.features.append(img)

            # If you want landmarks, here they are
            if self.landmarks:
                
                land_line_annos = [int(i) for i in [f for f in land_anno[i+1].split(" ") if f!=""][1:]]
                
                self.land_labels.append(land_line_annos)
        

    def __getitem__(self, index):
        filename = self.features[index]
        labels = self.labels[index]
        img = self.transform(Image.open(filename))
        if self.landmarks and self.extras: return img, labels, self.land_labels[index], self.extras_l[index]
        if self.extras: return img, labels, self.extras_l[index]
        if self.landmarks: return img, labels, self.land_labels[index],
        else: return img, labels

    def __len__(self):
        return len(self.labels)
class CelebASeg(Dataset):
    def __init__(self, filename = "/share/datasets/CelebAMask-HQ", split = "train", transforms = T.ToTensor(), seg_transforms = T.ToTensor(), selected_attr = list(range(40)), seg_attr = list(range(17)), landmarks = False):
        super().__init__()
    
        self.features = []
        self.labels = []
        self.seg_transforms = seg_transforms
        # Open file and get basic characteristics
        annotations_filename = os.path.join(filename, "CelebAMask-HQ-attribute-anno.txt")
        annotations = open(annotations_filename, "r").read().splitlines()
        
        self.size = int(annotations[0])
        self.transform = transforms
        self.segments = []
        if split == "train": img_range = range(1, int((self.size+1)*0.7)+1)
        elif split == "test": img_range = range(int((self.size+1)*0.7)+1, int((self.size+1)*0.9)+1)
        elif split == "val": img_range = range(int((self.size+1)*0.9)+1, self.size+1)
        else: raise NotImplementedError(
              f"Only options for dataset are \"train\", \"test\", \"val\". You entered \"{split}\".")       
        parts = ["cloth", "ear_r", "eye_g", "hair", "hat","l_brow", "l_ear", "l_eye", "l_lip", "neck", "neck_l", "nose", "r_brow", "r_eye", "skin", "u_lip"]
        # Start building real characteristics
        for i in img_range:
            words_in_line = annotations[i+1].split(" ")
            filename_img = words_in_line[0]
            
            # Turn labels from -1 and 1 strings into 0 and 1 ints and then add to list
            label = [(int(w) + 1) // 2 for w in words_in_line[1:] if w] 
            self.labels.append(torch.Tensor(label)[selected_attr])

            # Get filename and add it to list  
            # It is extremely slow to just add the image here, better to let dataloader parallelize          
            img = os.path.join(filename, f"{filename}/CelebA-HQ-img/{filename_img}")
            self.features.append(img)

            # If you want landmarks, here they are
        
        for j in img_range:
            seg = []
            for attr in seg_attr:
                i = j - 1
                app = f"{filename}/CelebAMask-HQ-mask-anno/{i//2000}/{(5-len(str(i)))*'0'+str(i)+'_'+parts[attr]}.png"
                seg.append(app)
            self.segments.append(seg)



        

    def __getitem__(self, index):
        filename = self.features[index]
        labels = self.labels[index]
        img = self.transform(Image.open(filename))
        seg = torch.zeros_like(img)
        for inc, file in enumerate(self.segments[index]):
            
            if os.path.isfile(file): 
                # print(self.seg_transforms(Image.open(file)).max())
                # exit()
                
                seg = torch.where(seg==0, seg + self.seg_transforms(Image.open(file)).round(), seg)
        return img, labels, seg[0]
        

    def __len__(self):
        return len(self.labels)
    
'''
Get train, test, and val loaders
'''
def get_dataloaders(filename = "../data/celebahq", batch_size = 64, num_workers = 3, transforms = 
                            T.Compose([T.ToTensor(),
                            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                            T.Resize((128, 128))]), selected_attr = list(range(40)), landmarks = False, extras = False):

    # handle transforms=None case
    

    train_loader = CelebA(filename=filename, split="train", transforms = transforms, selected_attr=selected_attr, landmarks = landmarks, extras = extras)
    train_loader = DataLoader(train_loader, shuffle=True, batch_size=batch_size)

    test_loader = CelebA(filename=filename, split="test", transforms = transforms, selected_attr=selected_attr, landmarks = landmarks, extras = extras)
    test_loader = DataLoader(test_loader, shuffle=True, batch_size=batch_size)

    val_loader = CelebA(filename=filename, split="val", transforms = transforms, selected_attr=selected_attr, landmarks = landmarks, extras = extras)
    val_loader = DataLoader(val_loader, shuffle=True, batch_size=batch_size)

    return train_loader, test_loader, val_loader
class CelebA_HiSD(Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, filename, transform, attr, active, conditions, start=3000, end = 30000, landmarks = False):
        """Initialize and preprocess the CelebA dataset."""
        
        self.transform = transform
        
        self.features = []
        self.labels = []
        self.attr = attr; self.active = active
        self.landmarks = landmarks
        if self.landmarks: land_anno = open(os.path.join(filename, "list_landmarks_align_celeba.txt"), "r").read().splitlines()
        self.land_labels = []
        
        # Open file and get basic characteristics
        annotations_filename = os.path.join(filename, "list_attr_celeba.txt")
        annotations = open(annotations_filename, "r").read().splitlines()
        self.length = int(annotations[0])
        for i in range(start, end):
            words_in_line = annotations[i+1].split(" ")
            filename_img = words_in_line[0]
            
            # Turn labels from -1 and 1 strings into 0 and 1 ints and then add to list
            label = [(int(w) + 1) // 2 for w in words_in_line[1:] if w] 
            if label[attr] == active:
                
                self.labels.append([label[conditions[0]], label[conditions[1]]])

                # Get filename and add it to list  
                # It is extremely slow to just add the image here, better to let dataloader parallelize          
                img = os.path.join(filename, f"images/{filename_img}")
                self.features.append(img)
                if self.landmarks:
                
                    land_line_annos = [int(i) for i in [f for f in land_anno[i+1].split(" ") if f!=""][1:]]
                    
                    self.land_labels.append(land_line_annos)

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        
        image = Image.open(self.features[index])

        return self.transform(image), torch.Tensor(self.labels[index])

    def __len__(self):
        """Return the number of images."""
        
        return len(self.features)


class CelebA_d2d(Dataset):
    def __init__(self, filename = "../data/celeba", split = "train", transforms = T.ToTensor(), attr_ind = 0, landmarks = False):
        super().__init__()
    
        self.features = [[], []]
        self.labels = [[], []]
        self.landmarks = landmarks
        
        # Open file and get basic characteristics
        annotations_filename = os.path.join(filename, "list_attr_celeba.txt")
        annotations = open(annotations_filename, "r").read().splitlines()
        self.size = int(annotations[0])
        self.transform = transforms
        if self.landmarks: land_anno = open(os.path.join(filename, "list_landmarks_align_celeba.txt"), "r").read().splitlines()
        self.land_labels = [[],[]]
        if split == "train": img_range = range(1, int((self.size+1)*0.7)+1)
        elif split == "test": img_range = range(int((self.size+1)*0.7)+1, int((self.size+1)*0.9)+1)
        elif split == "val": img_range = range(int((self.size+1)*0.9)+1, self.size+1)
        else: raise NotImplementedError(
                f"Only options for dataset are \"train\", \"test\", \"val\". You entered \"{split}\".")       

        # Start building real characteristics
        for i in img_range:
            words_in_line = annotations[i+1].split(" ")
            filename_img = words_in_line[0]
            
            # Turn labels from -1 and 1 strings into 0 and 1 ints and then add to list
            label = [(int(w) + 1) // 2 for w in words_in_line[1:] if w] 
            self.labels[(label[attr_ind]+1)//2].append(label)

            # Get filename and add it to list  
            # It is extremely slow to just add the image here, better to let dataloader parallelize          
            img = os.path.join(filename, f"images/{filename_img}")
            self.features[(label[attr_ind]+1)//2].append(img)
            if self.landmarks:
                
                land_line_annos = [int(i) for i in [f for f in land_anno[i+1].split(" ") if f!=""][1:]]
                
                self.land_labels[(label[attr_ind]+1)//2].append(land_line_annos)
        

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.features[0][index]))       
        item_B = self.transform(Image.open(self.features[1][(randind:=random.randint(0, index))]))
        if not self.landmarks: return {'A': item_A, 'B': item_B}
        else: return {'A': (item_A, self.land_labels[0][index]), 'B': (item_B, self.land_labels[1][randind])}

    def __len__(self):
        return min(len(self.labels[0]), len(self.labels[1]))

