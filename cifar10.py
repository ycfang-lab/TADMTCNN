import os,math
import torch
from torch.functional import unique
from torch.utils import data
import torchvision
import numpy as np
import PIL.Image as Image 
from torchvision import transforms
from torch.utils.data import Dataset


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



class cifar10(Dataset):
    def __init__(self, cifar10_dir, mode='train'):
        self.transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                            ])
        images = []
        labels = []
        if mode == 'train':
            for i in range(1, 1 + 5):
                data_batch = unpickle(os.path.join(cifar10_dir, 'data_batch_{}'.format(i)))
                images_batch = data_batch[b'data'].reshape(-1, 3, 32, 32)
                labels_batch = data_batch[b'labels']
                images.append(images_batch)
                labels.append(labels_batch)
        else:
            data_batch = unpickle(os.path.join(cifar10_dir, 'test_batch'))
            images_batch = data_batch[b'data'].reshape(-1, 3, 32, 32)
            labels_batch = data_batch[b'labels']
            images.append(images_batch)
            labels.append(labels_batch)
        self.images = np.concatenate(images)
        self.labels = np.concatenate(labels)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.transforms(self.images[idx].transpose(1, 2, 0))
        label = self.labels[idx]
        return sample, label


class classifier(torch.nn.Module):
    def __init__(self, in_channels, task_group):
        super(classifier, self).__init__()
        self.task_group = task_group
        group_list = task_group.keys()
        self._classifier = torch.nn.ModuleDict(
            {task:torch.nn.Linear(in_channels, 1) for group in group_list for task in task_group[group]}
        )
    
    def forward(self, inputs):
        groups = inputs.keys()
        return {task:self._classifier[task](inputs[group]) for group in groups for task in self.task_group[group]}

def CelebAcast2num_40(taskName):
    label_indicate={
            '5_o_Clock_Shadow':0,
            'Arched_Eyebrows':1,
            'Attractive':2,
            'Bags_Under_Eyes':3,
            'Bald':4,
            'Bangs':5,
            'Big_Lips':6,
            'Big_Nose':7,
            'Black_Hair':8,
            'Blond_Hair':9,
            'Blurry':10,
            'Brown_Hair':11,
            'Bushy_Eyebrows':12,
            'Chubby':13,
            'Double_Chin':14,
            'Eyeglasses':15,
            'Goatee':16,
            'Gray_Hair':17,
            'Heavy_Makeup':18,
            'High_Cheekbones':19,
            'Male':20,
            'Mouth_Slightly_Open':21,
            'Mustache':22,
            'Narrow_Eyes':23,
            'No_Beard':24,
            'Oval_Face':25,
            'Pale_Skin':26,
            'Pointy_Nose':27,
            'Receding_Hairline':28,
            'Rosy_Cheeks':29,
            'Sideburns':30,
            'Smiling':31,
            'Straight_Hair':32,
            'Wavy_Hair':33,
            'Wearing_Earrings':34,
            'Wearing_Hat':35,
            'Wearing_Lipstick':36,
            'Wearing_Necklace':37,
            'Wearing_Necktie':38,
            'Young':39
    }
    return label_indicate[taskName]