# --- from pytorch-resnet3d ---
import torchvision
import random
from PIL import Image
import numbers
import torch
import torchvision.transforms.functional as F

class GroupResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)
        
    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images

class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupRandomHorizontalFlip(object):
    def __call__(self, img_group):
        if random.random() < 0.5:
            img_group = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
        return img_group

class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor): # (T, 3, 224, 224)
        for b in range(tensor.size(0)):
            for t, m, s in zip(tensor[b], self.mean, self.std):
                t.sub_(m).div_(s)
        return tensor

class LoopPad(object):

    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, tensor):
        length = tensor.size(0)

        if length==self.max_len:
            return tensor

        # repeat the clip as many times as is necessary
        n_pad = self.max_len - length
        pad = [tensor]*(n_pad//length)
        if n_pad%length>0:
            pad += [tensor[0:n_pad%length]]

        tensor = torch.cat([tensor]+pad, 0)
        return tensor

# NOTE: Returns [0-255] rather than torchvision's [0-1]
class ToTensor(object):
    def __init__(self):
        self.worker = lambda x: F.to_tensor(x)*255

    def __call__(self, img_group):
        img_group = [self.worker(img) for img in img_group]
        return torch.stack(img_group, 0)

# -- tianyu0207 --

"""mean = [114.75, 114.75, 114.75]
std = [57.375, 57.375, 57.375]

split == '10_crop_ucf'

transform = transforms.Compose([
            GroupResize(256),
            GroupTenCrop(224),
            GroupTenCropToTensor(),
            GroupNormalize_ten_crop(mean, std),
            LoopPad(max_len),
            ])"""

class GroupTenCrop(object):
    def __init__(self, size):        
        transform = torchvision.transforms.Compose([
        torchvision.transforms.TenCrop(size),
        torchvision.transforms.Lambda(lambda crops: torch.stack([torchvision.transforms.ToTensor()(crop) for crop in crops])),
        ])
        self.worker = transform 
    
    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class ToTensor_tianyu0207(object):
    def __init__(self):
        self.worker = lambda x: F.to_tensor(x) * 255
    def __call__(self, img_group):
        img_group = [self.worker(img) for img in img_group]
        return torch.stack(img_group, 0)

# --- Holdwsq ---

class GroupTenCropToTensor(object):
    def __init__(self):
        self.worker = lambda crops: torch.stack([torchvision.transforms.ToTensor()(crop) * 255 for crop in crops])
    
    def __call__(self, crops):
        group_ = [self.worker(crop) for crop in crops]
        stack = torch.stack(group_, 1)
        return stack

class GroupTenNormalize(object):
  def __init__(self, mean, std):
      self.worker = GroupNormalize(mean, std)
  
  def __call__(self, crops):
      group_ = [self.worker(crop) for crop in crops]
      stack = torch.stack(group_, 0)
      return stack

class GroupTenCrop_new(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.TenCrop(size)
    
    def __call__(self, img_group):
        # 经过tencrop后原来的一帧被裁剪为10张，并由元素的形式保存
        group_ = [self.worker(img) for img in img_group]
        return group_

class My_ToTensor(object):
    def __init__(self):
        self.worker = lambda x: x*255
    
    def __call__(self, img_group):
        print('ToTensor')
        img_group = [torch.stack(t, 0) for t in img_group]
        group_ = [self.worker(img) for img in img_group]
        return group_

class My_Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor): # (T, 3, 224, 224)
        #print("Normalize")
        for b in range(len(tensor)):
            for j in range(len(tensor[b])):
                for i in range(tensor[b][j].shape[0]):
                    for t, m, s in zip(tensor[b][j][i], self.mean, self.std):
                        t.sub_(m).div_(s)
        return tensor


if __name__ == '__main__':
    import torchvision.transforms as transforms
    mean = [114.75, 114.75, 114.75]
    std = [57.375, 57.375, 57.375]

    transform = transforms.Compose([
                GroupResize(256),
                GroupTenCrop_new(224),
                My_ToTensor(),
                My_Normalize(mean, std),
                ])
    t1 = GroupResize(256)
    t2 = GroupTenCrop_new(224)
    t3 = My_ToTensor()
    t4 = My_Normalize(mean, std)

    img = torch.randn(19, 16, 3, 224, 224)
    img = t1(img)
    print(len(img), img[0].shape)
    img = t2(img)
    print(len(img), len(img[0]), img[0][0].shape)
    print(img[0][0][0])
    img = t3(img)
    # img = [torch.stack(t, 0) for t in img]
    img = t4(img)
    img = torch.stack(img, 0)
    # img = img.transpose(1, 0).contiguous()
    print(img.shape)
    img = img.permute(1, 0, 2, 4, 5, 3).contiguous()
    print(img.shape)
