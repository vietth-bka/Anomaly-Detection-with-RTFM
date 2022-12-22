from glob import glob
from tqdm import tqdm
import os

folds = ['Anomaly-Videos-Part-1_feat_v2', 'Anomaly-Videos-Part-2_feat_v2', 'Anomaly-Videos-Part-3_feat_v2','Anomaly-Videos-Part-4_feat_v2',
         'Testing_Normal_Videos_feat_v2', 'Training-Normal-Videos-Part-1_feat_v2', 'Training-Normal-Videos-Part-2_feat_v2']

ft_list = [glob('/media/v100/DATA4/thviet/AnomalyDetection/I3D_extractor/Features/v2/'+f+'/*.npy') for f in folds]
ft_list = [i for j in ft_list for i in j]
# print(len(ft_list), ft_list[:10])

ft_dict = {}
for f in ft_list:
    if f.split('/')[-1] not in ft_dict:
        ft_dict[f.split('/')[-1]] = f
    else:
        print(f'overlaped video {f}')

test_list_pth = '../RTFM/list/ucf-i3d-test.list'
train_list_pth = '../RTFM/list/ucf-i3d.list'

with open(test_list_pth, 'r') as f:
    test = f.readlines()
test = [t.split('\\')[-1].replace('_i3d','').replace('\n','') for t in test]

with open(train_list_pth, 'r') as f:
    train = f.readlines()
train = [t.split('\\')[-1].replace('_i3d','').replace('\n','') for t in train]

print(len(test), len(train))

f = open('ucf-test_v2.txt', 'w')
for t in test:
    if t in ft_dict:
        f.write(ft_dict[t]+'\n') 
f.close()

f = open('ucf-train_v2.txt', 'w')
for t in train:
    if t in ft_dict:
        f.write(ft_dict[t]+'\n') 
f.close()