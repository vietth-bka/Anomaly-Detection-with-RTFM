print('Ver2 extractor')
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import numpy as np
import torch
from natsort import natsorted
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm
import torchvision.transforms as transforms
from gtransform import *

def PIL_pad(image):
    w, h = image.size
    d = int(w - h)
    if d == 0:
        return image
    elif d > 0:
        return transforms.Pad((0, d//2, 0, d - d//2))(image)
    else:
        return transforms.Pad((abs(d)//2, 0, abs(d) - abs(d)//2, 0))(image)

def load_frame(frame_file):
	# print(frame_file)
	data = Image.open(frame_file)
	data = PIL_pad(data)
	data = data.resize((256, 256), Image.ANTIALIAS)
	data = torchvision.transforms.TenCrop(224)(data)
	data = [transforms.ToTensor()(ten_crop) for ten_crop in data]
	data = torch.stack(data, 0)

	return data


def load_rgb_batch(frames_dir, rgb_files, frame_indices):
	# batch_data = np.zeros(frame_indices.shape + (256,340,3))
	batch_data = torch.zeros(frame_indices.shape+(10,3,224,224))
	for i in range(frame_indices.shape[0]):
		for j in range(frame_indices.shape[1]):
			batch_data[i,j,:,:,:,:] = load_frame(os.path.join(frames_dir, rgb_files[frame_indices[i][j]]))

	return batch_data


def oversample_data(data):
    mean = [114.75, 114.75, 114.75]
    std = [57.375, 57.375, 57.375]

    data = data*255
    transform = My_Normalize(mean, std)
    data = transform(data)
    # print('data:', data.shape)
    # data = data.permute(1, 0, 3, 2, 4, 5).contiguous() #10xBx3x16x224x224
    data = data.permute(2, 0, 1, 4, 5, 3).contiguous() #10xBx16x224x224x3

    return data

def repeat_lastInd(indices):
	a = range(16)
	b = np.array_split(a, len(indices), axis=0)
	c = [len(i) for i in b]

	out = []
	for i in range(len(c)):
		out.extend([indices[i]]*c[i])
	
	return out


def run(i3d, frequency, frames_dir, batch_size, sample_mode):
	assert(sample_mode in ['oversample', 'center_crop'])
	# print("batchsize", batch_size)
	chunk_size = 16
	def forward_batch(b_data):		
		# b_data = b_data.transpose([0, 4, 1, 2, 3])
		# b_data = torch.from_numpy(b_data)   # b,c,t,h,w  # 40x3x16x224x224
		b_data = b_data.permute(0, 4, 1, 2, 3).contiguous() #Bx3x16x224x224
		with torch.no_grad():
			b_data = Variable(b_data.cuda()).float()
			inp = {'frames': b_data}
			features = i3d(inp)
		return features.cpu().numpy()

	rgb_files = natsorted([i for i in os.listdir(frames_dir)])
	frame_cnt = len(rgb_files)
	# Cut frames
	assert(frame_cnt > chunk_size)
	
	clipped_length = frame_cnt - chunk_size
	clipped_length = (clipped_length // frequency) * frequency  # The start of last chunk
	frame_indices = [] # Frames to chunks
	for i in range(clipped_length // frequency + 1):
		frame_indices.append([j for j in range(i * frequency, i * frequency + chunk_size)])
	
	if frame_cnt % 16 != 0:
		print(frame_cnt, frame_cnt%16)
		last_indices = range(frame_cnt - frame_cnt % 16, frame_cnt)	
		last_indices = repeat_lastInd(last_indices)
		print(last_indices)
		frame_indices.append(last_indices)
	
	frame_indices = np.array(frame_indices)
	# print('frame_indices:', frame_indices.shape)
	chunk_num = frame_indices.shape[0]
	batch_num = int(np.ceil(chunk_num / batch_size))    # Chunks to batches
	print('Total frams:', frame_cnt, "chunk_num:", chunk_num, "Batch num:", batch_num)
	frame_indices = np.array_split(frame_indices, batch_num, axis=0)
	#print('Frames indices:', [i.shape for i in frame_indices])
	
	if sample_mode == 'oversample':
		full_features = [[] for i in range(10)]
	else:
		full_features = [[]]


	for batch_id in tqdm(range(batch_num)): 
		batch_data = load_rgb_batch(frames_dir, rgb_files, frame_indices[batch_id]) #19x16x224x224x3
		# print("batch_data shape:", batch_data.shape)
		if(sample_mode == 'oversample'):
		   batch_data_ten_crop = oversample_data(batch_data)		
		#    print("batch_data_ten_crop:", batch_data_ten_crop.shape)
		   for i in range(10):
			   assert(batch_data_ten_crop[i].shape[-2]==224)
			   assert(batch_data_ten_crop[i].shape[-3]==224)
			#    print('oversample shape:', batch_data_ten_crop[i].shape)
			   temp = forward_batch(batch_data_ten_crop[i])
			   full_features[i].append(temp)

		elif(sample_mode == 'center_crop'):
			batch_data = batch_data[:,:,16:240,58:282,:]
			assert(batch_data.shape[-2]==224)
			assert(batch_data.shape[-3]==224)
			print('batch_data shape:', batch_data.shape)
			temp = forward_batch(batch_data)
			full_features[0].append(temp)
	
	full_features = [np.concatenate(i, axis=0) for i in full_features]
	full_features = [np.expand_dims(i, axis=0) for i in full_features]
	full_features = np.concatenate(full_features, axis=0)
	full_features = full_features[:,:,:,0,0,0]
	full_features = np.array(full_features).transpose([1,0,2])
	return full_features
