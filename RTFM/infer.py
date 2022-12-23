import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
from utils import Visualizer
from torch.utils.data import DataLoader
import option
from model import Model
from dataset import Dataset, Infer
from config import *
from scipy.io import loadmat

def test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)

        for i, input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))

        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy')
        else:
            gt = np.load('list/gt-ucf.npy')

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        # np.save('fpr.npy', fpr)
        # np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))

        # precision, recall, th = precision_recall_curve(list(gt), pred)
        # pr_auc = auc(recall, precision)
        # np.save('precision.npy', precision)
        # np.save('recall.npy', recall)
        # viz.plot_lines('pr_auc', pr_auc)
        # viz.plot_lines('auc', rec_auc)
        # viz.lines('scores', pred)
        # viz.lines('roc', tpr, fpr)
        return rec_auc

def get_gt(file):
    gt = []
    temporal_root = '/home/quantum/Desktop/Workspace/vietth/Anomaly_detection/Datasets/Temporal_Anomaly_Annotation_For_Testing_Videos/Matlab_formate'
    # '/home/yu/PycharmProjects/DeepMIL-master/list/Matlab_formate/'
    mat_name_list = os.listdir(temporal_root)

    features = np.load(file.strip('\n'), allow_pickle=True)
    # features = [t.cpu().detach().numpy() for t in features]
    features = np.array(features, dtype=np.float32)
    num_frame = features.shape[0] * 16

    split_file = file.split('/')[-1].split('_')[0]
    mat_prefix = '_x264.mat'
    mat_file = split_file + mat_prefix
    print(mat_file)
 
    count = 0
    if 'Normal_' in file: # if it's normal
        # print('hello')
        for i in range(0, num_frame):
            gt.append(0.0)
            count+=1

    else: #if it's abnormal # get the name from temporal file
        if mat_file in mat_name_list:
            second_event = False
            annots = loadmat(os.path.join(temporal_root, mat_file))
            annots_idx = annots['Annotation_file']['Anno'].tolist()

            start_idx = annots_idx[0][0][0][0]
            end_idx = annots_idx[0][0][0][1]

            if len(annots_idx[0][0]) == 2:
                second_event = True
                
            # check if there's second events
            if not second_event:
                for i in range(0, start_idx):
                    gt.append(0.0)
                    count +=1
                if not (end_idx + 1) > num_frame:
                    for i in range(start_idx, end_idx + 1):
                        gt.append(1.0)
                        count += 1
                    for i in range(end_idx+1, num_frame):
                        gt.append(0.0)
                        count += 1
                else:
                    for i in range(start_idx, end_idx):
                        gt.append(1.0)
                        count += 1


            else:
                start_idx_2 = annots_idx[0][0][1][0]
                end_idx_2 = annots_idx[0][0][1][1]
                for i in range(0, start_idx):
                    gt.append(0.0)
                    count += 1
                for i in range(start_idx, end_idx + 1):
                    gt.append(1.0)
                    count += 1
                for i in range(end_idx+1, start_idx_2):
                    gt.append(0.0)
                    count += 1
                if not (end_idx_2 + 1) > num_frame:
                    for i in range(start_idx_2, end_idx_2 + 1):
                        gt.append(1.0)
                        count += 1
                    for i in range(end_idx_2 + 1, num_frame):
                        gt.append(0.0)
                        count += 1
                else:
                    for i in range(start_idx_2, end_idx_2):
                        gt.append(1.0)
                        count += 1

                if count != num_frame:
                    print(annots_idx)
                    print(num_frame)
                    print(count)
                    print(end_idx_2 +1)
    
    return gt


def test_video(dataloader, model, args, device, name):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)

        for i, input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            print(score_abnormal, score_normal)
            sig = logits
            pred = torch.cat((pred, sig))

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        # viz.lines(name, pred)       
            
    return pred

# viz = Visualizer(env='vietth5', use_incoming_socket=False)

if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)

    infer_loader = DataLoader(Infer(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    print('Test loader is done ..')

    # -- my model --
    model = Model(args.feature_size, args.batch_size)
    model.load_state_dict(torch.load('ckpt/rtfm-NAdam-i3d_v2.pkl'))
    print('Model loaded ..')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    pred = test_video(infer_loader, model, args, device, 'my model')
    np.save(args.save_scores, pred)
    print('Infer done my model ..')

