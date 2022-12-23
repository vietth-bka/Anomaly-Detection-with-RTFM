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
from process_videos import gen_video

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
    np.save('./scores/scores.npy', pred)
    print('Infer done my model. Processing video ..')
    gen_video(input=args.input_video, scores=pred, out=args.output_video)
    print('Video is saved at', args.output_video)

