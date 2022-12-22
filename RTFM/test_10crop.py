import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np

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
            gt = np.load('list/gt_ucf_v2.npy')

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        np.save('fpr.npy', fpr)
        np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)
        # print('auc : ' + str(rec_auc))

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)        
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)
        # print(len(th))
        # print('pr_auc', pr_auc)
        # viz.plot_lines('pr_auc', pr_auc)
        # viz.plot_lines('auc', rec_auc)
        # viz.lines('scores', pred)
        # viz.lines('roc', tpr, fpr)
        j = np.argmax(precision * recall)
        max_ap = precision[j] * recall[j]

        return rec_auc, max_ap, (recall[j], precision[j], th[j])
    
def test_ckpt(dataloader, model, args, viz, device):
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
            gt = np.load('list/gt_ucf_v2.npy')

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        np.save('fpr.npy', fpr)
        np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)
        # print('auc : ' + str(rec_auc))

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)        
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)
        # print(len(th))
        # print('pr_auc', pr_auc)
        # viz.plot_lines('pr_auc', pr_auc)
        # viz.plot_lines('auc', rec_auc)
        # viz.lines('scores', pred)
        # viz.lines('roc', tpr, fpr)
        max_ap = np.max(precision * recall)

        return rec_auc, max_ap, (fpr, tpr, threshold, precision, recall, th)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from model import Model
    from dataset import Dataset
    import option
    from matplotlib import pyplot
    import os

    args = option.parser.parse_args()

    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    print('test_loader:', len(test_loader))

    model = Model(args.feature_size, args.batch_size)
    model = model.cuda()
    model_pth = './ckpt/rtfm-Adam-i3d_s6.pkl'
    model.load_state_dict(torch.load(model_pth))
    print('Loaded model ..', os.path.basename(model_pth))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --testing--
    auc, pr, metrics = test_ckpt(test_loader, model, args, None, device)
    print("AUC roc=%2.5f, pr=%2.5f" % (auc, pr))
    fpr, tpr, th1, precision, recall, th2 = metrics
    pyplot.plot(recall, precision, marker='.', label='PR_curve')
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    pyplot.savefig('./pictures/PR_curve.png')
    pyplot.close()

    pyplot.plot(fpr, tpr, marker='.', label='ROC_curve')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.savefig('./pictures/ROC_curve.png')

    i = np.argmax((1-fpr) * tpr)
    print(fpr[i], tpr[i], th1[i])

    # j = np.argmin(np.abs(th1[i] - th2))
    j = np.argmax(precision * recall)
    print(precision[j], recall[j], th2[j], 'ap:', precision[j]* recall[j])
