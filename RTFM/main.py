from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from model import Model
from dataset import Dataset
from train import train
from test_10crop import test
import option
from tqdm import tqdm
# from utils import Visualizer
from config import *

viz = None #Visualizer(env='shanghai tech 10 crop', use_incoming_socket=False)

if __name__ == '__main__':
    args = option.parser.parse_args()
    # config = Config(args)

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    print('train_nloader:', len(train_nloader), 'train_aloader:', len(train_aloader), 'test_loader:', len(test_loader))

    model = Model(args.feature_size, args.batch_size)
    print('Training from scratch ..')

    # model.load_state_dict(torch.load('./ucf-i3d-ckpt.pkl'))
    # print('Loaded pretrained ..')
    
    # for name, value in model.named_parameters():
    #     print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.005)
    elif args.optim == 'NAdam':
        optimizer = optim.NAdam(model.parameters(), lr=args.lr, weight_decay=0.005)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.005)
    elif args.optim == 'RMS':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=0.005)
    

    print('Optimizer:', args.optim)
    scheduler = None
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch, args.lr/100)
    # scheduler =torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000, 10000], gamma=0.5)
    print('Lr scheduler:', scheduler)

    test_info = {"epoch": [], "roc_AUC":[], "max_ap": [], 'recall':[], 'prec':[], 'th':[], "cost": []}
    best_AP = -1
    output_path = ''   # put your own path here
    auc, ap, metrics = test(test_loader, model, args, viz, device)
    print("AUC roc=%2.5f, ap=%2.5f, rc=%2.5f" % (auc, ap, metrics[0]))

    step_iter = tqdm(
                range(1, args.max_epoch + 1),
                desc="Training... (loss=X.X, lr=X.X)",
                bar_format="{l_bar}{r_bar}",
                total=args.max_epoch,
                dynamic_ncols=True)

    for step in step_iter:
        # if step <= 25:
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = step/25 * config.lr

        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        cost = train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, scheduler, viz, device)
        step_iter.set_description("Scratch s6 with %s... (loss=%2.5f, lr=%2.7f)" % (args.optim, cost, optimizer.param_groups[0]["lr"]))

        if step % 10 == 0 and step > 0:
            print('\nTesting at step %d ..'%step)
            roc_auc, max_ap, (rec, prec, th) = test(test_loader, model, args, viz, device)
            print("Results: roc=%2.5f, ap=%2.5f, rc=%2.5f, pr=%2.5f, th=%2.5f" % (roc_auc, max_ap, rec, prec, th))
            test_info["epoch"].append(step)
            test_info["roc_AUC"].append(roc_auc)
            test_info["max_ap"].append(max_ap)
            test_info["recall"].append(rec)
            test_info["prec"].append(prec)
            test_info["th"].append(th)
            test_info["cost"].append(cost)

            if max_ap > best_AP and roc_auc > 0.0 and rec > 0.7 and th > 0.6:
                best_AP = max_ap
                # print('Best AP:', best_AP, ', Recall:', rec, ', Th:', th)
                name = './ckpt/' + args.model_name + '-' + args.optim + '-i3d_s6.pkl'
                torch.save(model.state_dict(), name)
                print('==> Saved as', name)
                save_best_record(test_info, os.path.join('record_v2', f'AUC_{args.optim}_s6.txt'))
            else:
                print('Not saving as roc_auc < 0.0 or rec < 0.8 or not the best')
    torch.save(model.state_dict(), './ckpt/' + args.model_name  + '-' + args.optim + 'final_s6.pkl')

