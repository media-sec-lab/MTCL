import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import time
from branch_decouple import SegFormer, SegFormer_Restore_returnDecoderFeatures

from utils import create_dir, argparse, create_file_list, compute_metrics
from dataset import Tampering_Dataset
from loss import SoftDiceLoss
torch.set_num_threads(6)

args = argparse()
'''设置随机数'''
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

args.mode = 'test'

final_result = []
args.root_path = "results"
restore_paths = ["ckpts/final_model.pth"]
is_restorations = [True]
# 根据自己的数据集路径进行设置
test_image_path = "/data1/zhuangpy/datasets/Testing/OSN_dataset/ImageForgeriesOSN_Dataset/CASIA/"
test_mask_path = "/data1/zhuangpy/datasets/Testing/OSN_dataset/ImageForgeriesOSN_Dataset/CASIA_GT/"
test_dataset = "CASIAv1"

for restore_path, is_restoration in zip(restore_paths, is_restorations):
    args.test_dataset = test_dataset
    args.test_image_path = test_image_path
    args.test_mask_path = test_mask_path

    args.test_images_split = ""
    args.patch_size = 512
    args.resize_factor = 0.5
    args.is_shuffle = True
    args.num_workers = 0

    args.restore_path = restore_path
    args.is_restoration = is_restoration
    args.save_result_path = os.path.join(args.root_path, args.test_dataset)

    args.is_shuffle_aug = True
    create_dir(os.path.join(args.save_result_path,"figs"))

    if args.mode=='test':
        val_file = create_file_list(args.test_image_path,args.test_mask_path,args.test_images_split)

        val_dataset = Tampering_Dataset(val_file, choice='test', patch_size=args.patch_size,resize_factor=args.resize_factor,is_shuffle=args.is_shuffle,is_shuffle_aug=args.is_shuffle_aug,args=args)

        valid_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,pin_memory=False,num_workers=args.num_workers,drop_last=False)

        if args.is_restoration==False:
            model = SegFormer()
        else:
            model = SegFormer_Restore_returnDecoderFeatures()

        if(args.restore_path!=""):
            model.load_state_dict(torch.load(args.restore_path))
        model = nn.DataParallel(model).cuda()

        criterion_loc_bce = torch.nn.BCELoss() # 篡改定位BCE loss
        criterion_loc_dice = SoftDiceLoss().cuda() # 篡改定位 Dice loss
        criterion_l1 = torch.nn.L1Loss() # 复原损失 L1 loss
        criterion_triplet_loss = torch.nn.TripletMarginLoss() # 特征度量损失 #


        f = open(os.path.join(args.save_result_path, "log.log"), 'w+')
        for k in args.__dict__:
            print(k + ": " + str(args.__dict__[k]))
            print(k + ": " + str(args.__dict__[k]), file = f)
        timestr = time.strftime('%Y%m%d_%H')


        model.eval()
        val_f1_ori, val_iou_ori, val_auc_ori = [], [], []
        val_f1_deg, val_iou_deg, val_auc_deg = [], [], []
        val_f1_restoraion, val_iou_restoraion, val_auc_restoraion = [], [], []
        total_time = []
        with torch.no_grad():
            for val_index, (img_ori,mask_ori,img_deg,mask_deg,image_name) in enumerate(valid_dataloader):
                print(img_ori.shape,mask_ori.shape)
                img_ori = img_ori.cuda()
                mask_ori = mask_ori.cuda()
                img_deg = img_deg.cuda()
                mask_deg = mask_deg.cuda()

                if args.is_restoration == True:
                    
                    '''return decoder features'''
                    t1 = time.time()
                    outputs_ori, restoraion, encoder_features_ori,decoder_features_ori = model(img_ori)  # 低质的训练和高质的一起训练
                    t2 = time.time()
                    total_time.append(t2-t1)
                    outputs_deg, restoration_deg, encoder_features_deg,decoder_features_deg = model(img_deg)  # 低质的图像才会返回复原的结果
                    outputs_restoration,_,_,_ = model((restoraion+img_ori))
                else:
                    '''return decoder features'''
                    t1 = time.time()
                    outputs_ori, encoder_features_ori,decoder_features_ori = model(img_ori)
                    t2 = time.time()
                    total_time.append(t2-t1)
                    outputs_deg, encoder_features_deg,decoder_features_deg = model(img_deg)

                    restoraion = torch.zeros_like(img_deg).cuda()
                    outputs_restoration = outputs_deg
                val_f1_ori, val_iou_ori, val_auc_ori = compute_metrics(img_ori, mask_ori, outputs_ori,
                                                                                f1_all=val_f1_ori,
                                                                                iou_all=val_iou_ori,
                                                                                auc_all=val_auc_ori,image_name = image_name[0], save_path=os.path.join(args.save_result_path,'figs'))
                
                print(
                    "Test index {} Image name {} "
                    "Ori F1 = {} IOU = {} AUC = {} ".format(str(val_index), image_name[0],
                                                            val_f1_ori[-1], val_iou_ori[-1],
                                                            val_auc_ori[-1]
                                                            ))
                print(
                    "Test index {} Image name {} "
                    "Ori F1 = {} IOU = {} AUC = {} ".format(str(val_index), image_name[0],
                                                            val_f1_ori[-1], val_iou_ori[-1],
                                                            val_auc_ori[-1]
                                                            ), file =f )

        print(
            "Test phase Dataset {}"
            "Ori F1 = {} IOU = {} AUC = {} ".format(args.test_dataset,
                np.mean(val_f1_ori),np.mean(val_iou_ori), np.mean(val_auc_ori)
            ))
        print("Mean test time (Remove first images): {}".format(np.mean(total_time[1:])))

        print(
            "Test phase Dataset {}"
            "Ori F1 = {} IOU = {} AUC = {} ".format(args.test_dataset,
                np.mean(val_f1_ori),np.mean(val_iou_ori), np.mean(val_auc_ori)
            ),file=f)
        print("Mean test time (Remove first images): {}".format(np.mean(total_time[1:])),file=f)
        f.close()
        final_result.append([args.test_dataset, np.mean(val_f1_ori), np.mean(val_auc_ori)])


    with open(os.path.join(args.root_path, 'log2.log'), 'w+') as f:
        pd_result = {}
        for dataset, f1, auc in final_result:
            pd_result[dataset+"_F1"] = [round(f1, 3)]
            pd_result[dataset + "_AUC"] = [round(auc, 3)]
            print("Dataset {} F1 = {} AUC = {}".format(dataset, f1, auc))
            print("Dataset {} F1 = {} AUC = {}".format(dataset, f1, auc), file = f)
    pd_result = pd.DataFrame(pd_result)
    pd_result.to_csv(os.path.join(args.root_path, 'results2.csv'), index=False)

