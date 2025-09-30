import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5,0'
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import time

from branch_decouple import SegFormer,SegFormer_Restore_returnDecoderFeatures
from utils import create_file_list, create_dir, argparse, get_logger, compute_metrics
from contrastive_loss_strategy import constrative_loss
from dataset import Tampering_Dataset
from loss import SoftDiceLoss
from utils import create_file_list_from_coco
torch.set_num_threads(6)

args = argparse()
'''设置随机数'''
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

args.mode = 'train'
args.dataset_root_path = "/data1/zhuangpy/" #数据集根目录，需要根据自己的目录来指定
if args.mode=='train':
    args.contrastive_encoder = True # 对比学习作用的是否是编码器的特征层
    args.with_degraded = True #有没有使用降质数据增强
    args.epochs, args.learning_rate, args.patience = [30, 0.0001, 2]
    args.batch_size = 8
    args.patch_size = 512
    args.max_iter = 1e4 * 10
    args.weight_decay = 0.0
    args.resize_factor = 0.5
    args.is_shuffle = True
    args.with_aug = True
    # args.sample_way 就是用的是不是我们提出的对比特征anchor采样策略， ‘prob'就是我们提出的策略，’naive‘就是普通的策略
    args.anchor_threshold,args.is_clustering,args.sample_way = 0.5,False,'prob'

    '''Ablation Study For SegFormer'''
    args.is_shuffle_aug = True # 是否使用数据增强模块
    args.is_triplet_loss = True # 是否使用对比学习的损失函数
    args.is_restoration = True # 是否有复原分支，如果没有的话，就是一个普通的SegFormer
    args.train_restoration = False # 是否会把复原的图像送进网络再进行训练
    args.contrastive_encoder_stage = 2 # 第几阶段进行对比学习
    args.weight_loc = 1 # lambda4 1
    args.train_loc = True
    args.weight_restoration = 1 # lambda2 1
    args.triplet_weight = 0.02 # lambda3 0.02
    args.loc_restoration_weight = 1 # lambda5 1
    args.lambda1 = 0.2 # 定位损失中，cross entropy的损失权重 0.2
    args.num_workers = 8

    args.dataset = 'CASIA2'
    args.train_images_split = ""
    args.test_images_split = ""
    args.train_image_path = args.dataset_root_path + "datasets/CASIA2/Tp_new/"
    args.train_mask_path = args.dataset_root_path + "datasets/CASIA2/mask_new/"
    args.train_image_path_imd2020 = args.dataset_root_path + "datasets/Testing/IMD2020/train/tamper_png/"
    args.train_mask_path_imd2020 = args.dataset_root_path + "datasets/Testing/IMD2020/train/mask_png/"
    args.train_images_split_imd2020 = ""
    args.tamper_coco_path = args.dataset_root_path + "/UNZIP_CAT/tampCOCO/"
    args.tamper_coco_files = ["bcmc_COCO_list.txt", "bcm_COCO_list.txt",
             "cm_COCO_list.txt", "sp_COCO_list.txt"]
    args.test_image_path = args.dataset_root_path + "datasets/Testing/IMD2020/test/tamper_png/"
    args.test_mask_path = args.dataset_root_path + "datasets/Testing/IMD2020/test/mask_png/"




    args.pretrained_path = "ckpts/mit_b5.pth"
    args.restore_path = ""

    args.save_model_path = os.path.join("ckpts/DifferentWeights/", "lambda1_{}_lambda2_{}_lambda3_{}_lambda4_{}_lambda5{}".format(str(args.lambda1),str(args.weight_restoration),
                                                                                                    str(args.triplet_weight),str(args.weight_loc),
                                                                                                str(args.loc_restoration_weight)))
    args.display_step = 20
    create_dir(args.save_model_path)



def train_restoration(model,restoration,criterion_loc_bce,criterion_loc_dice,label, args):
    outputs,_,_,_ = model(restoration)
    loss = args.lambda1 * criterion_loc_bce(outputs.view(outputs.size(0), -1),label.view(label.size(0), -1)) + (1-args.lambda1) * criterion_loc_dice(outputs.view(outputs.size(0), -1), label.view(label.size(0), -1))
    return loss




if args.mode=='train':
    train_file_casia2 = create_file_list(args.train_image_path,args.train_mask_path,args.train_images_split)
    train_file_imd2020 = create_file_list(args.train_image_path_imd2020,args.train_mask_path_imd2020,args.train_images_split_imd2020)
    train_file_tampercoco = create_file_list_from_coco(args.tamper_coco_path, args.tamper_coco_files)
    train_file = []
    for file in train_file_casia2:
        train_file.append(file)
    for file in train_file_imd2020:
        train_file.append(file)
    for file in train_file_tampercoco:
        train_file.append(file)
    random.shuffle(train_file)
    val_file = create_file_list(args.test_image_path,args.test_mask_path,args.test_images_split)

    train_dataset = Tampering_Dataset(train_file, choice='train', patch_size=args.patch_size,resize_factor=args.resize_factor,is_shuffle=args.is_shuffle,is_shuffle_aug=args.is_shuffle_aug,args=args)
    val_dataset = Tampering_Dataset(val_file, choice='val', patch_size=args.patch_size,resize_factor=args.resize_factor,is_shuffle=args.is_shuffle,is_shuffle_aug=args.is_shuffle_aug, args=args)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,pin_memory=False,num_workers=args.num_workers)
    if(args.patch_size!=None):
        valid_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=args.num_workers,drop_last=True)
    else:
        valid_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,pin_memory=False,num_workers=args.num_workers,drop_last=True)

    if args.is_restoration==False:
        model = SegFormer()
    else:
        model = SegFormer_Restore_returnDecoderFeatures()

    if args.pretrained_path != "":
        pretrained_dict = torch.load(args.pretrained_path)
        model_dict = {}
        state_dict = model.state_dict()
        # print("Model:")
        # for k,v in state_dict.items():
        # 	print(k)
        # print("Model pretrained:")
        for k, v in pretrained_dict.items():
            # print(k)
            # k = 'encoder.' + k
            if k in state_dict:
                model_dict[k] = v
        for key, value in model_dict.items():
            print(key)
        # print(model_dict)
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)

    if(args.restore_path!=""):
        model.load_state_dict(torch.load(args.restore_path))
    model = nn.DataParallel(model).cuda()

    criterion_loc_bce = torch.nn.BCELoss() # 篡改定位BCE loss
    criterion_loc_dice = SoftDiceLoss().cuda() # 篡改定位 Dice loss
    criterion_l1 = torch.nn.L1Loss() # 复原损失 L1 loss
    criterion_triplet_loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) # Optimizer

    lr_schdular = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.99)
    logger = get_logger(os.path.join(args.save_model_path,"log_2.log"))
    logger.info("CASIA_dataset {}, IMD2020 dataset {}, TamperCOCO dataset {}".format(len(train_file_casia2),
                                                                                      len(train_file_imd2020), len(train_file_tampercoco)))
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))
    # logger.info(model)
    timestr = time.strftime('%Y%m%d_%H')
    best_loss,best_f1, best_auc = 999999,0,0
    total_iter = 0
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]['lr']
        model.train()
        train_epoch_loss,train_epoch_loc_loss,train_epoch_restore_loss,train_epoch_triplet_loss,train_epoch_restoration_loc_loss = [], [], [],[],[]
        train_f1_ori,train_iou_ori,train_auc_ori = [],[],[]
        train_f1_deg, train_iou_deg, train_auc_deg = [], [], []
        for idx, (img_ori,mask_ori,img_deg,mask_deg,image_name) in enumerate(train_dataloader, 0):
            img_ori = img_ori.cuda()
            mask_ori = mask_ori.cuda()
            img_deg = img_deg.cuda()
            mask_deg = mask_deg.cuda()
            optimizer.zero_grad()
            total_iter += 1
            if args.is_restoration == True:
                outputs_ori, restoration_ori, encoder_features_ori,decoder_features_ori = model(img_ori)  # 低质的训练和高质的一起训练
                outputs_deg, restoraion, encoder_features_deg,decoder_features_deg = model(img_deg)  # 低质的图像才会返回复原的结果
            else:
                outputs_ori,encoder_features_ori,decoder_features_ori = model(img_ori)
                outputs_deg,encoder_features_deg,decoder_features_deg = model(img_deg)

            # 原始和降质的图像的损失 #
            loss_loc_ori = args.lambda1 * criterion_loc_bce(outputs_ori.view(outputs_ori.size(0), -1),mask_ori.view(mask_ori.size(0), -1)) + (1-args.lambda1) * criterion_loc_dice(outputs_ori.view(outputs_ori.size(0), -1), mask_ori.view(mask_ori.size(0), -1))
            loss_loc_deg = args.lambda1 * criterion_loc_bce(outputs_deg.view(outputs_deg.size(0), -1),mask_ori.view(mask_ori.size(0), -1)) + (1-args.lambda1) * criterion_loc_dice(outputs_deg.view(outputs_deg.size(0), -1), mask_ori.view(mask_ori.size(0), -1))
            loss_loc = loss_loc_ori + loss_loc_deg

            # 是否有复原的分支 #
            if args.is_restoration == True:
                loss_restoration = criterion_l1(restoraion.contiguous().view(restoraion.size(0),-1),(img_ori - img_deg).contiguous().view(restoraion.size(0),-1)) # restoration 的是残差图像
            else:
                loss_restoration = torch.tensor(0.0).cuda() # 如果没有复原分支，那么loss_restoration 为0

            # 是否使用Triplet loss #
            if args.is_triplet_loss == True:
                if (args.contrastive_encoder==True):
                    # encoder_features 的不同index对应不同stage的特征【0->stage#1, 1->stage#2, 2->stage#3, 3->stage#4】

                    outputs_16_16_ori = torch.clip(torch.nn.functional.interpolate(outputs_ori, size=(encoder_features_ori[args.contrastive_encoder_stage].shape[2],
                                                                                                      encoder_features_ori[args.contrastive_encoder_stage].shape[3]),mode='bilinear', align_corners=True), 0, 1)
                    outputs_16_16_deg = torch.clip(torch.nn.functional.interpolate(outputs_deg, size=(encoder_features_deg[args.contrastive_encoder_stage].shape[2],
                                                                                                      encoder_features_deg[args.contrastive_encoder_stage].shape[3]), mode='bilinear',align_corners=True), 0, 1)
                    GT_16_16_ori = torch.nn.functional.interpolate(mask_ori,size=(encoder_features_ori[args.contrastive_encoder_stage].shape[2],
                                                                                  encoder_features_ori[args.contrastive_encoder_stage].shape[3]),mode='nearest')
                    GT_16_16_deg = torch.nn.functional.interpolate(mask_deg,size=(encoder_features_deg[args.contrastive_encoder_stage].shape[2],
                                                                                  encoder_features_deg[args.contrastive_encoder_stage].shape[3]),mode='nearest')

                    triplet_loss_ori = constrative_loss(outputs_16_16_ori,GT_16_16_ori,encoder_features_ori[args.contrastive_encoder_stage],args.anchor_threshold,args.is_clustering,args.sample_way,criterion_triplet_loss) # Ori 图像的 Triplet loss
                    triplet_loss_deg = constrative_loss(outputs_16_16_deg,GT_16_16_deg,encoder_features_deg[args.contrastive_encoder_stage],args.anchor_threshold,args.is_clustering,args.sample_way,criterion_triplet_loss) # deg 图像的 Triplet loss

                    triplet_loss = triplet_loss_ori + triplet_loss_deg
                else:
                    '''another triplet loss'''
                    outputs_4_4_ori = torch.clip(torch.nn.functional.interpolate(outputs_ori, size=(decoder_features_ori.shape[2], decoder_features_ori.shape[3]), mode='bilinear',align_corners=True), 0, 1)
                    outputs_4_4_deg = torch.clip(torch.nn.functional.interpolate(outputs_deg, size=(decoder_features_deg[-2].shape[2], decoder_features_deg.shape[3]), mode='bilinear',align_corners=True), 0, 1)
                    GT_4_4_ori = torch.nn.functional.interpolate(mask_ori, size=(decoder_features_ori.shape[2], decoder_features_ori.shape[3]), mode='nearest')
                    GT_4_4_deg = torch.nn.functional.interpolate(mask_deg, size=(decoder_features_deg.shape[2], decoder_features_deg.shape[3]), mode='nearest')
                    triplet_loss_ori = constrative_loss(outputs_4_4_ori, GT_4_4_ori,
                                                                               decoder_features_ori, args.anchor_threshold,
                                                                               args.is_clustering, args.sample_way,
                                                                               criterion_triplet_loss)  # Ori 图像的 Triplet loss
                    triplet_loss_deg = constrative_loss(outputs_4_4_deg, GT_4_4_deg,
                                                                               decoder_features_deg, args.anchor_threshold,
                                                                               args.is_clustering, args.sample_way,
                                                                               criterion_triplet_loss)  # deg 图像的 Triplet loss

                    triplet_loss = triplet_loss_ori + triplet_loss_deg
            else:
                triplet_loss = torch.tensor(0.0).cuda()
            # 是否使用复原之后的图像再进行训练 #
            if args.train_restoration == True:
                restoration_ori = (restoration_ori + img_ori)
                loss_loc_restoration_ori = train_restoration(model, restoration_ori,criterion_loc_bce, criterion_loc_dice,mask_ori, args)
                restoraion_deg = (restoraion + img_deg)
                loss_loc_restoration_deg = train_restoration(model, restoraion_deg, criterion_loc_bce,criterion_loc_dice, mask_deg, args)
                loss_loc_restoration = loss_loc_restoration_ori + loss_loc_restoration_deg
            else:
                loss_loc_restoration = torch.tensor(0.0).cuda()

            loss_total = args.weight_loc * loss_loc + args.weight_restoration * loss_restoration + args.triplet_weight * triplet_loss + args.loc_restoration_weight * loss_loc_restoration

            loss_total.backward()
            optimizer.step()

            train_f1_ori, train_iou_ori, train_auc_ori = compute_metrics(img_ori,mask_ori,outputs_ori,f1_all=train_f1_ori,iou_all=train_iou_ori,auc_all=train_auc_ori)
            train_f1_deg, train_iou_deg, train_auc_deg = compute_metrics(img_deg,mask_deg,outputs_deg,f1_all=train_f1_deg,iou_all=train_iou_deg,auc_all=train_auc_deg)

            #梯度清零s

            # 阶段1定位损失，包括高质和低质图像

            train_epoch_loss.append(loss_total.item())
            train_epoch_loc_loss.append(loss_loc.item())
            train_epoch_restore_loss.append(loss_restoration.item())
            train_epoch_triplet_loss.append(triplet_loss.item())
            train_epoch_restoration_loc_loss.append(loss_loc_restoration.item())
            if total_iter % args.display_step == 0:
                logger.info("Image aug {} epoch={}/{},{}/{}of train {}, Learning rate={} Total loss = {} Loc loss = {} Restoraion loc loss = {} Restore los = {} Triplet loss = {} Ori: F1 = {} IOU = {} AUC = {}"
                            "Deg: F1 = {} IOU = {} AUC = {}".format(args.with_degraded,epoch, args.epochs, idx,
                               len(train_dataloader), total_iter, lr,np.mean(train_epoch_loss),np.mean(train_epoch_loc_loss),np.mean(train_epoch_restoration_loc_loss),np.mean(train_epoch_restore_loss),np.mean(train_epoch_triplet_loss),
                                                                    np.mean(train_f1_ori),np.mean(train_iou_ori),np.mean(train_auc_ori),
                                                                    np.mean(train_f1_deg),np.mean(train_iou_deg),np.mean(train_auc_deg)))

        if (True):
            model.eval()
            val_epoch_loss, val_epoch_loc_loss, val_epoch_restore_loss, val_epoch_triplet_loss,val_epoch_restoration_loc_loss = [], [], [], [],[]
            val_f1_ori, val_iou_ori, val_auc_ori = [], [], []
            val_f1_deg, val_iou_deg, val_auc_deg = [], [], []
            with torch.no_grad():
                for val_index, (img_ori,mask_ori,img_deg,mask_deg,image_name) in enumerate(valid_dataloader):
                    img_ori = img_ori.cuda()
                    mask_ori = mask_ori.cuda()
                    img_deg = img_deg.cuda()
                    mask_deg = mask_deg.cuda()

                    if args.is_restoration == True:
                        outputs_ori, restoraion_ori, encoder_features_ori, decoder_features_ori = model(img_ori)  # 低质的训练和高质的一起训练
                        outputs_deg, restoraion, encoder_features_deg, decoder_features_deg = model(img_deg)  # 低质的图像才会返回复原的结果
                    else:

                        outputs_ori, encoder_features_ori, decoder_features_ori = model(img_ori)
                        outputs_deg, encoder_features_deg, decoder_features_deg = model(img_deg)

                    # 原始和降质的图像的损失 #
                    loss_loc_ori = 0.2 * criterion_loc_bce(outputs_ori.view(outputs_ori.size(0), -1),
                                                           mask_ori.view(mask_ori.size(0),
                                                                         -1)) + 0.8 * criterion_loc_dice(
                        outputs_ori.view(outputs_ori.size(0), -1), mask_ori.view(mask_ori.size(0), -1))
                    loss_loc_deg = 0.2 * criterion_loc_bce(outputs_deg.view(outputs_deg.size(0), -1),
                                                           mask_ori.view(mask_ori.size(0),
                                                                         -1)) + 0.8 * criterion_loc_dice(
                        outputs_deg.view(outputs_deg.size(0), -1), mask_ori.view(mask_ori.size(0), -1))
                    loss_loc = loss_loc_ori + loss_loc_deg
                    # 是否有复原的分支 #
                    if args.is_restoration == True:
                        loss_restoration = criterion_l1(restoraion.contiguous().view(restoraion.size(0), -1),
                                                        (img_ori - img_deg).contiguous().view(restoraion.size(0),
                                                                                              -1))  # restoration 的是残差图像
                    else:
                        loss_restoration = torch.tensor(0.0).cuda()  # 如果没有复原分支，那么loss_restoration 为0

                    # 是否使用Triplet loss #
                    if args.is_triplet_loss == True:
                        if args.contrastive_encoder == True:
                            outputs_16_16_ori = torch.clip(torch.nn.functional.interpolate(outputs_ori, size=(
                            encoder_features_ori[-2].shape[2], encoder_features_ori[-2].shape[3]), mode='bilinear',
                                                                                           align_corners=True), 0, 1)
                            outputs_16_16_deg = torch.clip(torch.nn.functional.interpolate(outputs_deg, size=(
                            encoder_features_deg[-2].shape[2], encoder_features_deg[-2].shape[3]), mode='bilinear',
                                                                                           align_corners=True), 0, 1)
                            GT_16_16_ori = torch.nn.functional.interpolate(mask_ori, size=(
                            encoder_features_ori[-2].shape[2], encoder_features_ori[-2].shape[3]), mode='nearest')
                            GT_16_16_deg = torch.nn.functional.interpolate(mask_deg, size=(
                            encoder_features_deg[-2].shape[2], encoder_features_deg[-2].shape[3]), mode='nearest')

                            triplet_loss_ori = constrative_loss(outputs_16_16_ori, GT_16_16_ori,
                                                                   encoder_features_ori[-2], args.anchor_threshold,
                                                                   args.is_clustering, args.sample_way,
                                                                   criterion_triplet_loss)  # Ori 图像的 Triplet loss
                            triplet_loss_deg = constrative_loss(outputs_16_16_deg, GT_16_16_deg,
                                                                   encoder_features_deg[-2], args.anchor_threshold,
                                                                   args.is_clustering, args.sample_way,
                                                                   criterion_triplet_loss)  # deg 图像的 Triplet loss

                            triplet_loss = triplet_loss_ori + triplet_loss_deg
                        else:
                            '''another triplet loss'''
                            outputs_4_4_ori = torch.clip(torch.nn.functional.interpolate(outputs_ori, size=(
                            decoder_features_ori.shape[2], decoder_features_ori.shape[3]), mode='bilinear',
                                                                                         align_corners=True), 0, 1)
                            outputs_4_4_deg = torch.clip(torch.nn.functional.interpolate(outputs_deg, size=(
                            decoder_features_deg[-2].shape[2], decoder_features_deg.shape[3]), mode='bilinear',
                                                                                         align_corners=True), 0, 1)
                            GT_4_4_ori = torch.nn.functional.interpolate(mask_ori, size=(
                            decoder_features_ori.shape[2], decoder_features_ori.shape[3]), mode='nearest')
                            GT_4_4_deg = torch.nn.functional.interpolate(mask_deg, size=(
                            decoder_features_deg.shape[2], decoder_features_deg.shape[3]), mode='nearest')

                            triplet_loss_ori = constrative_loss(outputs_4_4_ori, GT_4_4_ori,
                                                                                       decoder_features_ori,
                                                                                       args.anchor_threshold,
                                                                                       args.is_clustering, args.sample_way,
                                                                                       criterion_triplet_loss)  # Ori 图像的 Triplet loss
                            triplet_loss_deg = constrative_loss(outputs_4_4_deg, GT_4_4_deg,
                                                                                       decoder_features_deg,
                                                                                       args.anchor_threshold,
                                                                                       args.is_clustering, args.sample_way,
                                                                                       criterion_triplet_loss)  # deg 图像的 Triplet loss
                            triplet_loss = triplet_loss_ori + triplet_loss_deg
                    else:
                        triplet_loss = torch.tensor(0.0).cuda()
                    if args.train_restoration == True:
                        restoraion_ori = (restoraion_ori + img_ori)
                        loss_loc_restoration_ori = train_restoration(model, restoraion_ori, criterion_loc_bce,
                                                                     criterion_loc_dice, mask_ori, args)
                        restoraion_deg = (restoraion + img_deg)
                        loss_loc_restoration_deg = train_restoration(model, restoraion_deg, criterion_loc_bce,
                                                                     criterion_loc_dice, mask_deg, args)
                        loss_loc_restoration = loss_loc_restoration_ori + loss_loc_restoration_deg
                    else:
                        loss_loc_restoration = torch.tensor(0.0).cuda()
                    # print(outputs.shape,mask_ori.shape,img_ori.shape)
                    loss_total = args.weight_loc * loss_loc + args.weight_restoration * loss_restoration + args.triplet_weight * triplet_loss+ args.loc_restoration_weight * loss_loc_restoration

                    val_f1_ori, val_iou_ori, val_auc_ori = compute_metrics(img_ori, mask_ori, outputs_ori,
                                                                                 f1_all=val_f1_ori,
                                                                                 iou_all=val_iou_ori,
                                                                                 auc_all=val_auc_ori)
                    val_f1_deg, val_iou_deg, val_auc_deg = compute_metrics(img_deg, mask_deg, outputs_deg,
                                                                                 f1_all=val_f1_deg,
                                                                                 iou_all=val_iou_deg,
                                                                                 auc_all=val_auc_deg)

                    # 梯度清零s

                    # 阶段1定位损失，包括高质和低质图像
                    val_epoch_loss.append(loss_total.item())
                    val_epoch_loc_loss.append(loss_loc.item())
                    val_epoch_restore_loss.append(loss_restoration.item())
                    val_epoch_triplet_loss.append(triplet_loss.item())
                    val_epoch_restoration_loc_loss.append(loss_loc_restoration.item())
            if np.mean(val_epoch_loc_loss) < best_loss:
                best_loss = np.mean(val_epoch_loc_loss)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path,
                                                                   "Epoch_{}_Loss_{}_Ori_F1_{}_AUC_{}_Deg_F1_{}_AUC_{}.pth".format(
                                                                       epoch,
                                                                       round(np.mean(val_epoch_loc_loss), 4),
                                                                       round(np.mean(val_f1_ori), 4),
                                                                       round(np.mean(val_auc_ori), 4),round(np.mean(val_f1_deg), 4),
                                                                       round(np.mean(val_auc_deg), 4))))

            if np.mean(val_f1_ori) > best_f1:
                best_f1 = np.mean(val_f1_ori)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path,
                                                                   "Epoch_{}_Loss_{}_Ori_F1_{}_AUC_{}_Deg_F1_{}_AUC_{}.pth".format(
                                                                       epoch,
                                                                       round(np.mean(val_epoch_loc_loss), 4),
                                                                       round(np.mean(val_f1_ori), 4),
                                                                       round(np.mean(val_auc_ori), 4),
                                                                       round(np.mean(val_f1_deg), 4),
                                                                       round(np.mean(val_auc_deg), 4))))
            if np.mean(val_auc_ori) > best_auc:
                best_auc = np.mean(val_auc_ori)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path,
                                                                   "Epoch_{}_Loss_{}_Ori_F1_{}_AUC_{}_Deg_F1_{}_AUC_{}.pth".format(
                                                                       epoch,
                                                                       round(np.mean(val_epoch_loc_loss), 4),
                                                                       round(np.mean(val_f1_ori), 4),
                                                                       round(np.mean(val_auc_ori), 4),
                                                                       round(np.mean(val_f1_deg), 4),
                                                                       round(np.mean(val_auc_deg), 4))))
            logger.info(
                "Model Type: Shuffle_aug_{}_Restoration_{}_Triplet_{} Validation {}, Learning rate = {} "
                "Total loss = {} Loc loss = {} Restoration loss = {} Triplet loss = {} "
                "Ori F1 = {} IOU = {} AUC = {} "
                "Deg F1 = {} IOU = {} AUC = {}"
                "Best loss = {} Best f1 = {} Best auc = {} ".format(str(args.is_shuffle_aug),str(args.is_restoration),str(args.is_triplet_loss),epoch, lr,
                    np.mean(val_epoch_loss), np.mean(val_epoch_loc_loss),np.mean(val_epoch_restore_loss),np.mean(val_epoch_triplet_loss),
                    np.mean(val_f1_ori),np.mean(val_iou_ori), np.mean(val_auc_ori),
                    np.mean(val_f1_deg),np.mean(val_iou_deg), np.mean(val_auc_deg),
                    best_loss,best_f1, best_auc
                ))

        lr_schdular.step()


