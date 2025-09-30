import os
import pickle
import random
from sklearn.metrics import roc_auc_score
from metrics import get_metrics
from skimage import io
import numpy as np
import logging

import subprocess

import matplotlib.pyplot as plt
import cv2
random.seed(44)

def get_available_gpus():
    # Run the nvidia-smi command to get GPU information
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free,index', '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE, encoding='utf-8')

    # Parse the output
    lines = result.stdout.strip().split('\n')
    gpus = []

    for line in lines:
        free_memory, index = line.split(', ')
        gpus.append((int(free_memory), int(index)))

    return gpus


def select_gpu_with_max_memory():
    gpus = get_available_gpus()
    if not gpus:
        raise RuntimeError("No GPUs found")

    # Sort GPUs by free memory in descending order and select the one with max memory
    gpus.sort(reverse=True, key=lambda x: x[0])
    max_memory_gpu = gpus[0]

    return max_memory_gpu[1]


def create_dir(path):
    if(os.path.exists(path)==False):
        os.makedirs(path)

class argparse():
    pass



def create_file_list(image_path,mask_path,image_file):
    if(image_file!=''):
        with open(image_file,'rb') as f:
            images = pickle.load(f)
        if ('train_images' in image_file):
            print("Training Mode")
            images = images
        else:
            print("Validation Mode")
            images = images
    else:
        images = os.listdir(image_path)
    files = []
    random.shuffle(images)
    for image in images:
        if(mask_path!=''):
            if ('IMD2020' in mask_path):
                mask_name = image.replace('.png', '_mask.png', 1)
                mask_name = mask_name.replace('.jpeg','_mask.png',1)
                mask_name = mask_name.replace('.jpg', '_mask.png', 1)
            elif ('DEFACTO' in mask_path):
                mask_name = image
            elif ('Mix3datasets' in mask_path):
                mask_name = image.replace('.jpg', '.png', 1)
                mask_name = mask_name.replace('ps', 'ms', 1)
            elif ('Mix_CASIA2Au_script_Dresden_script_CMFD_DEFACTO' in mask_path):
                mask_name = image.replace('.jpg', '.png', 1)
                mask_name = mask_name.replace('ps', 'ms', 1)
            elif ('DEFACTO' in mask_path):
                mask_name = image

            elif ('CASIA2' in mask_path):
                mask_name = image
            elif ('CASIA' in mask_path and "PostProcessing" not in mask_path):
                mask_name = image.replace('.jpg', '_gt.png')
                mask_name = mask_name.replace('.jpeg', '_gt.png')
            elif ('CASIA' in mask_path and "PostProcessing" in mask_path):
                mask_name = image.replace('.png', '_gt.png')
            elif ('IMD2020' in mask_path):
                mask_name = image.replace('.png', '_mask.png')
                mask_name = mask_name.replace('.jpg', '_mask.png')

            elif ('NIST' in mask_path):
                mask_name = image.replace('.jpg', '_gt.png')
                mask_name = mask_name.replace('.jpeg', '_gt.png')
            elif ('DSO' in mask_path):
                mask_name = image.replace('.png', '_gt.png')
                mask_name = mask_name.replace('.jpg', '_gt.png')
                mask_name = mask_name.replace('.jpeg', '_gt.png')
            elif ('Columbia' in mask_path):
                mask_name = image.replace('.tif', '_gt.png')
                mask_name = mask_name.replace('.jpg', '_gt.png')
                mask_name = mask_name.replace('.jpeg', '_gt.png')
            elif ('Coverage' in mask_path):
                mask_name = image
            elif ('Certificate' in mask_path):
                mask_name = image.replace('ps_', 'ms_')
                mask_name = mask_name.replace('rma_2', 'rma_3')
                mask_name = mask_name.replace('.jpg', '.png')
            elif ('IFC' in mask_path):
                mask_name = image.replace('ps', 'ms')
            elif ('Coverage' in mask_path):
                mask_name = image
            elif('SYSU' in mask_path):
                mask_name = image
            elif('IFC' in mask_path):
                mask_name = image.replace('ps','ms')
            elif('mix_500' in mask_path):
                mask_name = image.replace('ps','ms')
                mask_name = mask_name.replace('.jpg','.png')
            elif ('Natural_scene' in mask_path):
                mask_name = image.replace('ps','ms')
                mask_name = mask_name.replace('.jpg','.png')
            elif ("tampCOCO" in mask_path):
                mask_name = image.replace('.jpg', '.png')
                while '.jpg' in mask_name:
                    mask_name = mask_name.replace('.jpg', '.png')
            else:
                mask_name = image
            files.append([os.path.join(image_path, image), os.path.join(mask_path, mask_name)])
        else:
            files.append([os.path.join(image_path,image)])
    return files


def create_file_list_from_coco(root_dir, files):
    image_mask_files = []
    for file in files:
        with open(os.path.join(root_dir, file), 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            for line in lines:
                image_name, mask_name = line.split(',')
                image_mask_files.append([os.path.join(root_dir, image_name), os.path.join(root_dir, mask_name)])
    random.shuffle(image_mask_files)
    image_mask_files = image_mask_files[:10000]
    return image_mask_files
def compute_metrics(image_batch,label_batch,outputs_batch,f1_all,iou_all,auc_all,image_name = '', save_path=""):
    if save_path != "":
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)
    # #计算每一个batch里面每一张图的F1和AUC
    image_batch = image_batch.cpu().detach().numpy()
    label_batch = label_batch.cpu().detach().numpy()
    outputs_batch = outputs_batch.cpu().detach().numpy()
    # print(image_batch.shape,label_batch.shape,outputs_softmax.shape,outputs_threshold.shape)
    for image, label, predict_map in zip(image_batch, label_batch, outputs_batch):
        predict_map = predict_map[0,:,:]

        label = label[0,:,:]
        # predict_map = cv2.resize(predict_map,(label.shape[1],label.shape[0]))
        # print(np.unique(predict_map))
        # fig,ax = plt.subplots(1,2)
        # ax[0].imshow(predict_map)
        # ax[1].imshow(label)
        # plt.show()
        # print(np.unique(predict_map),np.unique(label))
        predict_threshold = np.copy(predict_map)
        predict_threshold[np.where(predict_map<0.5)] = 0
        predict_threshold[np.where(predict_map>=0.5)] = 1
        if(np.mean(label)!=0):
            try:
                # f1 = f1_score(label.reshape(-1, ), predict_threshold.reshape(-1, ), zero_division=0)
                tpr_recall, tnr, precision, f1, mcc, iou, tn, tp, fn, fp = get_metrics(predict_threshold, label)
                auc = roc_auc_score(label.reshape(-1, ), predict_map.reshape(-1, ))
                f1_all.append(f1)
                auc_all.append(auc)
                iou_all.append(iou)
            except Exception as e:
                print(e) #对于没有篡改像素的图像，直接跳过指标计算
                continue
        if save_path != "":
            image_name = image_name.replace('.jpg','.png')
            image_name = image_name.replace('.jpeg','.png')
            io.imsave(os.path.join(save_path, image_name),np.uint8(255*predict_map))
    return f1_all,iou_all,auc_all

def compute_metrics_save_results(image_batch,label_batch,outputs_batch,f1_all,iou_all,auc_all,is_restoration,restoration_outputs_batch,restorations,image_name = ""):
    # #计算每一个batch里面每一张图的F1和AUC
    image_batch = image_batch.cpu().detach().numpy()
    label_batch = label_batch.cpu().detach().numpy()
    outputs_batch = outputs_batch.cpu().detach().numpy()
    restoration_outputs_batch = restoration_outputs_batch.cpu().detach().numpy()
    restorations = restorations.cpu().detach().numpy()
    # print(image_batch.shape,label_batch.shape,outputs_softmax.shape,outputs_threshold.shape)
    for image, label, predict_map,restoration,restoration_output in zip(image_batch, label_batch, outputs_batch,restorations,restoration_outputs_batch):
        predict_map = predict_map[0,:,:]
        restoration_output = restoration_output[0,:,:]
        label = label[0,:,:]
        # print(np.unique(predict_map),np.unique(label))
        predict_threshold = np.copy(predict_map)
        predict_threshold[np.where(predict_map<0.5)] = 0
        predict_threshold[np.where(predict_map>=0.5)] = 1

        restoration_threshold = np.copy(restoration_output)
        restoration_threshold[np.where(restoration_output < 0.5)] = 0
        restoration_threshold[np.where(restoration_output >= 0.5)] = 1
        if(np.mean(label)!=0):
            try:
                # f1 = f1_score(label.reshape(-1, ), predict_threshold.reshape(-1, ), zero_division=0)
                tpr_recall, tnr, precision, f1, mcc, iou, tn, tp, fn, fp = get_metrics(predict_threshold, label)
                auc = roc_auc_score(label.reshape(-1, ), predict_map.reshape(-1, ))
                f1_all.append(f1)
                auc_all.append(auc)
                iou_all.append(iou)
            except Exception as e:
                print(e) #对于没有篡改像素的图像，直接跳过指标计算
                continue
        
    return f1_all,iou_all,auc_all


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w+")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

