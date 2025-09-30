import torch
from torch.utils.data import Dataset
import cv2
from shuffle_aug_v2 import shuffle_aug_online_tampering_flip_rotate, shuffle_aug_degradation
import random
import numpy as np


class Tampering_Dataset(Dataset):
    def __init__(self, file,choice='train',resize_factor = None,patch_size = None,is_shuffle=True,is_shuffle_aug=False, args = None):
        self.filelist = file
        self.choice = choice
        self.resize_factor = resize_factor
        # self.original_resolution = original_resolution
        self.patch_size = patch_size
        self.is_shuffle = is_shuffle
        self.is_shuffle_aug = is_shuffle_aug
        self.args = args
    # def __getitem__(self, idx):
    #     cv2.ocl.setUseOpenCL(False)  # 设置opencv不使用多进程运行，但这句命令只在本作用域有效。
    #     cv2.setNumThreads(0)
    #     return self.load_item(idx)

    def __len__(self):

        return len(self.filelist)

    def __getitem__(self, idx):
        # cv2.ocl.setUseOpenCL(False)
        # cv2.setNumThreads(0)
        # if self.choice != 'test':
        #     fname1, fname2 = self.filelist[idx]
        # else:
        #     fname1, fname2 = self.filelist[idx], ''
        if len(self.filelist[idx]) > 1:
            fname1, fname2 = self.filelist[idx]
        else:
            fname1, fname2 = self.filelist[idx][0], ''
        img = cv2.imread(fname1)
        H, W, _ = img.shape
        if fname2 == '':
            mask = np.zeros([H, W, 3])
        else:
            mask = cv2.imread(fname2)
            if(np.max(mask)<=1):
                mask = mask*255
        # fig,ax = plt.subplots(3,2)
        # ax[0][0].imshow(img)
        # ax[0][1].imshow(mask)
        if self.choice =='train':

            if self.is_shuffle_aug: # 加不加随机数据增强 #

                img2 = cv2.imread(self.filelist[random.randint(0, len(self.filelist) - 1)][0])
                img_ori,mask_ori = shuffle_aug_online_tampering_flip_rotate(img,img2,mask,self.is_shuffle) #Plain tampered and mask
                img_degradation,mask_degradation, resize_factor_W_deg,resize_factor_H_deg = shuffle_aug_degradation(img_ori,mask_ori,self.is_shuffle) # 降质处理
            else:
                img_ori,mask_ori = img,mask
                img_degradation, mask_degradation, resize_factor_W_deg, resize_factor_H_deg = img_ori,mask_ori,1,1
            if(self.patch_size!=None): # 是否缩放到某个分辨率进行训练 #
                img_ori = cv2.resize(img_ori,(self.args.patch_size,self.args.patch_size))
                mask_ori = cv2.resize(mask_ori,(self.args.patch_size,self.args.patch_size),interpolation=cv2.INTER_NEAREST)
                img_degradation = cv2.resize(img_degradation,(self.args.patch_size,self.args.patch_size))
                mask_degradation = cv2.resize(mask_degradation,(self.args.patch_size,self.args.patch_size),interpolation=cv2.INTER_NEAREST)
        # fig,ax = plt.subplots(2,2)
        # ax[0][0].imshow(img_ori)
        # ax[0][1].imshow(mask_ori)
        # ax[1][0].imshow(img_degradation)
        # ax[1][1].imshow(mask_degradation)
        # plt.show()

        if self.choice == 'val':
            img_ori, mask_ori = img, mask
            img_degradation, mask_degradation, resize_factor_W_deg, resize_factor_H_deg = shuffle_aug_degradation(img_ori, mask_ori, self.is_shuffle)  # 降质处理
            # result, encimg = cv2.imencode('.jpg', img_ori, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            # img_degradation = cv2.imdecode(encimg, 1)
            # mask_degradation = mask
            if (self.patch_size != None):
                img_ori = cv2.resize(img_ori, (self.args.patch_size, self.args.patch_size))
                mask_ori = cv2.resize(mask_ori, (self.args.patch_size, self.args.patch_size),interpolation=cv2.INTER_NEAREST)
                img_degradation = cv2.resize(img_degradation, (self.args.patch_size, self.args.patch_size))
                mask_degradation = cv2.resize(mask_degradation, (self.args.patch_size, self.args.patch_size),interpolation=cv2.INTER_NEAREST)

        if self.choice == 'test':
            img_ori, mask_ori = img, mask

            if (self.patch_size != None):
                img_ori = cv2.resize(img_ori, (self.args.patch_size, self.args.patch_size))
                mask_ori = cv2.resize(mask_ori, (self.args.patch_size, self.args.patch_size),interpolation=cv2.INTER_NEAREST)
            img_degradation, mask_degradation = img_ori, mask_ori

        # fig,ax = plt.subplots(2,3)
        # ax[0][0].imshow(img_ori)
        # ax[0][1].imshow(mask_ori)
        # ax[0][2].imshow(np.abs(img_ori - img_degradation))
        # ax[1][0].imshow(img_degradation)
        # ax[1][1].imshow(mask_degradation)
        # ax[1][2].imshow(np.abs(img_ori - img_degradation))
        # plt.show()
        '''Ori image'''
        img_ori = img_ori[:, :, ::-1].astype('float') / 255.
        mask_ori = mask_ori.astype('float')
        mask_ori[np.where(mask_ori < 127.5)] = 0
        mask_ori[np.where(mask_ori >= 127.5)] = 1

        '''Degradation image'''
        img_degradation = img_degradation[:,:,::-1].astype('float') / 255.
        mask_degradation = mask_degradation.astype('float')
        mask_degradation[np.where(mask_degradation < 127.5)] = 0
        mask_degradation[np.where(mask_degradation >= 127.5)] = 1

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        img_ori = (img_ori-mean)/std
        img_degradation = (img_degradation-mean)/std
        # print(img_ori.shape,img_ori_resize.shape,img_degradation.shape,img_degradation_resize.shape)
        # fig,ax = plt.subplots(1,2)
        # ax[0].imshow(img_degradation)
        # ax[1].imshow(mask_degradation)
        # plt.show()
        if(self.choice=='train' or self.choice=='val'):
            return self.tensor(img_ori),self.tensor(mask_ori[:, :, :1]),self.tensor(img_degradation), self.tensor(mask_degradation[:, :, :1]),fname1.split('/')[-1]
        elif(self.choice=='test'):
            return self.tensor(img_ori),self.tensor(mask_ori[:, :, :1]),self.tensor(img_degradation), self.tensor(mask_degradation[:, :, :1]),fname1.split('/')[-1]

    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)
