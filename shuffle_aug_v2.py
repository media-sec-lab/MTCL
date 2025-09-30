import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(1000)
def inpainting(image, mask, inpainting_h, inpainting_w):
    row, col = mask.shape[0], mask.shape[1]
    if (row <= inpainting_h or col <= inpainting_w):
        image = cv2.resize(image, (int(inpainting_w * 2), int(inpainting_h * 2)))
        mask = cv2.resize(mask, (int(inpainting_w * 2), int(inpainting_h * 2)), interpolation=cv2.INTER_NEAREST)
    mask_ = np.zeros(image.shape[:2], dtype="uint8")
    row, col = mask_.shape[0], mask_.shape[1]
    x0, y0 = random.randint(0, row - inpainting_h - 1), random.randint(0, col - inpainting_w - 1)
    x1, y1 = x0 + inpainting_h, y0 + inpainting_w
    mask_[x0:x1, y0:y1] = 255
    img = cv2.inpaint(image, mask_, 7, cv2.INPAINT_NS)
    mask[x0:x1, y0:y1, :] = 255
    return img, mask

def copy_move(img, img2, msk, copy_move_h, copy_move_w):
    scale_prop = 1
    img = cv2.resize(img, (512,512))
    msk = cv2.resize(msk, (512,512), interpolation=cv2.INTER_NEAREST)
    row, col = msk.shape[0], msk.shape[1]
    if(row<=copy_move_h or col<=copy_move_w):
        img = cv2.resize(img,(int(copy_move_w*2),int(copy_move_h*2)))
        msk = cv2.resize(msk, (int(copy_move_w * 2), int(copy_move_h * 2)),interpolation=cv2.INTER_NEAREST)
        row, col = msk.shape[0], msk.shape[1]
    x0, y0 = random.randint(0, row - copy_move_h - 1), random.randint(0, col - copy_move_w - 1)
    x1, y1 = x0 + copy_move_h, y0 + copy_move_w
    x2, y2 = random.randint(0, row - copy_move_h - 1), random.randint(0, col - copy_move_w - 1)
    x3, y3 = x2 + copy_move_h, y2 + copy_move_w
    if img2 is None:
        img[x0:x1, y0:y1, :] = img[x2:x3, y2:y3, :]
        msk[x0:x1, y0:y1, :] = np.ones_like(msk[x2:x3, y2:y3, :]) * 255
    else:
        img2 = cv2.resize(img2, (col, row))
        # print(img.shape,img2.shape,x0,x1,y0,y1,x2,y2,x3,y3)
        img[x0:x1, y0:y1, :] = img2[x2:x3, y2:y3, :]
        msk[x0:x1, y0:y1, :] = np.ones_like(msk[x2:x3, y2:y3, :]) * 255
    return img, msk


def flip_rotate(img,mask):
    if random.random() < 0.5:
        img = cv2.flip(img, 0)
        mask = cv2.flip(mask, 0)
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
    if random.random() < 0.5:
        tmp = random.random()
        if tmp < 0.33:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        elif tmp < 0.66:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            img = cv2.rotate(img, cv2.ROTATE_180)
            mask = cv2.rotate(mask, cv2.ROTATE_180)
    return img,mask


def gaussian_blur(img,mask = None):
    blur_kernel = random.choice([3, 5, 7, 9, 11])
    img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), sigmaX=0, sigmaY=0)
    return img,mask
def median_blur(img,mask = None):
    blur_kernel = random.choice([3, 5, 7, 9, 11])
    img = cv2.medianBlur(img, blur_kernel)
    return img,mask
def mean_blur(img,mask = None):
    blur_kernel = random.choice([3, 5, 7, 9, 11])
    img = cv2.blur(img, (blur_kernel, blur_kernel))
    return img,mask

def add_gaussian_noise(img,mask = None):
    H, W, C = img.shape
    N = np.random.randint(10, 100) / 10. * np.random.normal(loc=0, scale=1, size=(H, W, 1))
    N = np.repeat(N, C, axis=2)
    img = img.astype(np.int32)
    img = N + img
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img,mask

def add_possion_noise(img,mask = None):
    H, W, C = img.shape
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    alpha = np.random.uniform(2,4)
    pos_noise = np.random.poisson(np.power(10,alpha)*gray)/np.power(10,alpha) - gray
    img = img + np.expand_dims(pos_noise,axis=2)
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img

def JPEG_compression(img,mask = None):
    compression_quality = random.randint(70, 100)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality])
    img = cv2.imdecode(encimg, 1)
    return img,mask
def resize(img,mask):
    # resize_factor = random.randint(5,20)/10
    # H,W,C = img.shape
    # img = cv2.resize(img,(int(W*resize_factor),int(H*resize_factor)),interpolation=cv2.INTER_LINEAR)
    # mask = cv2.resize(mask,(int(W*resize_factor),int(H*resize_factor)),interpolation=cv2.INTER_NEAREST)
    if random.random() < 0.5:
        r1 = random.randint(70,130)/100.0
        r2 = random.randint(70,130)/100.0
        H, W, C = img.shape
        img = cv2.resize(img, (int(W * r1), int(H * r2)))
        mask = cv2.resize(mask, (int(W * r1), int(H * r2)),interpolation=cv2.INTER_NEAREST)
        mask[mask < 127.5] = 0
        mask[mask >= 127.5] = 255
    return img,mask

def resize_withfactor(img,mask,resize_factor_W,resize_factor_H):

    H,W,C = img.shape
    # img = cv2.resize(img,(int(W*resize_factor),int(H*resize_factor)),interpolation=cv2.INTER_LINEAR)
    # mask = cv2.resize(mask,(int(W*resize_factor),int(H*resize_factor)),interpolation=cv2.INTER_NEAREST)
    img = cv2.resize(img,(int(W*resize_factor_W),int(H*resize_factor_H)),interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask,(int(W*resize_factor_W),int(H*resize_factor_H)),interpolation=cv2.INTER_NEAREST)
    mask[mask < 127.5] = 0
    mask[mask >= 127.5] = 255
    return img,mask,resize_factor_W,resize_factor_H
def random_crop(img,mask):
    patch_size = random.randint(256,768)
    crop_shape = (patch_size,patch_size)
    if (img.shape[0] == crop_shape[0] and img.shape[1] == crop_shape[1]):
        return img, mask
    elif img.shape[0] < crop_shape[0] or img.shape[1] < crop_shape[1]:
        img = cv2.resize(img, (crop_shape[1], crop_shape[0]))
        mask = cv2.resize(mask, (crop_shape[1], crop_shape[0]), interpolation=cv2.INTER_NEAREST)
        return img, mask
    else:
        original_shape = mask.shape
        count = 0
        start_h = np.random.randint(0, original_shape[0] - crop_shape[0] + 1)
        start_w = np.random.randint(0, original_shape[1] - crop_shape[1] + 1)
        crop_img = img[start_h: start_h + crop_shape[0], start_w: start_w + crop_shape[1], :]
        crop_mask = mask[start_h: start_h + crop_shape[0], start_w: start_w + crop_shape[1], :]
        count += 1
        return crop_img, crop_mask



def shuffle_aug_online_tampering_flip_rotate(img,img2,mask,is_shuffle):
    options = {0: gaussian_blur,
               1: median_blur,
               2: mean_blur,
               3: add_gaussian_noise,
               4: JPEG_compression,
               5: resize,
               6: flip_rotate,
               7: random_crop,
               8: copy_move,
               9: inpainting,
               10: resize_withfactor}
    random_function = [6,8,9]
    probs = {6:1,8:0.3,9:0.15}
    if(is_shuffle):
        random.shuffle(random_function)
    for i, index in enumerate(random_function):
        img = np.uint8(img)
        mask = np.uint8(mask)
        prob = probs[index]
        if (random.random() < prob):
            if (index == 8):
                copy_move_h, copy_move_w = random.randint(100, 150), random.randint(100, 150)
                if (random.random() < 0.5):
                    img, mask = options[index](img, img2, mask, copy_move_h, copy_move_w)
                else:
                    img, mask = options[index](img, None, mask, copy_move_h, copy_move_w)
            elif (index == 9):
                inpainting_h, inpainting_w = random.randint(100, 150), random.randint(100, 150)
                img, mask = options[index](img, mask, inpainting_h, inpainting_w)
            else:
                img, mask = options[index](img, mask)
    return img,mask


def shuffle_aug_degradation(img,mask,is_shuffle):
    options = {0: gaussian_blur,
               1: median_blur,
               2: mean_blur,
               3: add_gaussian_noise,
               4: JPEG_compression,
               5: resize,
               6: flip_rotate,
               7: random_crop,
               8: copy_move,
               9: inpainting,
               10: resize_withfactor}
    random_function = [0,1,2,3,4,10]
    probs = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2,10: 0.2} #每种后处理的概率都是0.2
    # probs = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5,10: 0.5} # 每种后处理概率都是0.5
    # probs = {0: 0.3, 1: 0.3, 2: 0.3, 3: 0.3, 4: 0.3,10: 0.0} #每种后处理的概率都是0.3
    # probs = {0: 0.8, 1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8,10: 0.8} # 每种后处理概率都是0.5

    # probs = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1,10: 1} # 每种后处理概率都是1

    # random_function = [5,6,2,1,0,3,8,9]
    # probs = {5:1,6:1,2:0.2,1:0.2,0:0.2,3:0.2,8:0.3,9:0.15}
    if(is_shuffle):
        random.shuffle(random_function)
    # probs = [1,1,0.2,0.2,0.2,0.2,0.3,0.15]
    # fig,ax = plt.subplots(2,len(random_function))
    resize_factor_W = 1.0
    resize_factor_H = 1.0
    post_processing_chains = []
    for i,index in enumerate(random_function):
        img = np.uint8(img)
        mask = np.uint8(mask)
        prob = probs[index]
        # print(prob)
        if(random.random()<prob):
            if(index==10):
                resize_factor_H = random.randint(5,20)/10.0
                resize_factor_W = random.randint(5,20)/10.0

                img, mask, resize_factor_W,resize_factor_H = options[index](img, mask, resize_factor_W,resize_factor_H)
            else:
                img, mask = options[index](img, mask)
        post_processing_chains.append(str(options[index]))
        # if (index == 10):
        #     resize_factor_H = random.randint(5, 20) / 10.0
        #     resize_factor_W = random.randint(5, 20) / 10.0
        #
        #     img, mask, resize_factor_W, resize_factor_H = options[index](img, mask, resize_factor_W,
        #                                                                  resize_factor_H)
        # else:
        #     img, mask = options[index](img, mask)
    #         ax[0][i].imshow(img)
    #         ax[1][i].imshow(mask)
    #         ax[0][i].set_title(str(index))
    # plt.show()
    # print(post_processing_chains)
    return img,mask, resize_factor_W,resize_factor_H

