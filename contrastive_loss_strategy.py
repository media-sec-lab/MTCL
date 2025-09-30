import torch
import random

def choose_triplet_loss(predict_map,GT,features,anchor_threshold,is_clustering,sample_way,triplet_loss):
    """
    @param predict_map: 对应分辨率的预测结果
    @param features: 对应分辨率提取到的特征
    @param anchor_threshold: anchor 特征的阈值化
    @param is_clustering: anchor 是否要使用平均后的作为anchor
    @param sample_way:
    @param GT: 对应分辨率的ground-truth
    @return:
    """
    '''计算triplet loss的代码'''
    B, C, H, W = features.shape  # 特征层的尺度
    # 特征进行变换#
    features_reshape = features.permute((0, 2, 3, 1)).contiguous().view((B * H * W, C))  # 将特征层进行reshape成B,C,H*W 再变成B,H*W,C的维度
    GT_reshape = torch.squeeze(GT.permute((0, 2, 3, 1)).contiguous().view((B * H * W, 1)),dim=-1)  # 对GT进行变化
    predict_map_reshape = torch.squeeze(predict_map.permute((0, 2, 3, 1)).contiguous().view((B * H * W, 1)),dim=-1)  # 对quarter output 进行变换


    GT_concat = GT_reshape

    features_ori = features_reshape[GT_concat == 0] #原始点的特征
    features_tamper = features_reshape[GT_concat == 1] #篡改点的特征

    outputs_ori = predict_map_reshape[GT_concat == 0] #原始点的预测概率
    outputs_tamper = predict_map_reshape[GT_concat == 1] #篡改点的预测概率
    '''每一个三元损失需要构造多少个三元样本呢？'''
    anchor_tamper_features = features_tamper[torch.where(outputs_tamper >= anchor_threshold)] #选择置信度高的那些tamper features 作为anchor
    anchor_nums = anchor_tamper_features.shape[0] # 选择到的anchor数量
    count = 0
    while (anchor_nums == 0):
        anchor_tamper_features = features_tamper[torch.where(outputs_tamper >= (anchor_threshold - 0.05 * count))]
        anchor_nums = anchor_tamper_features.shape[0]
        count += 1

    if is_clustering:
        anchor_tamper_features = torch.mean(anchor_tamper_features,dim = 0)
    if sample_way=='prob':
        '''接下来选择这个batch 中，与各个anchor tamper feature 距离最远的tamper feature 作为 positive feature'''
        positive_tamper_features_index = (1 - outputs_tamper).multinomial(anchor_nums, replacement=False) # 根据概率进行采样，（1-预测篡改概率）大的采样概率大
        positive_tamper_features = features_tamper[positive_tamper_features_index]
        negative_ori_features_index = (outputs_ori).multinomial(anchor_nums, replacement=False) # 根据概率进行采样，（预测篡改概率）大的采样概率大
        negative_ori_features= features_ori[negative_ori_features_index]
    elif sample_way=='navie':
        positive_tamper_features = features_tamper[random.sample(range(features_tamper.shape[0]),anchor_nums)]
        negative_ori_features = features_ori[random.sample(range(features_ori.shape[0]),anchor_nums)]
    features_triplet_loss = triplet_loss(anchor_tamper_features, positive_tamper_features, negative_ori_features)
    return features_triplet_loss


def constrative_loss(predict_map,GT,features,anchor_threshold,is_clustering,sample_way,criterion,T = 0.07):
    """
    @param predict_map: 对应分辨率的预测结果
    @param features: 对应分辨率提取到的特征
    @param anchor_threshold: anchor 特征的阈值化
    @param is_clustering: anchor 是否要使用平均后的作为anchor
    @param sample_way:
    @param GT: 对应分辨率的ground-truth
    @return:
    """
    '''计算triplet loss的代码'''
    B, C, H, W = features.shape  # 特征层的尺度
    # 特征进行变换#
    features_reshape = features.permute((0, 2, 3, 1)).contiguous().view((B * H * W, C))  # 将特征层进行reshape成B,C,H*W 再变成B,H*W,C的维度
    GT_reshape = torch.squeeze(GT.permute((0, 2, 3, 1)).contiguous().view((B * H * W, 1)),dim=-1)  # 对GT进行变化
    predict_map_reshape = torch.squeeze(predict_map.permute((0, 2, 3, 1)).contiguous().view((B * H * W, 1)),dim=-1)  # 对quarter output 进行变换


    GT_concat = GT_reshape

    features_ori = features_reshape[GT_concat == 0] #原始点的特征
    features_tamper = features_reshape[GT_concat == 1] #篡改点的特征

    outputs_ori = predict_map_reshape[GT_concat == 0] #原始点的预测概率
    outputs_tamper = predict_map_reshape[GT_concat == 1] #篡改点的预测概率
    '''每一个三元损失需要构造多少个三元样本呢？'''
    anchor_tamper_features = features_tamper[torch.where(outputs_tamper >= anchor_threshold)] #选择置信度高的那些tamper features 作为anchor
    anchor_nums = anchor_tamper_features.shape[0] # 选择到的anchor数量
    count = 0
    while (anchor_nums == 0):
        anchor_tamper_features = features_tamper[torch.where(outputs_tamper >= (anchor_threshold - 0.05 * count))]
        anchor_nums = anchor_tamper_features.shape[0]
        count += 1

    if is_clustering:
        anchor_tamper_features = torch.mean(anchor_tamper_features,dim = 0)
    if sample_way=='prob':
        positive_tamper_features_index = (1 - outputs_tamper).multinomial(anchor_nums, replacement=False) # 根据概率进行采样，（1-预测篡改概率）大的采样概率大
        positive_tamper_features = features_tamper[positive_tamper_features_index]
        # negative_ori_features_index = (outputs_ori).multinomial(anchor_nums, replacement=False) # 根据概率进行采样，（预测篡改概率）大的采样概率大
        negative_ori_features_index = (outputs_ori).multinomial(outputs_ori.shape[0], replacement=False) # 根据概率进行采样，（预测篡改概率）大的采样概率大

        # negative_ori_features_index = (outputs_ori).multinomial(np.min([outputs_ori.shape[0],65536]), replacement=False) # 根据概率进行采样，（预测篡改概率）大的采样概率大
        # negative_ori_features_index = (outputs_ori).multinomial(np.min([outputs_ori.shape[0],10000]), replacement=False) # 根据概率进行采样，（预测篡改概率）大的采样概率大

        negative_ori_features= features_ori[negative_ori_features_index]
    elif sample_way=='naive':
        positive_tamper_features = features_tamper[random.sample(range(features_tamper.shape[0]),anchor_nums)]
        # negative_ori_features = features_ori[random.sample(range(features_ori.shape[0]),np.min([anchor_nums,10000]))]
        negative_ori_features = features_ori
    '''计算对比学习的损失函数'''
    # 先对anchor features、positive_tamper_features 和 negative_ori_features 进行归一化#
    q = torch.nn.functional.normalize(anchor_tamper_features, dim=1)
    k_positive = torch.nn.functional.normalize(positive_tamper_features, dim=1)
    k_negative = torch.nn.functional.normalize(negative_ori_features, dim=1)
    k_negative = torch.permute(k_negative,dims = [1,0])
    l_pos = torch.einsum('nc,nc->n', [q, k_positive]).unsqueeze(-1)

    l_neg = torch.einsum('nc,ck->nk', [q, k_negative])
    logits = torch.cat([l_pos, l_neg], dim=1)
    # print("l_pos.shape {},l_neg.shape {}, logits.shape {} ".format(l_pos.shape,l_neg.shape,logits.shape))

    # print("Positive tamper nums {} Negative ori nums {} l_pos.shape {},l_neg.shape {}, logits.shape {} ".format(positive_tamper_nums,negative_ori_nums,l_pos.shape,l_neg.shape,logits.shape))
    logits /= T
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    features_triplet_loss = criterion(logits, labels)
    # features_triplet_loss = triplet_loss(anchor_tamper_features, positive_tamper_features, negative_ori_features)
    return features_triplet_loss

def choose_triplet_loss_neg_positive_anchor(predict_map,GT,features,anchor_threshold,is_clustering,sample_way,triplet_loss,is_show=False):
    """
    @param predict_map: 对应分辨率的预测结果
    @param features: 对应分辨率提取到的特征
    @param anchor_threshold: anchor 特征的阈值化
    @param is_clustering: anchor 是否要使用平均后的作为anchor
    @param sample_way:
    @param GT: 对应分辨率的ground-truth
    @return:
    """
    '''计算triplet loss的代码'''
    B, C, H, W = features.shape  # 特征层的尺度
    # 特征进行变换#
    features_reshape = features.permute((0, 2, 3, 1)).contiguous().view((B * H * W, C))  # 将特征层进行reshape成B,C,H*W 再变成B,H*W,C的维度
    GT_reshape = torch.squeeze(GT.permute((0, 2, 3, 1)).contiguous().view((B * H * W, 1)),dim=-1)  # 对GT进行变化
    predict_map_reshape = torch.squeeze(predict_map.permute((0, 2, 3, 1)).contiguous().view((B * H * W, 1)),dim=-1)  # 对quarter output 进行变换


    GT_concat = GT_reshape

    features_ori = features_reshape[GT_concat == 0] #原始点的特征
    features_tamper = features_reshape[GT_concat == 1] #篡改点的特征

    outputs_ori = predict_map_reshape[GT_concat == 0] #原始点的预测概率
    outputs_tamper = predict_map_reshape[GT_concat == 1] #篡改点的预测概率
    '''篡改样本的 anchor 构造'''
    anchor_tamper_features = features_tamper[torch.where(outputs_tamper >= anchor_threshold)] #选择置信度高的那些tamper features 作为anchor
    anchor_nums = anchor_tamper_features.shape[0] # 选择到的anchor数量
    count = 0
    while (anchor_nums == 0):
        anchor_tamper_features = features_tamper[torch.where(outputs_tamper >= (anchor_threshold - 0.05 * count))]
        anchor_nums = anchor_tamper_features.shape[0]
        count += 1

    '''原始样本的 anchor 构造'''
    anchor_original_features = features_ori[torch.where((1-outputs_ori) >= anchor_threshold)] #选择置信度高的那些tamper features 作为anchor
    anchor_nums_ori = anchor_original_features.shape[0] # 选择到的anchor数量
    count = 0
    while (anchor_nums_ori == 0):
        anchor_original_features = features_ori[torch.where((1-outputs_ori) >= (anchor_threshold - 0.05 * count))]
        anchor_nums_ori = anchor_original_features.shape[0]
        count += 1
    # 原始的anchor 数量要小于等于篡改像素的数量 #


    min_num = np.min([anchor_nums_ori,features_ori.shape[0],features_tamper.shape[0],anchor_nums])
    # print(anchor_nums_ori, features_ori.shape[0], features_tamper.shape[0],min_num)
    # 随机再从anchor tamper features 和 anchor original features里面再选部分作为最后的anchor #
    anchor_nums = min_num
    # print(min_num)
    anchor_tamper_index = random.sample(range(anchor_tamper_features.shape[0]),anchor_nums)
    anchor_tamper_features = anchor_tamper_features[anchor_tamper_index]
    anchor_nums_ori = min_num
    anchor_ori_index = random.sample(range(anchor_original_features.shape[0]),anchor_nums_ori)
    # anchor_original_features = anchor_original_features[:anchor_nums_ori]
    anchor_original_features = anchor_original_features[anchor_ori_index]
    if is_clustering:
        anchor_tamper_features = torch.mean(anchor_tamper_features,dim = 0,keepdim=False)
        anchor_original_features = torch.mean(anchor_original_features,dim = 0,keepdim=False)
        # print(anchor_tamper_features.shape,anchor_original_features.shape)
    if sample_way=='prob':
        # print(sample_way)
        '''接下来选择这个batch 中，与各个anchor tamper feature 距离最远的tamper feature 作为 positive feature'''
        positive_tamper_features_index = (1 - outputs_tamper).multinomial(anchor_nums, replacement=False) # 根据概率进行采样，（1-预测篡改概率）大的采样概率大
        # print("positive_tamper_features_index:",positive_tamper_features_index)
        # print("outputs_tamper[positive_tamper_features_index: ",outputs_tamper[positive_tamper_features_index])
        # print("outputs_tamper: ",outputs_tamper)
        positive_tamper_features = features_tamper[positive_tamper_features_index]
        negative_ori_features_index = (outputs_ori).multinomial(anchor_nums, replacement=False) # 根据概率进行采样，（预测篡改概率）大的采样概率大
        negative_ori_features= features_ori[negative_ori_features_index]

        '''接下来选择这个batch 中，与各个anchor original feature 距离最远的original feature 作为 positive feature'''
        # positive_original_features_index = outputs_ori.multinomial(anchor_nums_ori,replacement=False)  # 根据概率进行采样，（1-预测篡改概率）大的采样概率大
        # positive_original_features = features_ori[positive_original_features_index]
        # negative_tamper_features_index = (1-outputs_tamper).multinomial(anchor_nums_ori,replacement=False)  # 根据概率进行采样，（预测篡改概率）大的采样概率大
        # negative_tamper_features = features_tamper[negative_tamper_features_index]
        positive_original_features = negative_ori_features
        negative_tamper_features = positive_tamper_features
        # print(positive_tamper_features.shape,negative_ori_features.shape)
        anchor_tamper_features_copy = torch.zeros_like(positive_tamper_features)
        anchor_original_features_copy = torch.zeros_like(positive_original_features)
        anchor_tamper_features_copy[:,:] = anchor_tamper_features
        anchor_original_features_copy[:,:] = anchor_original_features
        anchor_tamper_features = anchor_tamper_features_copy
        anchor_original_features = anchor_original_features_copy
    elif sample_way=='navie':
        positive_tamper_features = features_tamper[random.sample(range(features_tamper.shape[0]),anchor_nums)]
        negative_ori_features = features_ori[random.sample(range(features_ori.shape[0]),anchor_nums)]

        positive_original_features = features_ori[random.sample(range(features_ori.shape[0]),anchor_nums_ori)]
        negative_tamper_features = features_tamper[random.sample(range(features_tamper.shape[0]),anchor_nums_ori)]
    # print(anchor_tamper_features.shape, positive_tamper_features.shape,negative_ori_features.shape)
    # for i in range(anchor_tamper_features.shape[0]):
    #     print(anchor_tamper_features[i,0])
    # 对需要计算的特征使用L2 归一化#
    anchor_tamper_features = F.normalize(anchor_tamper_features, p=2, dim=1)
    positive_tamper_features = F.normalize(positive_tamper_features, p=2, dim=1)
    negative_ori_features = F.normalize(negative_ori_features, p=2, dim=1)
    anchor_original_features = F.normalize(anchor_original_features, p=2, dim=1)
    positive_original_features = F.normalize(positive_original_features, p=2, dim=1)
    negative_tamper_features = F.normalize(negative_tamper_features, p=2, dim=1)

    # positive_tamper_features /= torch.norm(positive_tamper_features,p =2,dim = 1,keepdim=True)
    # negative_ori_features /= torch.norm(negative_ori_features,p =2,dim = 1,keepdim=True)
    # anchor_original_features /= torch.norm(anchor_original_features,p =2,dim = 1,keepdim=True)
    # positive_original_features /= torch.norm(positive_original_features,p =2,dim = 1,keepdim=True)
    # negative_tamper_features /= torch.norm(negative_tamper_features,p =2,dim = 1,keepdim=True)
    # print(anchor_tamper_features.shape, positive_tamper_features.shape,negative_ori_features.shape)

    features_triplet_loss_tamper = triplet_loss(anchor_tamper_features, positive_tamper_features, negative_ori_features)
    features_triplet_loss_original = triplet_loss(anchor_original_features, positive_original_features, negative_tamper_features)
    triplet_loss_value = features_triplet_loss_original + features_triplet_loss_tamper
    return triplet_loss_value


