import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'train'))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb
from torch.nn import init
from model_util_old import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from model_util_old import point_cloud_masking, parse_output_to_tensors, point_cloud_masking_rough
from model_util_old import FrustumPointNetLoss
#from model_util import g_type2class, g_class2type, g_type2onehotclass
#from model_util import g_type_mean_size
#from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
from provider import compute_box3d_iou


NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8 # one cluster for each type
NUM_OBJECT_POINT = 512

g_type2class={'Car':0, 'Van':1, 'Truck':2, 'Pedestrian':3,
              'Person_sitting':4, 'Cyclist':5, 'Tram':6, 'Misc':7}
g_class2type = {g_type2class[t]:t for t in g_type2class}
g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

g_type_mean_size = {'Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
                    'Van': np.array([5.06763659,1.9007158,2.20532825]),
                    'Truck': np.array([10.13586957,2.58549199,3.2520595]),
                    'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
                    'Person_sitting': np.array([0.80057803,0.5983815,1.27450867]),
                    'Cyclist': np.array([1.76282397,0.59706367,1.73698127]),
                    'Tram': np.array([16.17150617,2.53246914,3.53079012]),
                    'Misc': np.array([3.64300781,1.54298177,1.92320313])}


g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3)) # size clustrs
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i,:] = g_type_mean_size[g_class2type[i]]

class PointNetInstanceSeg(nn.Module):
    def __init__(self,n_classes=3,n_channel=4):
        '''v1 3D Instance Segmentation PointNet
        :param n_classes:3
        :param one_hot_vec:[bs,n_classes]
        '''
        super(PointNetInstanceSeg, self).__init__()
        self.conv1 = nn.Conv1d(n_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.n_classes = n_classes
        self.dconv1 = nn.Conv1d(1088+n_classes, 512, 1)
        self.dconv2 = nn.Conv1d(512, 256, 1)
        self.dconv3 = nn.Conv1d(256, 128, 1)
        self.dconv4 = nn.Conv1d(128, 128, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.dconv5 = nn.Conv1d(128, 2, 1)
        self.dbn1 = nn.BatchNorm1d(512)
        self.dbn2 = nn.BatchNorm1d(256)
        self.dbn3 = nn.BatchNorm1d(128)
        self.dbn4 = nn.BatchNorm1d(128)

    def forward(self, pts, one_hot_vec): # bs,4,n；bs,3
        '''
        :param pts: [bs,4,n]: x,y,z,intensity
        :return: logits: [bs,n,2],scores for bkg/clutter and object
        '''
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        out1 = F.relu(self.bn1(self.conv1(pts))) # bs,64,n
        out2 = F.relu(self.bn2(self.conv2(out1))) # bs,64,n
        out3 = F.relu(self.bn3(self.conv3(out2))) # bs,64,n
        out4 = F.relu(self.bn4(self.conv4(out3)))# bs,128,n
        out5 = F.relu(self.bn5(self.conv5(out4)))# bs,1024,n
        global_feat = torch.max(out5, 2, keepdim=True)[0] #bs,1024,1

        expand_one_hot_vec = one_hot_vec.view(bs,-1,1)#bs,3,1
        expand_global_feat = torch.cat([global_feat, expand_one_hot_vec],1)#bs,1027,1
        expand_global_feat_repeat = expand_global_feat.view(bs,-1,1).repeat(1,1,n_pts)# bs,1027,n
        concat_feat = torch.cat([out2, expand_global_feat_repeat],1)
        # bs, (64+1024+3)=1091, n

        x = F.relu(self.dbn1(self.dconv1(concat_feat)))#bs,512,n
        x = F.relu(self.dbn2(self.dconv2(x)))#bs,256,n
        x = F.relu(self.dbn3(self.dconv3(x)))#bs,128,n
        x = F.relu(self.dbn4(self.dconv4(x)))#bs,128,n
        x = self.dropout(x)
        x = self.dconv5(x)#bs, 2, n

        seg_pred = x.transpose(2,1).contiguous()#bs, n, 2
        return seg_pred

class PointNetEstimation(nn.Module):
    def __init__(self,n_classes=3):
        '''v1 Amodal 3D Box Estimation Pointnet
        :param n_classes:3
        :param one_hot_vec:[bs,n_classes]
        '''
        super(PointNetEstimation, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.n_classes = n_classes

        self.fc1 = nn.Linear(512+n_classes, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4)
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)

    def forward(self, pts,one_hot_vec): # bs,3,m
        '''
        :param pts: [bs,3,m]: x,y,z after InstanceSeg
        :return: box_pred: [bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4]
            including box centers, heading bin class scores and residuals,
            and size cluster scores and residuals
        '''
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        out1 = F.relu(self.bn1(self.conv1(pts))) # bs,128,n
        out2 = F.relu(self.bn2(self.conv2(out1))) # bs,128,n
        out3 = F.relu(self.bn3(self.conv3(out2))) # bs,256,n
        out4 = F.relu(self.bn4(self.conv4(out3)))# bs,512,n
        global_feat = torch.max(out4, 2, keepdim=False)[0] #bs,512

        expand_one_hot_vec = one_hot_vec.view(bs,-1)#bs,3
        expand_global_feat = torch.cat([global_feat, expand_one_hot_vec],1)#bs,515

        x = F.relu(self.fcbn1(self.fc1(expand_global_feat)))#bs,512
        x = F.relu(self.fcbn2(self.fc2(x)))  # bs,256
        box_pred = self.fc3(x)  # bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4
        return box_pred

class STNxyz(nn.Module):
    def __init__(self,n_classes=3):
        super(STNxyz, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        #self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.fc1 = nn.Linear(256+n_classes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

        init.zeros_(self.fc3.weight)
        init.zeros_(self.fc3.bias)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.fcbn1 = nn.BatchNorm1d(256)
        self.fcbn2 = nn.BatchNorm1d(128)
    def forward(self, pts,one_hot_vec):
        bs = pts.shape[0]
        x = F.relu(self.bn1(self.conv1(pts)))# bs,128,n
        x = F.relu(self.bn2(self.conv2(x)))# bs,128,n
        x = F.relu(self.bn3(self.conv3(x)))# bs,256,n
        x = torch.max(x, 2)[0]# bs,256
        expand_one_hot_vec = one_hot_vec.view(bs, -1)# bs,3
        x = torch.cat([x, expand_one_hot_vec],1)#bs,259
        x = F.relu(self.fcbn1(self.fc1(x)))# bs,256
        x = F.relu(self.fcbn2(self.fc2(x)))# bs,128
        x = self.fc3(x)# bs,
        ###if np.isnan(x.cpu().detach().numpy()).any():
        ###    ipdb.set_trace()
        return x


def index_labels(label, idx):   # B,7168  B, 1024
    device = label.device
    B = label.shape[0]
    view_shape = list(idx.shape)        # view_shape=[B,S]  # 将idx的shape转换为list (B, S)
    view_shape[1:] = [1] * (len(view_shape) - 1)        # [ B , 1 ] [1:]切片操作，去掉第0个数，剩下list里的数都变为1
    repeat_shape = list(idx.shape)      # repeat_shape=[B,S]
    repeat_shape[0] = 1     # repeat_shape=[1,S]
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)  # batch_indices的维度[B,S]
    new_label = label[batch_indices, idx]      # 从points当中取出每个batch_indices对应索引的数据点
    return new_label.squeeze()


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)        # view_shape=[B,S]  # 将idx的shape转换为list (B, S)
    view_shape[1:] = [1] * (len(view_shape) - 1)        # [ B , 1 ] [1:]切片操作，去掉第0个数，剩下list里的数都变为1
    repeat_shape = list(idx.shape)      # repeat_shape=[B,S]
    repeat_shape[0] = 1     # repeat_shape=[1,S]
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)  # batch_indices的维度[B,S]
    # torch.arange(B) 用于产生一个从0开始,到B结束（注意不包括B）(步长为step的Tensor, 并且可以设置 Tensor的device和dtyp)
    # Tensor[0,1,2....,B-1]--》View后变成列向量--》repeat将列向量复制S次
    new_points = points[batch_indices, idx, :]      # 从points当中取出每个batch_indices对应索引的数据点
    return new_points.squeeze()


def square_distance(src, dst):  # 通道没考虑好，4通道的话矩阵乘法会出错
    """
    Calculate Euclid distance between each two points.      # 计算的是原点（src）集合中N个点到目标点（dst）集合中M点的距离（平方距离，没有开根号），以Batch为单位，输出B×N×M的tensor。

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape     # 单下划线表示不关心
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))      # permute为转置,[B, N, M]     2*(xn * xm + yn * ym + zn * zm)
    # torch.matmul()也是一种类似于矩阵相乘操作的tensor联乘操作。但是它可以利用python中的广播机制,处理一些维度不同的tensor结构进行相乘操作。
    dist += torch.sum(src ** 2, -1).view(B, N, 1)       # [B, N, M] + [B, N, 1]，dist的每一列都加上后面的列值
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)       # [B, N, M] + [B, 1, M], dist的每一行都加上后面的行值
    return dist


# radius为球形领域的半径，nsample为每个领域中要采样的点，ball_center为S个球形领域的中心， S=1
# xyz为所有的点云；输出为每个样本的每个球形领域的nsample个采样点集的索引[B,S,nsample]
def query_ball_point(radius, nsample, xyz, ball_center):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        ball_center: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    B, N, C = xyz.shape
    _, S, _ = ball_center.shape
    group_idx = torch.arange(N, dtype=torch.long).cuda().view(1, 1, N).repeat([B, S, 1])    # [B, S, N]
    sqrdists = square_distance(ball_center, xyz)    # B*S*N  记录中心点与所有点之间的距离
    group_idx[sqrdists > radius ** 2] = N-1       # 找到所有距离大于radius^2的，其group_idx直接置为N；其余的保留原来的值
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]   # 做升序排列，取出前nsample个点, 剩下的都是N
    # 考虑到有可能前nsample个点中也有被赋值为N的点（即球形区域内不足nsample个点），这种点需要舍弃，直接用第一个点来代替
    # group_first: [B, S, k]， 实际就是把group_idx中的第一个点的值复制为了[B, S, K]的维度，便利于后面的替换
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])

    mask = group_idx == N-1   # 找到group_idx中值等于N的点
    group_idx[mask] = group_first[mask]     # 将这些点的值替换为第一个点的值
    group_xyz = index_points(xyz, group_idx)

    return group_xyz, group_idx.squeeze()

class FrustumPointNetv1(nn.Module):
    def __init__(self,n_classes=3,n_channel=4):
        super(FrustumPointNetv1, self).__init__()
        self.n_classes = n_classes
        self.InsSeg = PointNetInstanceSeg(n_classes=3,n_channel=n_channel)
        self.STN = STNxyz(n_classes=3)
        self.est = PointNetEstimation(n_classes=3)

    def forward(self, origin_pts, label_full, pts, one_hot_vec):    # B,m,4; B,7168; B,4,n; B,3
        # 3D Instance Segmentation PointNet  第一次分割找到粗粗中心
        logits_old = self.InsSeg(pts, one_hot_vec)   # bs,n,2

        # Mask Point Centroid
        mask_xyz_mean1 = point_cloud_masking_rough(pts, logits_old)  # B, 4, 1
        mask_xyz_mean1 = mask_xyz_mean1.transpose(2, 1)  # B, 1, 4
        # 找到额外点以及对应的label并返回与原有label拼接
        additional_point, group_idx = query_ball_point(4, 1024, origin_pts, mask_xyz_mean1)  # 以粗粗中心进行球查询  B*1024*4, B,1024
        additional_label = index_labels(label_full, group_idx)  # B, 1024
        additional_point = additional_point.transpose(2, 1)    # B*4*1024
        pts_new = torch.cat([pts, additional_point], 2)     # B*4* (n+1024)

        # 第二次分割
        logits_new = self.InsSeg(pts_new, one_hot_vec)      # B*(n+1024)*2
        object_pts_xyz, mask_xyz_mean, mask = point_cloud_masking(pts_new, logits_new)
        # print(mask.shape)
        ## 原始：object_pts_xyz, mask_xyz_mean, mask = point_cloud_masking(pts, logits)  # logits.detach()

        # T-Net
        object_pts_xyz = object_pts_xyz.cuda()
        center_delta = self.STN(object_pts_xyz,one_hot_vec)  # (32,3)
        stage1_center = center_delta + mask_xyz_mean    # (32,3)     # 调整中心到物体质心位置

        if(np.isnan(stage1_center.cpu().detach().numpy()).any()):   # debug
            ipdb.set_trace()
        object_pts_xyz_new = object_pts_xyz - \
                    center_delta.view(center_delta.shape[0],-1,1).repeat(1,1,object_pts_xyz.shape[-1])  # B,3,n

        # 3D Box Estimation
        box_pred = self.est(object_pts_xyz_new,one_hot_vec)  # (32, 59)

        center_boxnet, \
        heading_scores, heading_residuals_normalized, heading_residuals, \
        size_scores, size_residuals_normalized, size_residuals = \
                parse_output_to_tensors(box_pred, logits_new, mask, stage1_center)

        center = center_boxnet + stage1_center #bs,3
        return logits_old, logits_new, mask, stage1_center, center_boxnet, \
            heading_scores, heading_residuals_normalized, heading_residuals, \
            size_scores, size_residuals_normalized, size_residuals, center,  additional_label


# class FrustumPointNetv1(nn.Module):
#     def __init__(self,n_classes=3,n_channel=4):
#         super(FrustumPointNetv1, self).__init__()
#         self.n_classes = n_classes
#         self.InsSeg = PointNetInstanceSeg(n_classes=3,n_channel=n_channel)
#         self.STN = STNxyz(n_classes=3)
#         self.est = PointNetEstimation(n_classes=3)
#
#     def forward(self, pts, one_hot_vec):    # bs,4,n; bs,3
#         # 3D Instance Segmentation PointNet
#         logits = self.InsSeg(pts,one_hot_vec)   # bs,n,2
#
#         # Mask Point Centroid
#         object_pts_xyz, mask_xyz_mean, mask = point_cloud_masking(pts, logits)  # logits.detach()
#
#         # T-Net
#         object_pts_xyz = object_pts_xyz.cuda()
#         center_delta = self.STN(object_pts_xyz,one_hot_vec)#(32,3)
#         stage1_center = center_delta + mask_xyz_mean    # (32,3)     # 调整中心到物体质心位置
#
#         if(np.isnan(stage1_center.cpu().detach().numpy()).any()):   # debug
#             ipdb.set_trace()
#         object_pts_xyz_new = object_pts_xyz - \
#                     center_delta.view(center_delta.shape[0],-1,1).repeat(1,1,object_pts_xyz.shape[-1])  # 32,3,n
#
#         # 3D Box Estimation
#         box_pred = self.est(object_pts_xyz_new,one_hot_vec)#(32, 59)
#
#         center_boxnet, \
#         heading_scores, heading_residuals_normalized, heading_residuals, \
#         size_scores, size_residuals_normalized, size_residuals = \
#                 parse_output_to_tensors(box_pred, logits, mask, stage1_center)      # logits, mask, stage1_center没用到
#
#         center = center_boxnet + stage1_center #bs,3
#         return logits, mask, stage1_center, center_boxnet, \
#             heading_scores, heading_residuals_normalized, heading_residuals, \
#             size_scores, size_residuals_normalized, size_residuals, center




if __name__ == '__main__':
    #python models/pointnet.py
    points = torch.zeros(size=(32,4,1024),dtype=torch.float32)
    label = torch.ones(size=(32,3))
    model = FrustumPointNetv1()
    logits, mask, stage1_center, center_boxnet, \
            heading_scores, heading_residuals_normalized, heading_residuals, \
            size_scores, size_residuals_normalized, size_residuals, center \
            = model(points,label)
    print('logits:',logits.shape,logits.dtype)
    print('mask:',mask.shape,mask.dtype)
    print('stage1_center:',stage1_center.shape,stage1_center.dtype)
    print('center_boxnet:',center_boxnet.shape,center_boxnet.dtype)
    print('heading_scores:',heading_scores.shape,heading_scores.dtype)
    print('heading_residuals_normalized:',heading_residuals_normalized.shape,\
          heading_residuals_normalized.dtype)
    print('heading_residuals:',heading_residuals.shape,\
          heading_residuals.dtype)
    print('size_scores:',size_scores.shape,size_scores.dtype)
    print('size_residuals_normalized:',size_residuals_normalized.shape,\
          size_residuals_normalized.dtype)
    print('size_residuals:',size_residuals.shape,size_residuals.dtype)
    print('center:', center.shape,center.dtype)
    '''
    logits: torch.Size([32, 1024, 2]) torch.float32
    mask: torch.Size([32, 1024]) torch.float32
    stage1_center: torch.Size([32, 3]) torch.float32
    center_boxnet: torch.Size([32, 3]) torch.float32
    heading_scores: torch.Size([32, 12]) torch.float32
    heading_residual_normalized: torch.Size([32, 12]) torch.float32
    heading_residual: torch.Size([32, 12]) torch.float32
    size_scores: torch.Size([32, 8]) torch.float32
    size_residuals_normalized: torch.Size([32, 8, 3]) torch.float32
    size_residuals: torch.Size([32, 8, 3]) torch.float32
    center: torch.Size([32, 3]) torch.float32
    '''
    loss = FrustumPointNetLoss()
    mask_label = torch.zeros(32,1024).float()
    center_label = torch.zeros(32,3).float()
    heading_class_label = torch.zeros(32).long()
    heading_residuals_label = torch.zeros(32).float()
    size_class_label = torch.zeros(32).long()
    size_residuals_label = torch.zeros(32,3).float()
    output_loss = loss(logits, mask_label, \
                center, center_label, stage1_center, \
                heading_scores, heading_residuals_normalized, heading_residuals, \
                heading_class_label, heading_residuals_label, \
                size_scores,size_residuals_normalized,size_residuals,
                size_class_label,size_residuals_label)
    print('output_loss',output_loss)
    print()