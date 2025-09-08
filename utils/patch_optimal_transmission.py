import ot
import numpy as np
import torch

def sinkhorn(a, b, M, reg=1e-3, num_iter=1000, eps=1e-6):
    """
    Sinkhorn 算法实现最优传输匹配。

    参数:
        a: 源分布 (batch_size, n), 必须满足 sum(a) = 1。
        b: 目标分布 (batch_size, m), 必须满足 sum(b) = 1。
        M: 代价矩阵 (batch_size, n, m), 表示从源到目标的传输代价。
        reg: 正则化参数 (默认 1e-3)。
        num_iter: 最大迭代次数 (默认 1000)。
        eps: 收敛阈值 (默认 1e-6)。

    返回:
        gamma: 最优传输计划 (batch_size, n, m)。
    """
    # 定义分布
    u = torch.ones_like(a) / a.shape[1] 
    v = torch.ones_like(b) / b.shape[1]  

    # Sinkhorn 迭代
    for i in range(num_iter):
        # 更新 u
        K = torch.exp(-M / reg)  # Gibbs 核
        u_prev = u
        u = a / (K @ v.unsqueeze(-1)).squeeze(-1)

        # 更新 v
        v = b / (K.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1)

        # 检查收敛
        if torch.norm(u - u_prev) < eps:
            break

    # 计算最优传输计划 gamma
    gamma = u.unsqueeze(-1) * K * v.unsqueeze(1)
    return gamma



def patch_ot(
    feature_vis,
    feature_ir
):
    reg = 0.1
    cut_patch_size  = 16
    feature_ir_patch = feature_ir.unfold(2, cut_patch_size, cut_patch_size).unfold(3, cut_patch_size, cut_patch_size)
    feature_vis_patch = feature_vis.unfold(2, cut_patch_size, cut_patch_size).unfold(3, cut_patch_size, cut_patch_size)

    print("patch_size:", feature_vis_patch.shape) 
            
    flat_feature_vis = feature_vis_patch.reshape(-1, cut_patch_size*cut_patch_size)
    flat_feature_ir = feature_ir_patch.reshape(-1, cut_patch_size*cut_patch_size)

    print("flat_patch_size:", flat_feature_vis.shape)

    # cost_matrix = ot.dist(detail_feature_vis, detail_feature_ir, metric='euclidean') 
    cost_matrix = ot.dist(flat_feature_vis.numpy(), flat_feature_ir.numpy(), metric='euclidean')

    n_samples = flat_feature_vis.shape[0]
    a = np.ones(n_samples) / n_samples    # define uniform distribution
    b = np.ones(n_samples) / n_samples
            
    transport_matrix = ot.sinkhorn(a, b, cost_matrix, reg=reg)

    alinged_flat_feature_vis = np.dot(transport_matrix.T, flat_feature_vis.numpy())

    alinged_flat_feature_vis = alinged_flat_feature_vis.reshape(8, 64, 8, 8, 16, 16)
    alinged_flat_feature_ir = flat_feature_ir.reshape(8, 64, 8, 8, 16, 16)

    batch_size, channels, num_patches_h, num_patches_w, patch_h, patch_w = feature_vis_patch.shape
    alinged_feature_vis = alinged_flat_feature_vis.contiguous().view(batch_size, channels, -1, patch_h * patch_w)
    alinged_feature_ir  = alinged_flat_feature_ir.contiguous().view(batch_size, channels, -1, patch_h * patch_w)

    return alinged_feature_vis, alinged_feature_ir


# def optimal_trans(
#     feature_vis,
#     feature_ir
# ):
    
#     # flat_feature_vis = feature_vis.reshape(8, 64, -1)   
#     # flat_feature_ir  = feature_ir.reshape(8, 64, -1)
#     flat_feature_vis = feature_vis.reshape(512, -1)   
#     flat_feature_ir  = feature_ir.reshape(512, -1)
#     # print(flat_feature_vis.shape)
#     feature_vis = feature_vis.cpu()
#     feature_ir  = feature_ir.cpu()

#     # cost_matrix = ot.dist(flat_feature_vis.detach().numpy(), flat_feature_ir.detach().numpy(), metric='euclidean')
#     cost_matrix = ot.dist(feature_vis.detach().numpy(), feature_ir.detach().numpy(), metric='euclidean')
#     # cost_matrix = torch.cdist(flat_feature_vis, flat_feature_ir, p=2)
#     print("cost_martrix shape:", cost_matrix.shape)
#     cost_matrix = cost_matrix.cpu().detach().numpy()

#     n_samples = flat_feature_vis.shape[0]
#     a = np.ones(n_samples) / n_samples    # define uniform distribution
#     b = np.ones(n_samples) / n_samples
            
#     transport_matrix = ot.sinkhorn(a, b, cost_matrix, reg=0.1)

#     # alinged_flat_feature_vis = np.dot(transport_matrix.T, flat_feature_vis.numpy())
#     alinged_feature_vis = np.dot(transport_matrix.T, feature_vis.detach().numpy())

#     return alinged_feature_vis, feature_ir

def MinMaxNorm(feature):
    min_val = feature.min()
    max_val = feature.max()

    feature_norm = (feature- min_val) / (max_val - min_val)
    return feature_norm


def optimal_trans(
    feature_vis,
    feature_ir
):
    feature_vis = feature_vis / feature_vis.sum(dim=-2, keepdim=True)
    feature_ir  = feature_ir  / feature_ir.sum(dim=-2, keepdim=True)
    

    # gamma = torch.zeros(8, 1, 128 * 128, 128 * 128).cpu()
    gamma = np.zeros(8, 128 * 128, 128 * 128).cpu()
    gamma = np.expand_dims(gamma, axis=1)
    print(gamma.shape)
    for batch in range(8):
        for channel in range(64):
            # 提取当前 batch 和 channel 的特征图 (128, 128)
            flat_feature_vis = feature_vis[batch, channel].view(-1).cpu()  # 展平为 (128*128,)
            flat_feature_ir  = feature_ir [batch, channel].view(-1).cpu()  # 展平为 (128*128,)

            # 计算代价矩阵 (128*128, 128*128)
            M = ot.dist(flat_feature_vis.unsqueeze(1), flat_feature_ir.unsqueeze(1))  # 使用欧几里得距离
            print(M.shape)
            print(type(M))
            M = M.detach().numpy()
            # 计算最优传输计划
            gamma[batch, 1] = ot.emd(flat_feature_vis.detach().numpy(), flat_feature_ir.detach().numpy(), M)
            alinged_feature_vis[batch, channel] = np.dot(gamma[batch, 1].T, flat_feature_vis.detach().numpy())

    alinged_feature_vis = alinged_feature_vis.reshape(8, 64, 128, 128)

    return alinged_feature_vis, feature_ir


def MIFtensorOP(
    feature_vis,
    feature_ir
):
    batch_size, channels, height, width = feature_vis.size()
    feature_vis = feature_vis.cuda()
    feature_ir  = feature_ir.cuda()

    flat_feature_vis = feature_vis.view(batch_size, channels, -1)
    flat_feature_ir  = feature_ir.view(batch_size, channels, -1)

    flat_feature_vis_norm = flat_feature_vis / flat_feature_vis.sum(dim=-2, keepdim=True)
    flat_feature_ir_norm = flat_feature_ir  / flat_feature_ir.sum(dim=-2, keepdim=True)
    print(flat_feature_vis_norm.shape)

    # M = torch.cdist(flat_feature_vis_norm, flat_feature_ir_norm, p=2)
    M = torch.sqrt((flat_feature_vis - flat_feature_ir)**2)
    print(M.shape)

    gamma = sinkhorn(flat_feature_vis_norm, flat_feature_ir_norm, M, reg=1e-3)

    alinged_feature_vis = torch.einsum('bcij,bcj->bci', gamma.T, flat_feature_vis_norm)
    
    return alinged_feature_vis, feature_ir