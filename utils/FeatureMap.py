import torch
import os
import numpy as np 
import matplotlib.pyplot as plt

def plt_show_decomp_feature(feature_vis,
                     feature_shd,
                     feature_ir,
                     channel_index, 
                     colarmap='viridis',
                     ):
    # if isinstance(feature, torch.Tensor):
    #     feature_map = feature[0, channel_index, :, :].detach().cpu().numpy()
    # else:
    #     feature_map = feature[0, channel_index, :, :]
    feature_map_vis = feature_vis[0, channel_index, :, :].detach().cpu().numpy()
    feature_map_shd = feature_shd[0, channel_index, :, :].detach().cpu().numpy()
    feature_map_ir  = feature_ir[0, channel_index, :, :].detach().cpu().numpy()

    print(feature_map_shd.shape)

    plt.figure(figsize=(11.6, 9.6))

    plt.subplot(3,1,1) 
    plt.imshow(feature_map_vis, cmap=colarmap)
    plt.colorbar()
    plt.title(f"VIS Specific Feature - Channel {channel_index}") 
    plt.axis("off")
    
    plt.subplot(3,1,2)
    plt.imshow(feature_map_shd, cmap=colarmap)
    plt.colorbar()
    plt.title(f"Cross Modality Shared Feature - Channel {channel_index}")
    plt.axis("off")

    plt.subplot(3,1,3)
    plt.imshow(feature_map_ir, cmap=colarmap)
    plt.colorbar()
    plt.title(f"IR Specific Feature - Channel {channel_index}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    # plt.savefig(save_pth, transparent=True)


def plt_save_decomp_feature(
                     image_name,
                     feature_vis,
                     feature_shd,
                     feature_ir,
                     channel_index, 
                     colarmap='viridis',
                     ):
    # if isinstance(feature, torch.Tensor):
    #     feature_map = feature[0, channel_index, :, :].detach().cpu().numpy()
    # else:
    #     feature_map = feature[0, channel_index, :, :]

    save_dir = '.feature_maps/MSRS'
    os.makedirs(save_dir, exist_ok=True)  # 自动创建目录（如果不存在）
    filename_vis = os.path.join(save_dir, f"{image_name}_vis.svg")
    filename_ir  = os.path.join(save_dir, f"{image_name}_ir.svg")
    filename_shd = os.path.join(save_dir, f"{image_name}_shd.svg")

    feature_map_vis = feature_vis[0, channel_index, :, :].detach().cpu().numpy()
    feature_map_shd = feature_shd[0, channel_index, :, :].detach().cpu().numpy()
    feature_map_ir  = feature_ir[0, channel_index, :, :].detach().cpu().numpy()

    # print(feature_map_shd.shape)

    plt.figure(figsize=(11.6, 9.6))

    plt.imshow(feature_map_vis, cmap=colarmap)
    plt.axis('off')
    plt.savefig(filename_vis, format='svg', bbox_inches='tight')
    plt.close()

    plt.imshow(feature_map_ir, cmap=colarmap)
    plt.axis('off')
    plt.savefig(filename_ir, format='svg', bbox_inches='tight')
    plt.close()

    plt.imshow(feature_map_shd, cmap=colarmap)
    plt.axis('off')
    plt.savefig(filename_shd, format='svg', bbox_inches='tight')
    plt.close()



def plt_show_fuse_feature(feature_fuse,
                     feature_shd,
                     channel_index, 
                     colarmap='viridis',
                     ):
    # if isinstance(feature, torch.Tensor):
    #     feature_map = feature[0, channel_index, :, :].detach().cpu().numpy()
    # else:
    #     feature_map = feature[0, channel_index, :, :]
    feature_map_fuse = feature_fuse[0, channel_index, :, :].detach().cpu().numpy()
    feature_map_shd = feature_shd[0, channel_index, :, :].detach().cpu().numpy()

    plt.figure(figsize=(11.6, 9.6))

    plt.subplot(2,1,1) 
    plt.imshow(feature_map_fuse, cmap=colarmap)
    plt.colorbar()
    plt.title(f"Cross Modality Fused Feature - Channel {channel_index}") 
    plt.axis("off")
    
    plt.subplot(2,1,2)
    plt.imshow(feature_map_shd, cmap=colarmap)
    plt.colorbar()
    plt.title(f"Cross Modality Shared Feature - Channel {channel_index}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def plt_show_distribution(feature,
                      channel_index,
                      ):
    channel_data = feature[0, channel_index, :, :].detach().cpu().numpy()
    flatten_data = channel_data.reshape(-1)

    plt.figure(figsize=(10,6))
    n, bins, patches = plt.hist(
                                flatten_data, 
                                bins=50,                   # 直方图柱子数量
                                color='skyblue',           # 颜色
                                edgecolor='black',         # 边缘色
                                alpha=0.7,                 # 透明度
                                density=True,              # 是否归一化为概率密度
                                label='Feature Values'     # 图例
                                )
    
    mean_val = np.mean(flatten_data)
    std_val = np.std(flatten_data)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    plt.axvline(mean_val + std_val, color='green', linestyle=':', linewidth=2, label=f'±1 Std: {std_val:.2f}')
    plt.axvline(mean_val - std_val, color='green', linestyle=':', linewidth=2)

    plt.xlabel('Feature Value')
    plt.ylabel('Probability Density' )
    plt.title(f'Distribution of Feature Values (Channel {channel_index})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()


def plt_show_decomp_distribution(feature_vis,
                                 feature_shd,
                                 feature_ir,
                                 channel_index,
                                ):
    channel_data_shd = feature_shd[0, channel_index, :, :].detach().cpu().numpy()
    flatten_data = channel_data_shd.reshape(-1)

    plt.figure(figsize=(10,6))
    n, bins, patches = plt.hist(
                                flatten_data, 
                                bins=50,                   # 直方图柱子数量
                                color='skyblue',           # 颜色
                                edgecolor='black',         # 边缘色
                                alpha=0.7,                 # 透明度
                                density=True,              # 是否归一化为概率密度
                                label='Feature Values'     # 图例
                                )
    
    mean_val = np.mean(flatten_data)
    std_val = np.std(flatten_data)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    plt.axvline(mean_val + std_val, color='green', linestyle=':', linewidth=2, label=f'±1 Std: {std_val:.2f}')
    plt.axvline(mean_val - std_val, color='green', linestyle=':', linewidth=2)

    plt.xlabel('Feature Value')
    plt.ylabel('Probability Density' )
    plt.title(f'Distribution of Feature Values (Channel {channel_index})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()    