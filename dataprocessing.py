import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from skimage.io import imread


# 根据文件后缀名读取文件，一般为图片
def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
                imagelist.append(os.path.join(parent, filename))
        return imagelist


# RGB图片转换成灰度图片
def rgb2y(img):
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
    return y


# 分割图形方便每一次对一小块进行操作
def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]  # 垂直尺度
    endw = img.shape[1]  # 水平尺度
    endh = img.shape[2]  # 通道数
    # [起始点：结束点：步长]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win, TotalPatNum], np.float32)  # 创建三维数组 保存patch取出的结果
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            # ：符号意味着我们沿着这个维度获取所有元素。“k”表示我们只取第二维度上的第“k”个元素。
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


# 用于确定图像是否具有低对比度
def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10,
                    upper_percentile=90):
    """Determine if an image is low contrast."""
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold


data_name="MSRS_train"
img_size=128   # patch size
stride=200     # patch stride

# 图像排序和检索
IR_files = sorted(get_img_file(r"MSRS_train/ir"))
VIS_files   = sorted(get_img_file(r"MSRS_train/vi"))

assert len(IR_files) == len(VIS_files)

h5f = h5py.File(os.path.join('.\\MSRS_train',
                                 data_name+'_imgsize_'+str(img_size)+"_stride_"+str(stride)+'.h5'),
                    'w')
# # 创建目标文件夹路径
# output_dir = './MIF_train'
# os.makedirs(output_dir, exist_ok=True)  # 自动创建目录

# # 构建完整的文件路径
# h5_path = os.path.join(output_dir, f'{data_name}_imgsize_{img_size}_stride_{stride}.h5')

# # 创建 .h5 文件
# h5f = h5py.File(h5_path, 'w')

# h5f = h5py.File(os.path.join('.\\data',
#                                  data_name+'_imgsize_'+str(img_size)+"_stride_"+str(stride)+'.h5'),
#                     'w')
h5_ir = h5f.create_group('ir_patchs')
h5_vis = h5f.create_group('vis_patchs')
train_num=0
for i in tqdm(range(len(IR_files))):
    # 对于每对IR和VIS图像，读取图像，将它们转换为float32，并将它们的像素值标准化为[0，1]范围
    I_VIS = imread(VIS_files[i]).astype(np.float32).transpose(2,0,1)/255. # [3, H, W] Uint8->float32
    # 从RGB转换为灰度
    I_VIS = rgb2y(I_VIS) # [1, H, W] Float32
    I_IR = imread(IR_files[i]).astype(np.float32)[None, :, :]/255.  # [1, H, W] Float32
        
    # crop
    # 使用指定为“img_size”和“步幅”的“Im2Patch（）”函数从IR和VIS图像创建patch
    I_IR_Patch_Group = Im2Patch(I_IR,img_size,stride)
    I_VIS_Patch_Group = Im2Patch(I_VIS, img_size, stride)  # (3, 256, 256, 12)
        
    for ii in range(I_IR_Patch_Group.shape[-1]):
        # 对于每个patch，使用“is_low_contrast（）”函数检查对比度是否较低。如果IR或VIS贴片的对比度不低，则认为该贴片有效
        bad_IR = is_low_contrast(I_IR_Patch_Group[0,:,:,ii])
        bad_VIS = is_low_contrast(I_VIS_Patch_Group[0,:,:,ii])
        # Determine if the contrast is low
        if not (bad_IR or bad_VIS):
            avl_IR= I_IR_Patch_Group[0,:,:,ii]  # available IR
            avl_VIS= I_VIS_Patch_Group[0,:,:,ii]
            avl_IR=avl_IR[None,...]
            avl_VIS=avl_VIS[None,...]

            # 为每个有效补丁创建一个新的数据集
            h5_ir.create_dataset(str(train_num),     data=avl_IR,
	                            dtype=avl_IR.dtype,   shape=avl_IR.shape)
            h5_vis.create_dataset(str(train_num),    data=avl_VIS,
	                            dtype=avl_VIS.dtype,  shape=avl_VIS.shape)
            train_num += 1

h5f.close()

with h5py.File(os.path.join('MSRS_train',
                                 data_name+'_imgsize_'+str(img_size)+"_stride_"+str(stride)+'.h5'),"r") as f:
# with h5py.File(os.path.join('data',
#                                  data_name+'_imgsize_'+str(img_size)+"_stride_"+str(stride)+'.h5'),"r") as f:
    for key in f.keys():
        print(f[key], key, f[key].name) 
    





    
