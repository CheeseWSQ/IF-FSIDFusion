import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import cv2
import os


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)  # 对image_y和image_ir都取出各自图片里面像素最大值
        loss_in=F.l1_loss(x_in_max,generate_img)  # 输入图片与输出图片之间平均元素绝对值差
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+100*loss_grad
        return loss_total,loss_in,loss_grad

class Fusionloss_10(nn.Module):
    def __init__(self):
        super(Fusionloss_10, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)  # 对image_y和image_ir都取出各自图片里面像素最大值
        loss_in=F.l1_loss(x_in_max,generate_img)  # 输入图片与输出图片之间平均元素绝对值差
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+50*loss_grad
        return loss_total,loss_in,loss_grad


class Fusionloss_50(nn.Module):
    def __init__(self):
        super(Fusionloss_50, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)  # 对image_y和image_ir都取出各自图片里面像素最大值
        loss_in=F.l1_loss(x_in_max,generate_img)  # 输入图片与输出图片之间平均元素绝对值差
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+50*loss_grad
        return loss_total,loss_in,loss_grad

class Fusionloss_200(nn.Module):
    def __init__(self):
        super(Fusionloss_200, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)  # 对image_y和image_ir都取出各自图片里面像素最大值
        loss_in=F.l1_loss(x_in_max,generate_img)  # 输入图片与输出图片之间平均元素绝对值差
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=loss_in+200*loss_grad
        return loss_total,loss_in,loss_grad




# 作用应该是用于计算梯度
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)  # 在kernelx前面加两个维度，变成（1，1，kernelx）的Tensor类型
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()  # 根据kernelx在GPU上计算卷积层的权重
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)  # 输出输入与两个卷积核卷积之后的绝对值之和


def cc(img1, img2):
    # “eps”现在将保持一个非常小的正数，接近于零。
    # 这对于防止计算中被零除或零的对数很有用。
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)  # 三维Tensor压缩成二维
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)  # 把CC算子压缩到（-1，1）之间
    return cc.mean()


def SaliencyMap(image):
    # 输入图像必须是 torch.Tensor 类型
    assert isinstance(image, torch.Tensor)
    B, C, H, W = image.shape
    # 转换为灰度图像（简单平均法或使用特定的颜色权重）
    if C == 3:
        image_gray = 0.2989 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.1140 * image[:, 2, :, :]
    else:
        image_gray = image[:, 0, :, :]  # 如果是单通道图像，直接使用
    # 将图像从 torch.Tensor 转换为 numpy 数组，方便使用 OpenCV 进行处理
    image_gray_np = image_gray.cpu().numpy()
    saliency_maps = []

    for i in range(B):
        # 将图像转为 uint8 格式以适应 OpenCV 的 Sobel 操作
        img_uint8 = (image_gray_np[i] * 255).astype(np.uint8)
        
        # 计算 Sobel 梯度（x方向和y方向）
        grad_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # 将梯度幅值归一化到 [0, 1]
        grad_magnitude_norm = cv2.normalize(grad_magnitude, None, 0, 1, cv2.NORM_MINMAX)
        
        saliency_maps.append(grad_magnitude_norm)
    # 将显著性图从 numpy 数组转换回 torch.Tensor，并添加通道维度
    saliency_map_tensor = torch.from_numpy(np.stack(saliency_maps)).unsqueeze(1).to(image.device)
    
    return saliency_map_tensor


class SaliencyLoss(nn.Module):
    def __init__(self):
        super(SaliencyLoss, self).__init__()

    def forward(self, img_vis, img_ir, img_fusion):
        saliency_vis = SaliencyMap(img_vis)
        saliency_ir  = SaliencyMap(img_ir)
        w_vis = saliency_vis / (saliency_vis + saliency_ir)
        w_ir  = saliency_ir  / (saliency_vis + saliency_ir)
        img_saliency = w_vis * img_vis + w_ir * img_ir
        loss = F.l1_loss(img_saliency, img_fusion)
        return loss


class focal_pixel_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_sp, self.gamma_sp = 1, 0.5
        self.alpha_lp, self.gamma_lp = 1, 1
        self.upscale_func = functools.partial(
            F.interpolate, mode='bicubic', align_corners=False)
        self.weig_func = lambda x, y, z: torch.exp((x-x.min()) / (x.max()-x.min()) * y) * z

    def forward(self, x, hr, lr):

        f_BI_x = self.upscale_func(lr, size=hr.size()[2:])

        y_sp = torch.abs(hr - f_BI_x)
        w_y_sp = self.weig_func(y_sp, self.alpha_sp, self.gamma_sp).detach()

        y_lp = torch.abs(hr - f_BI_x - x)
        w_y_lp = self.weig_func(y_lp, self.alpha_lp, self.gamma_lp).detach()

        y_hat = hr - f_BI_x
        loss = torch.mean(w_y_sp * w_y_lp * torch.abs(x - y_hat))

        return loss


class LongTailFusionLoss(nn.Module):
    def __init__(self):
        super(LongTailFusionLoss, self).__init__()
        self.sobelconv=Sobelxy()
        self.alpha_sp, self.gamma_sp = 1, 0.5
        self.alpha_lp, self.gamma_lp = 1, 1
        self.upscale_func = functools.partial(
            F.interpolate, mode='bicubic', align_corners=False)
        self.weig_func = lambda x, y, z: torch.exp((x-x.min()) / (x.max()-x.min()) * y) * z

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)  # 对image_y和image_ir都取出各自图片里面像素最大值
        
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        
        y_sp = torch.abs(x_in_max - generate_img)
        w_y_sp = self.weig_func(y_sp, self.alpha_sp, self.gamma_sp).detach()

        y_lp = torch.abs(x_grad_joint - generate_img_grad)
        w_y_lp = self.weig_func(y_lp, self.alpha_lp, self.gamma_lp).detach()
        
        y_hat = x_grad_joint - generate_img_grad
        loss = torch.mean(w_y_sp * w_y_lp * torch.abs(generate_img - y_hat))

        return loss


if __name__ == '__main__':
    height = 128
    width = 128
    window_size = 8
    batch_size = 8
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = torch.rand([8, 1, 32, 32])
    saliency_loss = SaliencyLoss()
    loss  = saliency_loss(data, data, data)
    print(loss)
