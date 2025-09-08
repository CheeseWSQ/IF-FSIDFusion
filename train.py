# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from net import Restormer_Decoder, DetailFeatureExtraction, Restormer_resolve_Encoder
from utils.dataset import H5Dataset
from utils.patch_optimal_transmission import optimal_trans, MIFtensorOP
import os
import ot


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from utils.loss import Fusionloss,Fusionloss_50,Fusionloss_200,Fusionloss_10, cc
import kornia

'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
criteria_fusion = Fusionloss()
model_str = 'SpecificDecompFuse'

# . Set the hyper-parameters for training
num_epochs = 120  # total epoch
epoch_gap = 40  # epoches of Phase I

lr = 1e-4
weight_decay = 0
batch_size = 8
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# Coefficients of the loss function
coeff_mse_loss_VF = 1.  # alpha1
coeff_mse_loss_IF = 1.
coeff_decomp = 0.1  # alpha2 and alpha4
coeff_tv = 5.
coeff_specific = 1
coeff_rec = 1

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5

# ot paremeters
feature_dim = 64
reg = 0.1

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DIDF_Encoder = nn.DataParallel(Restormer_resolve_Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
SepcificFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=4)).to(device)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    SepcificFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIM(11, reduction='mean')

# data loader
trainloader = DataLoader(H5Dataset(r"MSRS_train/MSRS_train_imgsize_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    ''' train '''
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        DIDF_Encoder.train()
        DIDF_Decoder.train()
        SepcificFuseLayer.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        SepcificFuseLayer.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        if epoch < epoch_gap:  # Phase I
            shared_feature, detail_feature_vis, detail_feature_ir = DIDF_Encoder(data_VIS, data_IR)

            data_VIS_hat, _ = DIDF_Decoder(data_VIS, shared_feature, detail_feature_vis)
            data_IR_hat, _ = DIDF_Decoder(data_IR, shared_feature, detail_feature_ir)

            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))

            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
                   mse_loss_I + coeff_tv * Gradient_loss

            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()
            optimizer2.step()

            batches_done = epoch * len(loader['train']) + i
            batches_left = num_epochs * len(loader['train']) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [Total-Loss: %f] ETA: %.10s \r[MSE-Loss: %f] [Gradient-Loss: %f]\n"          
        
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
                coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * mse_loss_I,
                coeff_tv * Gradient_loss,
                
              )
            )

        else:  # Phase II
            shared_feature, detail_feature_vis, detail_feature_ir = DIDF_Encoder(data_VIS, data_IR)
            data_VIS_hat, _ = DIDF_Decoder(data_VIS, shared_feature, detail_feature_vis)
            data_IR_hat, _ = DIDF_Decoder(data_IR, shared_feature, detail_feature_ir)

            feature_F_Sepcific = SepcificFuseLayer(detail_feature_vis + detail_feature_ir)

            data_Fuse, feature_F = DIDF_Decoder(data_VIS, shared_feature, feature_F_Sepcific)

            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))
            

            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)


            fusionloss, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse)

            loss = fusionloss + coeff_rec *(mse_loss_V + mse_loss_I + coeff_tv * Gradient_loss) 
            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

            nn.utils.clip_grad_norm_(
                SepcificFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()

            batches_done = epoch * len(loader['train']) + i
            batches_left = num_epochs * len(loader['train']) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [Total-Loss: %f] ETA: %.10s \r[Fusion-Loss: %f][MSE-Loss: %f] [Gradient-Loss: %f] \n"          
        
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
                fusionloss,
                coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * mse_loss_I,
                coeff_tv * Gradient_loss,
                
              )
            )

    # adjust the learning rate

    scheduler1.step()
    scheduler2.step()
    if not epoch < epoch_gap:
        scheduler3.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6

if True:
    checkpoint = {
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'SepcificFuseLayer': SepcificFuseLayer.state_dict(),
    }
    torch.save(checkpoint, os.path.join("models/FSIDFusion_IVIF" + timestamp + '.pth'))