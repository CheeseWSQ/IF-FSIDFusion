from net import Restormer_Decoder, DetailFeatureExtraction, Restormer_resolve_Encoder
import os
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# ckpt_path=r"models/CDDFuse_IVF.pth"
ckpt_path= r"models/FSIDFusion-IVIF.pth"
for dataset_name in ["MSRS_detect"]:
    print("\n"*2+"="*120)
    model_name="SpaceCrossDecompFuse    "
    print("The test result of "+dataset_name+' :')
    test_folder=os.path.join('test_img',dataset_name) 
    test_out_folder=os.path.join('test_result',dataset_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Encoder = nn.DataParallel(Restormer_resolve_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    # SepcificFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
    SepcificFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=4)).to(device)
    # SepcificFuseLayer = nn.DataParallel(SpecificFeatureCross()).to(device)

    Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    # BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    SepcificFuseLayer.load_state_dict(torch.load(ckpt_path)['SepcificFuseLayer'])
    Encoder.eval()
    Decoder.eval()
    # BaseFuseLayer.eval()
    SepcificFuseLayer.eval()

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder,"ir")):

            data_IR=image_read_cv2(os.path.join(test_folder,"ir",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
            data_VIS = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0

            data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

            # feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
            # feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
            shared_feature, specific_feature_vis, specific_feature_ir = Encoder(data_VIS, data_IR)
        #     print(feature_V_B)
            # feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
            specific_feature_fuse = SepcificFuseLayer(specific_feature_vis + specific_feature_ir)
            # specific_feature_fuse = SepcificFuseLayer(specific_feature_vis, specific_feature_ir)
            data_Fuse, _ = Decoder(data_VIS, shared_feature, specific_feature_fuse)
            data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            # 下面这一行是我加的
            fi = fi.astype(np.uint8)
            #
            img_save(fi, img_name.split(sep='.')[0], test_out_folder)


    eval_folder=test_out_folder  
    ori_img_folder=test_folder

    metric_result = np.zeros((11))
    for img_name in os.listdir(os.path.join(ori_img_folder,"ir")):
            ir = image_read_cv2(os.path.join(ori_img_folder,"ir", img_name), 'GRAY')
            vi = image_read_cv2(os.path.join(ori_img_folder,"vi", img_name), 'GRAY')
            fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')
            metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                        , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                        , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                        , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)
                                        , Evaluator.AG(fi), Evaluator.CC(fi, ir, vi)
                                        , Evaluator.PSNR(fi, ir, vi)])

    metric_result /= len(os.listdir(eval_folder))
    print("\t\t\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM\tAG\tCC\tPSNR")
    print(model_name+'\t'+str(np.round(metric_result[0], 2))+'\t'
            +str(np.round(metric_result[1], 2))+'\t'
            +str(np.round(metric_result[2], 2))+'\t'
            +str(np.round(metric_result[3], 2))+'\t'
            +str(np.round(metric_result[4], 2))+'\t'
            +str(np.round(metric_result[5], 2))+'\t'
            +str(np.round(metric_result[6], 2))+'\t'
            +str(np.round(metric_result[7], 2))+'\t'
            +str(np.round(metric_result[8], 2))+'\t'
            +str(np.round(metric_result[9], 2))+'\t'
            +str(np.round(metric_result[10], 2))
            )
    print("="*120)