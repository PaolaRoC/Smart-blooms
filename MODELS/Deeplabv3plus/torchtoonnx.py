# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:02:46 2024

@author: Paola
"""
import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from /deeplabv3net import Deeplabv3Plus
from /unet3plus_backbones import UNet_3Plus_DeepSup_CGM_ResNet50, UNet_3Plus_DeepSup_CGM_DenseNet201, UNet_3Plus_DeepSup_CGM_EfficientNetB6


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
     '--config_file', help='path of config file.', required=True)
    args = parser.parse_args()
    
    # Load the config file
    with open( args.config_file , "r") as ymlfile:
       config_file = yaml.load(ymlfile, Loader=yaml.Loader)
       

    
    NUM_CLASSES = config_file["DATASET"]["NUM_CLASSES"]

    MODEL = config_file["MODEL"]["TYPE"]
    NET_BACKBONE = config_file["MODEL"]["NET_BACKBONE"]
    MODEL_path = config_file["MODEL"]["MODEL_PATH"] # "/home/yadirapr/Respositorio/Deeplabv3_pytorch/ResNet50_224_12_430_0001/epoch_400.pth"
    ONNX_path = config_file["MODEL"]["ONNX_PATH"]
    
    BATCH_SIZE = config_file["TRAIN"]["BATCH_SIZE"]
    initial_learning_rate = config_file["TRAIN"]["LEARNING_RATE"]
    num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]
    
    

    if MODEL == 'Deeplabv3Plus':
        
        model = Deeplabv3Plus(num_classes=NUM_CLASSES, network_backbone= NET_BACKBONE)
        model.load_state_dict(torch.load(MODEL_path, map_location={'cuda:1': 'cuda:0'}))
        model.eval()
    
        torch_input = torch.randn(1, 3, 224, 224)
    
        onnx_program = torch.onnx.export(model,torch_input,f=ONNX_path, opset_version= 17, do_constant_folding =True, input_names=["pixel_values"], output_names=["labels"])
        print("Done saving!")
        import onnx
        onnx_model = onnx.load(ONNX_path)
        onnx.checker.check_model(onnx_model)
    if MODEL == 'Unet3plus':
        

if __name__ == '__main__':
  main()
