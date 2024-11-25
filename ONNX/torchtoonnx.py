# -*- coding: utf-8 -*-

import argparse
import yaml
import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..MODELS.Deeplabv3plus.deeplabv3net import Deeplabv3Plus
from ..MODELS.Unet3plus.unet3plus_backbones import UNet_3Plus_DeepSup_CGM_ResNet50, UNet_3Plus_DeepSup_CGM_DenseNet201, UNet_3Plus_DeepSup_CGM_EfficientNetB6
from transformers import SegformerForSemanticSegmentation
from importlib import import_module
from ..MODELS.SAM.segment_anything import sam_model_registry

from ..MODELS.SAM.sam_lora_image_encoder import LoRA_Sam
import onnx
from onnx import helper, shape_inference

def remove_input(model, input_name):
    graph = model.graph
    new_inputs = [input for input in graph.input if input.name != input_name]
    graph.ClearField('input')
    graph.input.extend(new_inputs)
    print(f"Input '{input_name}' has been removed.")

def remove_output(model, output_name):
    graph = model.graph
    new_outputs = [output for output in graph.output if output.name != output_name]
    graph.ClearField('output')
    graph.output.extend(new_outputs)
    print(f"Output '{output_name}' has been removed.")
    
def remove_node_by_name(model, node_name):
    graph = model.graph
    node_to_remove = None
    for node in graph.node:
        if node.name == node_name:
            node_to_remove = node
            break

    if node_to_remove:
        graph.node.remove(node_to_remove)
        print(f"Node '{node_name}' has been removed.")
    else:
        print(f"Node '{node_name}' not found in the model.")

    inferred_model = shape_inference.infer_shapes(model)
    return inferred_model

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
    
    DYNAMIC_SHAPE = config_file["MODEL"]["DYNAMIC_SHAPE"]
    

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
        
        if NET_BACKBONE == 'ResNet50':
            net = UNet_3Plus_DeepSup_CGM_ResNet50(in_channels=3, n_classes=NUM_CLASSES)
            net.load_state_dict(torch.load(MODEL_path, map_location={'cuda:1': 'cuda:0'}), strict=False)
            net.eval()
            torch_input = torch.randn(1, 3, 224, 224)

            onnx_program = torch.onnx.export(net,torch_input,f=ONNX_path, opset_version= 17, do_constant_folding =True, input_names=["input"], output_names=["out_d0", "out_d1", "out_d2", "out_d3", "out_d4"])

            print("Done saving!")
            import onnx
            onnx_model = onnx.load(ONNX_path)
            onnx.checker.check_model(onnx_model)
        elif NET_BACKBONE == 'DenseNet201':
            net = UNet_3Plus_DeepSup_CGM_DenseNet201(in_channels=3, n_classes=NUM_CLASSES)
            net.load_state_dict(torch.load(MODEL_path, map_location={'cuda:1': 'cuda:0'}), strict=False)
            net.eval()
            torch_input = torch.randn(1, 3, 224, 224)
            
            onnx_program = torch.onnx.export(net,torch_input,f=ONNX_path, opset_version= 17, do_constant_folding =True, input_names=["input"], output_names=["out_d0", "out_d1", "out_d2", "out_d3", "out_d4"])
            
            print("Done saving!")
            import onnx
            onnx_model = onnx.load(ONNX_path)
            onnx.checker.check_model(onnx_model)
        elif NET_BACKBONE == 'EfficientNetB6':
            net = UNet_3Plus_DeepSup_CGM_EfficientNetB6(in_channels=3, n_classes=NUM_CLASSES)
            net.load_state_dict(torch.load(MODEL_path, map_location={'cuda:1': 'cuda:0'}), strict=False)
            net.eval()
            torch_input = torch.randn(1, 3, 224, 224)
            
            onnx_program = torch.onnx.export(net,torch_input,f=ONNX_path, opset_version= 17, do_constant_folding =True, input_names=["input"], output_names=["out_d0", "out_d1", "out_d2", "out_d3", "out_d4"])

            print("Done saving!")
            import onnx
            onnx_model = onnx.load(ONNX_path)
            onnx.checker.check_model(onnx_model)
        else:
            print("Choose a backbone: ResNet50, EfficientNetB6, DenseNet201")
            exit()
        
    if MODEL == 'SegFormer':
        torch_model = SegformerForSemanticSegmentation.from_pretrained(MODEL_path)
        torch_input = torch.zeros(1, 3, 224, 224).cuda()

        onnx_program = torch.onnx.export(torch_model.cuda(),torch_input,f=ONNX_path, input_names=["pixel_values"], output_names=["logits"],opset_version= 17, do_constant_folding =True )
        #onnx_program.save("/homelocal/yadirapr_local/modelo/ONNX/my_semseg.onnx")

        import onnx
        onnx_model = onnx.load(ONNX_path)
        onnx.checker.check_model(onnx_model)
    if MODEL == "SAMv1":
        ckpt= './MODELS/SAM/checkpoints/sam_vit_b_01ec64.pth'
        sam, img_embedding_size = sam_model_registry['vit_b'](image_size=224,
                                                                num_classes=NUM_CLASSES-1,
                                                                checkpoint=ckpt,
                                                                pixel_mean=[0.485, 0.456, 0.406],
                                                                pixel_std=[0.229, 0.224, 0.225])

        net = LoRA_Sam(sam, 4)
        
        lora_ckpt= MODEL_path
        net.load_lora_parameters(lora_ckpt)
        net.cuda()
        net.eval()
        
        torch_input = torch.randn(1, 3, 224, 224).cuda()
        if DYNAMIC_SHAPE ==True:
            torch.onnx.export(net, (torch_input, True, 224 ),f=ONNX_path, opset_version= 17, do_constant_folding =True, export_params=True, dynamic_axes={'batched_input' : {0 : '-1'},'masks' : {0 : '-1'}})

        else:
            torch.onnx.export(net, (torch_input, True, 224 ),f=ONNX_path, opset_version= 17, do_constant_folding =True, export_params=True)

        # Cargar el modelo
        model_path = ONNX_path
        model = onnx.load(model_path)
        
        # Especificar los nombres de las entradas y salidas que deseas eliminar
        input_name_to_remove = "image_size"
        nodes_i =["/sam/Unsqueeze_1", "/sam/Unsqueeze_2", "/sam/Unsqueeze_3", "/sam/Unsqueeze_4", "/sam/Concat_2", 
                                "/sam/Cast_1", ]
        #output_name_to_remove = "9731"
        output_name_to_remove = "9727"
        #nodes_o=["/sam/Resize_1", "/sam/Concat_3", "/sam/Slice_4", "/sam/Shape_1", "/sam/Slice_3", "/sam/Slice_2",
        #         "/sam/Resize","/sam/Concat_1", "/sam/Slice_1", "/sam/Shape"]
        nodes_o=["/sam/Shape", "/sam/Slice_1", "/sam/Concat_1", "/sam/Resize", "/sam/Slice_2", "/sam/Slice_3", "/sam/Shape_1", "/sam/Slice_4", "/sam/Concat_3", "/sam/Resize_1"]
        # Eliminar la entrada
        remove_input(model, input_name_to_remove)
        
        # Eliminar la salida
        remove_output(model, output_name_to_remove)
        
        # Eliminar restos de nodos
        
        for i in nodes_i:
            remove_node_by_name(model, i)
        
        for i in nodes_o:
            remove_node_by_name(model, i)    
        
        # Validar el modelo
        onnx.checker.check_model(model)
        
        # Guardar el modelo actualizado
        updated_model_path = ONNX_path
        onnx.save(model, updated_model_path)
    
    else:
            print("Choose a model: Deeplabv3Plus,Unet3plus, SegFormer,EfficientNetB6, SAMv1")
            exit()
        

if __name__ == '__main__':
  main()
