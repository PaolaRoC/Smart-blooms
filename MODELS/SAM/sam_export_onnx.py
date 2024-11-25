# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:57:21 2024

@author: Paola
"""


from importlib import import_module
from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel
import torch
import warnings



# register model
ckpt= 'checkpoints/sam_vit_b_01ec64.pth'
sam, img_embedding_size = sam_model_registry['vit_b'](image_size=224,
                                                                num_classes=3,
                                                                checkpoint=ckpt,
                                                                pixel_mean=[0.485, 0.456, 0.406],
                                                                pixel_std=[0.229, 0.224, 0.225])

pkg = import_module('sam_lora_image_encoder')
net = pkg.LoRA_Sam(sam, 4)

lora_ckpt= "/homelocal/yadirapr_local/SAMed/output/Cyano_224_pretrain_vit_b_epo404_bs12_lr0.001/epoch_319.pth"
net.load_lora_parameters(lora_ckpt)
net.cuda()
net.eval()
#for i in net.named_parameters():
#    print(f"{i[0]} -> {i[1].device}")

# Export images encoder from SAM model to ONNX
torch_input = torch.randn(1, 3, 224, 224).cuda()

torch.onnx.export(net, (torch_input, True, 224 ),f="/homelocal/yadirapr_local/SAMed/output/sam_model_r4_001_12_319_bt64.onnx", opset_version= 17, do_constant_folding =True, export_params=True)
# dynamic_axes={'batched_input' : {0 : '-1'},'masks' : {0 : '-1'}}
"""
# Export masks decoder from SAM model to ONNX
onnx_model = SamOnnxModel(net.sam, return_single_mask=True,use_stability_score=False,return_extra_metrics=False)

              
dynamic_axes = {
    "point_coords": {1: "num_points"},
    "point_labels": {1: "num_points"},
}

for i in onnx_model.named_parameters():
    print(f"{i[0]} -> {i[1].device}")
    
embed_dim = net.sam.prompt_encoder.embed_dim
embed_size = net.sam.prompt_encoder.image_embedding_size
mask_input_size = [4 * x for x in embed_size]
print('embed_dim',embed_dim)
print('embed_size',embed_size)
print('mask_input_size',mask_input_size)

dummy_inputs = {
    "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
    "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
    "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
    "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
    "has_mask_input": torch.tensor([1], dtype=torch.float),
    #"orig_im_size": torch.tensor([224, 224], dtype=torch.float),
}

_ = onnx_model(**dummy_inputs)

output_names = ["iou_predictions", "low_res_masks"]
onnx_model_path= "/homelocal/yadirapr_local/SAMed/output/SAM_model_decoder_r16_0001_12_399.onnx"
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    with open(onnx_model_path, "wb") as f:
        torch.onnx.export(
            onnx_model,
            tuple(dummy_inputs.values()),
            f,
            export_params=True,
            verbose=False,
            opset_version=17,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        ) 
"""