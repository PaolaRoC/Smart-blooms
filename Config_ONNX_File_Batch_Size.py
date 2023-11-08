# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:11:33 2023

@author: Paola
"""


import onnx
BATCH_SIZE =32
onnx_model = onnx.load_model("/homelocal/yadirapr_local/Modelos/ONNX/tempXcep.onnx")
inputs = onnx_model.graph.input
for input in inputs:
    dim1 = input.type.tensor_type.shape.dim[0]
    dim1.dim_value = BATCH_SIZE

model_name = "/homelocal/yadirapr_local/Modelos/ONNX/Xception.onnx"
onnx.save_model(onnx_model, model_name)
print("Done saving!")

onnx_model = onnx.load_model("/homelocal/yadirapr_local/Modelos/ONNX/tempResNet.onnx")
inputs = onnx_model.graph.input
for input in inputs:
    dim1 = input.type.tensor_type.shape.dim[0]
    dim1.dim_value = BATCH_SIZE

model_name = "/homelocal/yadirapr_local/Modelos/ONNX/ResNet50V2.onnx"
onnx.save_model(onnx_model, model_name)
print("Done saving!")