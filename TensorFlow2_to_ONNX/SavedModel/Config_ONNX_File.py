# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:11:33 2023

@author: Paola
"""
import onnx
import argparse

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
     '--load_tempmodel', help='load the temporary ONNX model.', required=True)
    parser.add_argument(
    '--save_onnxmodel', help='Save the ONNX model', required=True)
    args = parser.parse_args()
    

    BATCH_SIZE =32
    onnx_model = onnx.load_model(args.load_tempmodel)
    inputs = onnx_model.graph.input
    for input in inputs:
        dim1 = input.type.tensor_type.shape.dim[0]
        dim1.dim_value = BATCH_SIZE
    
    model_name = args.save_onnxmodel
    onnx.save_model(onnx_model, model_name)
    print("Done saving!")


if __name__ == '__main__':
  main()
