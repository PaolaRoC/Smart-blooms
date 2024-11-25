# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:06:16 2024

@author: Paola
"""

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
# Cargar el modelo
model_path = "/homelocal/yadirapr_local/SAMed/output/sam_model_r4_001_12_319_bt64.onnx"
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
updated_model_path = "/homelocal/yadirapr_local/SAMed/output/sam_model_r4_001_12_319_bt64_update.onnx"
onnx.save(model, updated_model_path)
