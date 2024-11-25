# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 19:33:08 2024

@author: Paola
"""
import argparse
from ast import arg
import tensorrt as trt
from cuda import cudart

import tensorflow as tf
import torch
import numpy as np
import time
from tqdm import tqdm
import csv
import os
import sys
sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import common

print(trt.__version__)


import os
from glob import glob

seed=42
IMAGE_SIZE = 224
NUM_CLASSES = 4
DATA_DIR = "/home/yadirapr/Datos"


from torch.utils.data import Dataset
import os
from PIL import Image

class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, image_processor, train=True,transform=None):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            image_processor (SegFormerImageProcessor): image processor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.train = train
        self.transform = transform

        sub_path = "TRAIN" if self.train else "TEST"
        self.img_dir = os.path.join(self.root_dir, sub_path + "_img")
        self.ann_dir = os.path.join(self.root_dir, sub_path + "_mask_ids")

        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
          image_file_names.extend(files)
        self.images = sorted(image_file_names)

        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
          annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)
        #print( len(self.images))
        #print(len(self.annotations))
        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.image_processor(image, segmentation_map, return_tensors="pt")
        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension
        if self.transform is not None:
          encoded_inputs= self.transform(encoded_inputs)
        return encoded_inputs

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

val_transforms = A.Compose([
    A.Normalize(p=1),
    ToTensorV2(),
])


def preprocess_val(examples):
  img = np.array(examples["pixel_values"].permute(1,2,0))
  mask_true = np.array(examples["labels"])
  augmented = val_transforms(image=img, mask=mask_true)

  examples["pixel_values"]  = augmented['image']
  examples["labels"]  = augmented['mask']
  return examples


#######################################################################################
def archivo_existe(archivo_csv):
    if os.path.exists(archivo_csv):
        print(f"El archivo CSV '{archivo_csv}' existe.")
        encabezado = tiene_encabezado(archivo_csv)
        print('Tiene encabesado: ', encabezado)
        return encabezado

    else:
        print(f"El archivo CSV '{archivo_csv}' no existe.")
        return False


def tiene_encabezado(archivo_csv):
    with open(archivo_csv, 'r', newline='') as file:
        reader = csv.reader(file)
        primera_linea = next(reader, None)

        if primera_linea:
            return True
        else:
            return False 

##############################################################

class TensorRTInfer:
    """
    Implements inference for the Model TensorRT engine.
    """

    def __init__(self, engine_path):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        return self.outputs[0]["shape"], self.outputs[0]["dtype"]

    def infer(self,BATCH_SIZE, batch,N_W_RUN,N_RUN,test_batches_label):
        N_warmup_run = N_W_RUN
        N_run = N_RUN
        elapsed_time = []
        shape_out, dtype_out = self.output_spec() 
        n_labels= 4
        #output = np.zeros(*self.output_spec()) # esto me crea una array del shape ((32, 224, 224, 4)) porque el modelo fue creado con ese tamaño en onnx. 
        #segformer
        output = np.zeros((test_batches_label.shape[0],n_labels,224,224), dtype_out) 

        # Process I/O and execute the network
        common.memcpy_host_to_device(self.inputs[0]['allocation'], np.ascontiguousarray(batch))

        for _ in range(N_warmup_run):
            self.context.execute_v2(self.allocations)

        for _ in tqdm(range(N_run)):
            start_time = time.time()

            self.context.execute_v2(self.allocations)
        
            end_time = time.time()
            elapsed_time = np.append(elapsed_time, end_time - start_time)

        common.memcpy_device_to_host(output, self.outputs[0]["allocation"])
        
      
        # Process the results
        IMG_S = N_run * BATCH_SIZE / elapsed_time.sum()
        logits = tf.transpose(output, [0, 2, 3, 1])
        predictions = tf.math.argmax(logits, axis=-1)

        #test_batches_label = np.squeeze(test_batches_label)
     
        keras_accuracy = [tf.keras.metrics.MeanIoU(num_classes=4, sparse_y_true = True, sparse_y_pred = True),
                    tf.keras.metrics.IoU(num_classes=4, target_class_ids=[1], sparse_y_true = True, sparse_y_pred = True),
                    tf.keras.metrics.IoU(num_classes=4, target_class_ids=[2], sparse_y_true = True, sparse_y_pred = True),
                    tf.keras.metrics.IoU(num_classes=4, target_class_ids=[3], sparse_y_true = True, sparse_y_pred = True),
                    tf.keras.metrics.IoU(num_classes=4, target_class_ids=[0], sparse_y_true = True, sparse_y_pred = True),
                    tf.keras.metrics.Accuracy()]
        

                           
        keras_accuracy[0](predictions, test_batches_label)
        keras_accuracy[1](predictions, test_batches_label)
        keras_accuracy[2](predictions, test_batches_label)
        keras_accuracy[3](predictions, test_batches_label)
        keras_accuracy[4](predictions, test_batches_label)
        keras_accuracy[5].update_state(test_batches_label,predictions)

        return [round((keras_accuracy[5].result().numpy()*100),3), round((keras_accuracy[0].result().numpy()*100),3), round((keras_accuracy[1].result().numpy()*100),3), round((keras_accuracy[2].result().numpy()*100),3), round((keras_accuracy[3].result().numpy()*100),3), round((keras_accuracy[4].result().numpy()*100),3), BATCH_SIZE, round((min(elapsed_time)*1000),1), round((max(elapsed_time)*1000),1), round((elapsed_time.mean()*1000),1), round(IMG_S)]

    
        

############################################################

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of model.', required=True)
  parser.add_argument(
      '--data_dir', help='Path of data.', required=True)
  parser.add_argument(
     '--batch_size', type=int, help='Size of batch', required=True)
  parser.add_argument(
     '--N_warmup_run', type=int, help= 'Number of warmup run', required=True)
  parser.add_argument(
     '--N_run', type=int,  help='Number run', required=True)
  parser.add_argument(
      '--name_csv', help='Name of csv file.', required=True)

  args = parser.parse_args()

  
  BATCH_SIZE = args.batch_size
  N_W_RUN = args.N_warmup_run
  N_RUN = args.N_run 
  print(BATCH_SIZE)
  DATA_DIR = args.data_dir
  archivo_csv = args.name_csv
  

  from transformers import SegformerImageProcessor
    
  root_dir = DATA_DIR
  
  image_processor = SegformerImageProcessor(size = {"height": 224, "width": 224},do_reduce_labels=False, do_rescale= False, do_normalize= False)
    
  valid_dataset = SemanticSegmentationDataset(root_dir=root_dir, image_processor=image_processor, train=False, transform=preprocess_val)
    
  print("Number of validation examples:", len(valid_dataset))
    
  from torch.utils.data import DataLoader
    
  valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, drop_last =True)
  device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    


  field = ["Modelo", "Placa", "Precision", "OA", "mIoU", "AguaIoU", "CyanoIoU", "Roca", "Fondo", "Batch", "Tiempo min", "Tiempo max", "Tiempo mean", "Img/s"]
  
  info_takes= []
  n_len_batch = round(len(valid_dataset)/BATCH_SIZE)
  print(n_len_batch)

  trt_infer = TensorRTInfer(args.model)
  
  for idx, batch in enumerate(valid_dataloader):
      pixel = batch["pixel_values"]
      lab = batch["labels"]
      info_takes.append(trt_infer.infer(BATCH_SIZE,pixel,N_W_RUN,N_RUN, lab))



  matriz_info = np.array(info_takes)
  matriz_info[matriz_info == 0] = np.nan

  nb_precision = args.model.split("_")[-2]
  nb_placa = args.model.split("_")[-1]
  nb_modelo = args.model.split("_")[2]
  verf_archivo = archivo_existe(archivo_csv)
  if verf_archivo == True:
    with open(archivo_csv,'a') as f1:
        writer= csv.writer(f1, delimiter='\t',lineterminator='\n',)
        writer.writerow([nb_modelo, nb_placa, nb_precision, round(np.nanmean(matriz_info[:,0]),3),round(np.nanmean(matriz_info[:,1]),3), round(np.nanmean(matriz_info[:,2]),3), round(np.nanmean(matriz_info[:,3]),3),  round(np.nanmean(matriz_info[:,4]),3), round(np.nanmean(matriz_info[:,5]),3), BATCH_SIZE, round(np.nanmean(matriz_info[:,7]),1), round(np.nanmean(matriz_info[:,8]),1), round(np.nanmean(matriz_info[:,9]),1), round(np.nanmean(matriz_info[:,10]))])
  else:
    with open(archivo_csv, 'w') as f1:
        writer= csv.writer(f1, delimiter='\t',lineterminator='\n',)
        writer.writerow(field)
        writer.writerow([nb_modelo, nb_placa, nb_precision, round(np.nanmean(matriz_info[:,0]),3),round(np.nanmean(matriz_info[:,1]),3), round(np.nanmean(matriz_info[:,2]),3), round(np.nanmean(matriz_info[:,3]),3),  round(np.nanmean(matriz_info[:,4]),3), round(np.nanmean(matriz_info[:,5]),3), BATCH_SIZE, round(np.nanmean(matriz_info[:,7]),1), round(np.nanmean(matriz_info[:,8]),1), round(np.nanmean(matriz_info[:,9]),1), round(np.nanmean(matriz_info[:,10]))])



if __name__ == '__main__':
  main()