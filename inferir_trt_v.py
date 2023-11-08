# -*- coding: utf-8 -*-
"""inferir_trt.ipynb
"""

import argparse
from ast import arg
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm
import csv
print(trt.__version__)


import os
from glob import glob
BATCH_SIZE = 4
seed=42
IMAGE_SIZE = 224
NUM_CLASSES = 4
DATA_DIR = "/home/yadirapr/Datos"
NUM_TRAIN_IMAGES= 323
NUM_TEST_IMAGES = 79

train_images = sorted(glob(os.path.join(DATA_DIR, "train_img/*")))[:NUM_TRAIN_IMAGES]
train_masks = sorted(glob(os.path.join(DATA_DIR, "train_mask_ids/*")))[:NUM_TRAIN_IMAGES]

val_images = sorted(glob(os.path.join(DATA_DIR, "test_img/*")))[:NUM_TEST_IMAGES]
val_masks = sorted(glob(os.path.join(DATA_DIR, "test_mask_ids/*")))[:NUM_TEST_IMAGES]

def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)

    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = tf.keras.applications.resnet50.preprocess_input(image)
    return image


def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def data_generator(image_list, mask_list):

    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data,num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


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
            print(primera_linea)
            return True
        else:
            return False 

##############################################################

#target_dtype = np.int8

#test_batches_img = test_batches_img.astype(target_dtype)
#print(test_batches_img.shape)

#f = open("/home/yadirapr/codigo/MobileNetV3Large_newlayers_9_36_int8.trt", "rb")
#runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

#engine = runtime.deserialize_cuda_engine(f.read())
#context = engine.create_execution_context()

#########################################################

#output = np.empty([BATCH_SIZE, 224,224,4], dtype = target_dtype) # Need to set output dtype to FP16 to enable FP16

# Allocate device memory
#d_input = cuda.mem_alloc(1 * test_batches_img.nbytes)
#d_output = cuda.mem_alloc(1 * output.nbytes)

#bindings = [int(d_input), int(d_output)]

#stream = cuda.Stream()

#########################################################


def allocate_buffers(engine,test_batches_img,batch_size,target_dtype, DTYPE ):
    h_input =  np.array(test_batches_img)
    h_input = h_input.astype(target_dtype)
    n_labels= 4
    #cuda.pagelocked_empty(batch_size * (trt.volume(engine.get_binding_shape(0))), dtype=trt.nptype(DTYPE))
    h_output = cuda.pagelocked_empty((batch_size, 224, 224,n_labels), dtype=trt.nptype(DTYPE))
    #cuda.pagelocked_empty(batch_size * (trt.volume(engine.get_binding_shape(1))), dtype=trt.nptype(DTYPE))
    d_input = cuda.mem_alloc( 1 * h_input.nbytes)
    d_output = cuda.mem_alloc(1 * h_output.nbytes)
    bindings= [int(d_input), int(d_output)]
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output,bindings,stream

def predict(context, h_input, d_input, h_output, d_output,bindings, stream,N_W_RUN,N_RUN,test_batches_label ): # result gets copied into output


    N_warmup_run = N_W_RUN
    N_run = N_RUN
    elapsed_time = []
    

    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, h_input, stream)

    for _ in range(N_warmup_run):
        context.execute_async_v2(bindings, stream.handle, None)

    for _ in tqdm(range(N_run)):
        start_time = time.time()
        
        # Execute model
        context.execute_async_v2(bindings, stream.handle, None)
        
        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)
    
    IMG_S = N_run * BATCH_SIZE / elapsed_time.sum()
    latency_mean = elapsed_time.mean() / N_run * 1000
    
    # Transfer predictions back
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Syncronize threads
    stream.synchronize()
    #quizas tendria que poner aqui un time 

    # Evaluation
    
    predictions = np.argmax(h_output, axis=3)

    print(predictions[0])
    test_batches_label = np.squeeze(test_batches_label)
    keras_accuracy = [tf.keras.metrics.MeanIoU(num_classes=4, sparse_y_true = True, sparse_y_pred = True),
                tf.keras.metrics.IoU(num_classes=4, target_class_ids=[1], sparse_y_true = True, sparse_y_pred = True),
                tf.keras.metrics.IoU(num_classes=4, target_class_ids=[2], sparse_y_true = True, sparse_y_pred = True)]
    
    keras_accuracy[0](predictions, test_batches_label)
    keras_accuracy[1](predictions, test_batches_label)
    keras_accuracy[2](predictions, test_batches_label)

    return [round((keras_accuracy[0].result().numpy()*100),3), round((keras_accuracy[1].result().numpy()*100),3), round((keras_accuracy[2].result().numpy()*100),3), BATCH_SIZE, round((min(elapsed_time)*1000),1), round((max(elapsed_time)*1000),1), round((elapsed_time.mean()*1000),1), round(IMG_S)]



############################################################

#print("Warming up...")

#trt_predictions = predict(test_batches_img).astype(np.float32)
#trt_predictions = np.argmax(trt_predictions, axis=3)

#test_batches_label = np.squeeze(test_batches_label)

#keras_accuracy = [tf.keras.metrics.MeanIoU(num_classes=4, sparse_y_true = True, sparse_y_pred = True),
#                tf.keras.metrics.IoU(num_classes=4, target_class_ids=[1], sparse_y_true = True, sparse_y_pred = True),
#                tf.keras.metrics.IoU(num_classes=4, target_class_ids=[2], sparse_y_true = True, sparse_y_pred = True)]
#keras_accuracy[0](trt_predictions, test_batches_label)
#keras_accuracy[1](trt_predictions, test_batches_label)
#keras_accuracy[2](trt_predictions, test_batches_label)

#print("Done warming up!")
#print("Raw model mIOU: {:.3%}".format(keras_accuracy[0].result()))
#print("Raw model Agua IOU: {:.3%}".format(keras_accuracy[1].result()))
#print("Raw model Cyano IOU: {:.3%}".format(keras_accuracy[2].result()))

#############################################################

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
     '--precison', help = 'type: float32, float16 or int8', required= True)
  parser.add_argument(
      '--name_csv', help='Name of csv file.', required=True)

  args = parser.parse_args()

  
  BATCH_SIZE = args.batch_size
  N_W_RUN = args.N_warmup_run
  N_RUN = args.N_run 
  print(BATCH_SIZE)
  DATA_DIR = args.data_dir
  archivo_csv = args.name_csv
  NUM_TEST_IMAGES = 32
  if args.precison == 'int8':
    DTYPE = trt.int8
    target_dtype = np.int8
  if args.precison == 'float16':
    DTYPE = trt.float16 
    target_dtype = np.float16
  else:
    DTYPE = trt.float32 
    target_dtype = np.float32
      



  val_images = sorted(glob(os.path.join(DATA_DIR, "test_img/*")))[:NUM_TEST_IMAGES]
  val_masks = sorted(glob(os.path.join(DATA_DIR, "test_mask_ids/*")))[:NUM_TEST_IMAGES]

  
  val_dataset = data_generator(val_images, val_masks)

  test_batches = (
    val_dataset
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE))
  

  field = ["Placa", "Precision","mIoU", "AguaIoU", "CyanoIoU", "Batch", "Tiempo min", "Tiempo max", "Tiempo mean", "Img/s"]
  
  info_takes= []
  n_len_batch = round(len(val_images)/BATCH_SIZE)
  print(n_len_batch)

  f = open(args.model, "rb")
  runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
  engine = runtime.deserialize_cuda_engine(f.read())
  context = engine.create_execution_context()
  
  for images, masks in test_batches.take(n_len_batch):
    print('siguente lote')
    h_input, d_input, h_output, d_output,bindings,stream = allocate_buffers(engine,images,BATCH_SIZE,target_dtype, DTYPE )
    info_takes.append(predict(context, h_input, d_input, h_output, d_output,bindings, stream,N_W_RUN,N_RUN, masks ))



  matriz_info = np.array(info_takes)

  nb_precision = args.model.split("_")[-2]
  nb_placa = args.model.split("_")[-1]
  verf_archivo = archivo_existe(archivo_csv)
  if verf_archivo == True:
    with open(archivo_csv,'a') as f1:
        writer= csv.writer(f1, delimiter='\t',lineterminator='\n',)
        writer.writerow([nb_placa, nb_precision, round(np.mean(matriz_info[:,0]),3), round(np.mean(matriz_info[:,1]),3), round(np.mean(matriz_info[:,2]),3), BATCH_SIZE, round(np.mean(matriz_info[:,4]),1), round(np.mean(matriz_info[:,5]),1), round(np.mean(matriz_info[:,6]),1), round(np.mean(matriz_info[:,7]))])
  else:
    with open(archivo_csv, 'w') as f1:
        writer= csv.writer(f1, delimiter='\t',lineterminator='\n',)
        writer.writerow(field)
        writer.writerow([nb_placa, nb_precision, round(np.mean(matriz_info[:,0]),3), round(np.mean(matriz_info[:,1]),3), round(np.mean(matriz_info[:,2]),3), BATCH_SIZE, round(np.mean(matriz_info[:,4]),1), round(np.mean(matriz_info[:,5]),1), round(np.mean(matriz_info[:,6]),1), round(np.mean(matriz_info[:,7]))])



if __name__ == '__main__':
  main()