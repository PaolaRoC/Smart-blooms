# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:11:50 2024

@author: Paola
"""

###
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unet3plus_backbones import UNet_3Plus_DeepSup_CGM_ResNet50, UNet_3Plus_DeepSup_CGM_DenseNet201, UNet_3Plus_DeepSup_CGM_EfficientNetB6
import csv  
###
import argparse
import yaml
from torch.utils.data import Dataset
import os
from PIL import Image
import time 
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
        list_dir= '/scratch/yadirapr/sam2/text'
        split_train = 'train_tot_nou_water'
        split_test = 'validation_dataset2'
        #split_test = 'test19'
        sample_list = open(os.path.join(list_dir, split_train +'.txt')).readlines() if self.train else open(os.path.join(list_dir, split_test +'.txt')).readlines()

        sub_path = "TRAIN" if self.train else "TEST"
        self.img_dir = os.path.join(self.root_dir, sub_path + "_img")
        self.ann_dir = os.path.join(self.root_dir, sub_path + "_mask_ids")

    
        
        self.images = sorted(sample_list)


        self.annotations = sorted(sample_list)
        #print( len(self.images))
        #print(len(self.annotations))
        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.img_dir, self.images[idx].strip('\n') + '.png'))
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx].strip('\n') + '.png'))

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.image_processor(image, segmentation_map, return_tensors="pt")
        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension
        if self.transform is not None:
          encoded_inputs= self.transform(encoded_inputs)
        return encoded_inputs
        


import albumentations as A
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

def compute_metrics(y_true, y_pred):

  class_wise_iou = []
  class_wise_dice_score = []
  list_labels_id = []

  smoothening_factor = 0.00001

  list_id= list(np.unique(y_true))
  dicc_id= {0: 'Background',1:'Water', 2:'Blooms', 3:'Rock'}


  for i in range(0,4): #n_labels=4 --> i= 0,1,2,3
      if i in list_id:
          
          intersection = np.sum((y_pred == i) * (y_true == i))
          y_true_area = np.sum((y_true == i))
          y_pred_area = np.sum((y_pred == i))
          combined_area = y_true_area + y_pred_area
        
          iou = (intersection + smoothening_factor) / (combined_area - intersection + smoothening_factor)
          class_wise_iou.append(iou)
        
          dice_score =  (2 *(intersection + smoothening_factor) / (combined_area + smoothening_factor))
          class_wise_dice_score.append(dice_score)
          list_labels_id.append(dicc_id[i])
      else: 
          iou= 0
          class_wise_iou.append(iou)
          dice_score = 0
          class_wise_dice_score.append(dice_score)
      
  return class_wise_iou, class_wise_dice_score, list_labels_id
 
def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


import matplotlib.pyplot as plt 

def plot_samples_matplotlib(display_list, display_string, idx ,outimg_path, figsize=(1, 3)):
    fig, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    title = ["Input Image", "True Mask", "Predicted Mask"]
    for i in range(len(display_list)):

        if display_list[i].shape[-1] == 3:
            axes[i].title.set_text(title[i])

            axes[i].imshow(display_list[i])

        else:
            axes[i].imshow(display_list[i])

    fig.text(0.5, 0.25,display_string, horizontalalignment='center',
     verticalalignment='bottom')
    #plt.show()
    result_path = os.path.join( outimg_path, "./unet3plus_pred_vs_ground_img_"+ str(idx) +".png")
    plt.savefig(result_path)

def evaluation_unet3plus(model: nn.Module, device, valid_dataloader, metric,NUM_CLASSES, outimg_path):
    
    model.to(device)
    model.eval()
    
    info_p_cat_iou= []
    info_mean_iou = 0.0
    info_overall_acc= 0.0
    elapsed_time = []
    
    plots_pred = []
    plots_img = []
    plots_label = []
    for idx, batch in enumerate(valid_dataloader):
      pixel = batch["pixel_values"].to(device)
      lab = batch["labels"].to(device)
      # forward pass
      with torch.no_grad():
        start_time = time.time()
    
        outputs = model(pixel)
        end_time = time.time()
      elapsed_time = np.append(elapsed_time, end_time - start_time)
      #logits = outputs.logits.cpu()
     
      predicted_segmentation_map = outputs[0].argmax(dim=1)
      
      plots_pred.append(predicted_segmentation_map.cpu().numpy())
      plots_img.append(pixel.permute(0,2,3,1).cpu().numpy())
      plots_label.append(lab.cpu().numpy())
      
      metrics = metric._compute(
                      predictions=[predicted_segmentation_map.cpu().numpy()],
                      references=[lab.cpu().numpy()],
                      num_labels=NUM_CLASSES,
                      ignore_index=255,
                      reduce_labels=False, # we've already reduced the labels ourselves
                  )
      info_mean_iou +=  np.nan_to_num(metrics['mean_iou'])
      info_overall_acc += np.nan_to_num(metrics['overall_accuracy'])
      #info_p_cat_iou += np.nan_to_num(metrics['per_category_iou'])
      info_p_cat_iou.append(np.nan_to_num(metrics['per_category_iou']))
    
    
    info_p_cat_iou_array = np.array(info_p_cat_iou)
    info_p_cat_iou_array[info_p_cat_iou_array == 0] = np.nan
    
    
    #Plots

    random_img = np.random.choice(range(len(valid_dataloader)), 3, replace=False)
    random_img= [1,4,7]
    colormap = np.array([[0,0,0],[39,206,215],[102,255,102],[255,153,0]])
    colormap = colormap.astype(np.uint8)
    
    for i in range(0,len(plots_img)):
        img= np.squeeze(np.array(plots_img[i]))
        label = np.squeeze(plots_label[i]).astype(np.uint8)
        prediction= np.squeeze(plots_pred[i]).astype(np.uint8)
    
        prediction_colormap = decode_segmentation_masks(prediction, colormap, NUM_CLASSES)
        ground_truth_colormap = decode_segmentation_masks(label, colormap, NUM_CLASSES)
        
        iou_list, dice_score_list, labels_id = compute_metrics(label,prediction)
        metrics_by_id = [(idx, iou, dice_score) for i, (idx,iou, dice_score) in enumerate(zip(iou_list, dice_score_list, labels_id)) if iou > 0.0]
        display_string_list = ["{}: IoU: {} Dice Score: {}".format(idx, iou, dice_score) for iou, dice_score, idx in metrics_by_id]
        display_string = "\n\n".join(display_string_list)
        plot_samples_matplotlib(
            [img, ground_truth_colormap,prediction_colormap], display_string, i, outimg_path, figsize=(18, 14)
        )


    # Results     
    
    IMG_S = len(valid_dataloader) / elapsed_time.sum()
    
    IMG_S = round(IMG_S,2)
    
    
    time_min = round((min(elapsed_time)*1000),1) 
    time_max = round((max(elapsed_time[1:])*1000),1)
    time_mean = round((elapsed_time[1:].mean()*1000),1)
    
    info_p_cat_iou_fondo=round((np.nanmean(info_p_cat_iou_array[:,0]))*100,3)
    info_p_cat_iou_agua=round((np.nanmean(info_p_cat_iou_array[:,1]))*100,3)
    info_p_cat_iou_ciano=round((np.nanmean(info_p_cat_iou_array[:,2]))*100,3)

    info_p_cat_iou_roca=round((np.nanmean(info_p_cat_iou_array[:,3]))*100,3)
    array_info_p_cat_iou = [info_p_cat_iou_agua,info_p_cat_iou_ciano,info_p_cat_iou_roca,info_p_cat_iou_fondo]
    info_mean_iou = round((info_mean_iou/ len(valid_dataloader))*100,3)
    info_overall_acc = round((info_overall_acc/ len(valid_dataloader))*100,3)
    print(" Categoria:",  array_info_p_cat_iou)
    print("MIoU: ", info_mean_iou)
    print("OA:", info_overall_acc)
    
    print("f/s:", IMG_S)
    print("Time_min: ", time_min)
    print("Time_max: ", time_max)
    print("Time_mean: ", time_mean)
    

    encabezados = ['OA', 'mIoU', 'Water', 'Cyano', 'Rock', 'Background', 'Time_min', 'Time_max', 'Time_mean', 'f/s']
    
 
    nombre_archivo = '/scratch/yadirapr/Respositorio/resultados_cpu.csv'
    verf_archivo = archivo_existe(nombre_archivo)
    if verf_archivo == True:
              
  
        with open(nombre_archivo, mode='a', newline='') as archivo:
            escritor_csv = csv.writer(archivo, delimiter='\t',lineterminator='\n',)
        
            # Escribir los datos de la matriz
            escritor_csv.writerows([[info_overall_acc,info_mean_iou,info_p_cat_iou_agua,info_p_cat_iou_ciano, info_p_cat_iou_roca,info_p_cat_iou_fondo,time_min,time_max,time_mean,  IMG_S ]])
                   
    else:

        with open(nombre_archivo, mode='w', newline='') as archivo:
            escritor_csv = csv.writer(archivo, delimiter='\t',lineterminator='\n',)
        

            escritor_csv.writerow(encabezados)
        

            escritor_csv.writerows([[info_overall_acc,info_mean_iou,info_p_cat_iou_agua,info_p_cat_iou_ciano, info_p_cat_iou_roca,info_p_cat_iou_fondo,time_min,time_max,time_mean,  IMG_S ]])
            
    return

def print_trainable_parameters(model):
        trainable_params, all_param = 0, 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")


from transformers import SegformerImageProcessor

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
     '--config_file', help='path of config file.', required=True)
    args = parser.parse_args()
    
    # Load the config file
    with open( args.config_file , "r") as ymlfile:
       config_file = yaml.load(ymlfile, Loader=yaml.Loader)
       

    
    root_dir = config_file["DATASET"]["DATA_PATH"] # "./Datos_originales/"
    NUM_CLASSES = config_file["DATASET"]["NUM_CLASSES"]

    NET_BACKBONE = config_file["MODEL"]["NET_BACKBONE"]
    MODEL_path = config_file["MODEL"]["MODEL_PATH"] # "./ResNet50_224_12_430_0001/epoch_400.pth"
 
    PREDICTED_MASK_PATH = config_file["OUTPUT"]["PREDICTED_MASK_PATH"]


    image_processor = SegformerImageProcessor(size = {"height": 224, "width": 224},do_reduce_labels=False, do_rescale= False, do_normalize= False)
    
    valid_dataset = SemanticSegmentationDataset(root_dir=root_dir, image_processor=image_processor, train=False, transform=preprocess_val)
    
    print("Number of validation examples:", len(valid_dataset))
    
    from torch.utils.data import DataLoader
    
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, drop_last= True)
    
    len(valid_dataloader)
    
    
    import evaluate
    
    metric = evaluate.load("mean_iou")
    
    if NET_BACKBONE == 'ResNet50':
        net = UNet_3Plus_DeepSup_CGM_ResNet50(in_channels=3, n_classes=NUM_CLASSES)
        net.load_state_dict(torch.load(MODEL_path, map_location={'cuda:1': 'cuda:0'}), strict=False)
        
    elif NET_BACKBONE == 'DenseNet201':
        net = UNet_3Plus_DeepSup_CGM_DenseNet201(in_channels=3, n_classes=NUM_CLASSES)
        net.load_state_dict(torch.load(MODEL_path, map_location={'cuda:1': 'cuda:0'}), strict=False)
        
    elif NET_BACKBONE == 'EfficientNetB6':
        net = UNet_3Plus_DeepSup_CGM_EfficientNetB6(in_channels=3, n_classes=NUM_CLASSES)
        net.load_state_dict(torch.load(MODEL_path, map_location={'cuda:1': 'cuda:0'}), strict=False)
        
    else:
        print("Choose a backbone: ResNet50, EfficientNetB6, DenseNet201")
        exit()
    
  
    print_trainable_parameters(net)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    
    """ create file """

    
    img_path = os.path.join( PREDICTED_MASK_PATH, 'predicted_mask')
    
    if not os.path.exists(img_path):
            os.makedirs(img_path)

           

    evaluation_unet3plus(net, device, valid_dataloader, metric,NUM_CLASSES, img_path)
    
    
if __name__ == '__main__':
    main() 
