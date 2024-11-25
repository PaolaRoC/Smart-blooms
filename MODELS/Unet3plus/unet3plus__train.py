# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import argparse
import yaml
import os
from PIL import Image
from transformers import get_linear_schedule_with_warmup
import logging
import sys
from unet3plus_backbones import UNet_3Plus_DeepSup_CGM_ResNet50, UNet_3Plus_DeepSup_CGM_DenseNet201, UNet_3Plus_DeepSup_CGM_EfficientNetB6
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.functional import one_hot, binary_cross_entropy
import evaluate


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
     
        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))


        encoded_inputs = self.image_processor(image, segmentation_map, return_tensors="pt")
        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension
        if self.transform is not None:
          encoded_inputs= self.transform(encoded_inputs)
        return encoded_inputs

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from transformers import SegformerImageProcessor





def preprocess_train(examples):
  train_transforms = A.Compose([
      A.RandomRotate90(p=0.5),
      A.HorizontalFlip(p=0.5),
      A.ShiftScaleRotate(p=0.5, scale_limit=0.5, rotate_limit=0, shift_limit=0.1, border_mode=0),
      A.Perspective(p=0.5),
      A.GaussNoise(p=0.2),
      A.RandomBrightnessContrast(p=0.9, brightness_limit=0.05, contrast_limit=0.05),
      A.RandomGamma(p=0.7),
      A.Blur(p=0.5, blur_limit= 3),
      A.Normalize(p=1),
      ToTensorV2(),
  ])
  img = np.array(examples["pixel_values"].permute(1,2,0))
  mask_true = np.array(examples["labels"])
  augmented = train_transforms(image=img, mask=mask_true)

  examples["pixel_values"]  = augmented['image']
  examples["labels"]  = augmented['mask']
  return examples

def preprocess_val(examples):
  val_transforms = A.Compose([
      A.Normalize(p=1),
      ToTensorV2(),
  ])
  img = np.array(examples["pixel_values"].permute(1,2,0))
  mask_true = np.array(examples["labels"])
  augmented = val_transforms(image=img, mask=mask_true)

  examples["pixel_values"]  = augmented['image']
  examples["labels"]  = augmented['mask']
  return examples



def calc_BCloss(outputs, labels):
    label_onehot = one_hot(labels,num_classes=4).permute(0,3,1,2)
    loss_bce = binary_cross_entropy(outputs, label_onehot.float())
    return loss_bce
    
    
def compute_metrics(y_true, y_pred):

  class_wise_iou = []
  class_wise_dice_score = []
  list_labels_id = []

  smoothening_factor = 0.00001

  list_id= list(np.unique(y_true))
  dicc_id= {0: 'Fondo',1:'Agua', 2:'Cianobacterias', 3:'Rocas'}


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
      
  return class_wise_iou


def calc_BCyIOUloss(outputs, labels, iou_weight:float=0.8):
    #Binary Cross Entropy
    
    bce_loss = calc_BCloss(outputs, labels)
    
    # IOU
    
    y_pred= outputs.argmax(dim=1)
    my_iou = compute_metrics(labels.detach().cpu().numpy(),y_pred.detach().cpu().numpy())
    iou_loss = torch.tensor(np.mean(my_iou))
    loss = (1 - iou_weight) * bce_loss + iou_weight * (1-iou_loss)
    return loss
    


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
     '--config_file', help='path of config file.', required=True)
    args = parser.parse_args()
    
    # Load the config file
    with open( args.config_file , "r") as ymlfile:
       config_file = yaml.load(ymlfile, Loader=yaml.Loader)
       

    
    DATA_DIR= config_file["DATASET"]["DATA_DIR"]
    IMG_SIZE = config_file["DATASET"]["IMG_SIZE"]
    NUM_CLASSES = config_file["DATASET"]["NUM_CLASSES"]

    
    NET_BACKBONE = config_file["MODEL"]["NET_BACKBONE"]
    folder_path = config_file["MODEL"]["SAVE_MODEL"]
    MULTI_GPU = config_file["MODEL"]["MULTI_GPU"]
    
    BATCH_SIZE = config_file["TRAIN"]["BATCH_SIZE"]
    initial_learning_rate = config_file["TRAIN"]["LEARNING_RATE"]
    num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]
    

    root_dir = DATA_DIR
    image_processor = SegformerImageProcessor(size = {"height": IMG_SIZE, "width": IMG_SIZE},do_reduce_labels=False, do_rescale= False, do_normalize= False)
    
    train_dataset = SemanticSegmentationDataset(root_dir=root_dir, image_processor=image_processor, transform=preprocess_train)

    
    from torch.utils.data import DataLoader
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
   
   
    if NET_BACKBONE == 'ResNet50':
        net = UNet_3Plus_DeepSup_CGM_ResNet50(in_channels=3, n_classes=4)
    elif NET_BACKBONE == 'DenseNet201':
        net = UNet_3Plus_DeepSup_CGM_DenseNet201(in_channels=3, n_classes=4)
    elif NET_BACKBONE == 'EfficientNetB6':
        net = UNet_3Plus_DeepSup_CGM_EfficientNetB6(in_channels=3, n_classes=4)
    else:
        print("Elige un backbone: ResNet50, EfficientNetB6, DenseNet201")
        exit()
    
    
    
    metric = evaluate.load("mean_iou")
    
    optimizer = torch.optim.Adam(net.parameters(), lr=initial_learning_rate)
    num_training_stp = (len(train_dataloader) * num_epochs)
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps= int(0.1 * num_training_stp),
        num_training_steps= num_training_stp,
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # move model to GPU
    if MULTI_GPU == True:
        net = nn.DataParallel(net, device_ids=[0, 1, 2])
        
    
    net.to(device)
    ce_loss = CrossEntropyLoss()
    folder_path = os.path.join(folder_path, NET_BACKBONE + '_' + str(IMG_SIZE)  + '_' + str(BATCH_SIZE) + '_' + str(num_epochs) + '_' + str(initial_learning_rate).split('.')[-1])

    if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
    logging.basicConfig(filename=folder_path + "/log.txt", level=logging.INFO)
    logging.getLogger()
       
     
    net.train()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
       print("Epoch:", epoch)
       for idx, batch in enumerate(tqdm(train_dataloader)):
           # get the inputs;
           pixel_values = batch["pixel_values"].to(device)
           labels = batch["labels"].type(torch.long).to(device)
            
           # zero the parameter gradients
           optimizer.zero_grad()
            
           # forward + backward + optimize
           outputs = net(pixel_values)
            
           loss0 = calc_BCyIOUloss(outputs[0], labels)
           loss1 = calc_BCyIOUloss(outputs[1], labels)
           loss2 = calc_BCyIOUloss(outputs[2], labels)
           loss3 = calc_BCyIOUloss(outputs[3], labels)
           loss4 = calc_BCyIOUloss(outputs[4], labels)
           loss = (0.25*loss4) + (0.25*loss3) + (0.25*loss2) + (0.25*loss1) + (1.0*loss0)
            
           loss.backward()
            
           optimizer.step()
           lr_scheduler.step()
            
           # evaluate
           with torch.no_grad():
               predicted = outputs[0].argmax(dim=1)
              # note that the metric expects predictions + labels as numpy arrays
               metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
    
           # let's print loss and metrics every 100 batches
           if idx % 100 == 0:
               
               metrics = metric._compute(
                        predictions=predicted.cpu(),
                        references=labels.cpu(),
                        num_labels= NUM_CLASSES,
                        ignore_index=255,
                        reduce_labels=False, # we've already reduced the labels ourselves
                    )
                # currently using _compute instead of compute
                # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
               logging.info('Loss0 : %f , loss : %f, Mean iou: %f, Mean accuracy: %f' % (loss0.item(), loss.item(), metrics["mean_iou"], metrics["mean_accuracy"]))
    
               print("Loss0:", loss0.item())
               print("Loss:", loss.item())
               print("Mean_iou:", metrics["mean_iou"])
               print("Mean accuracy:", metrics["mean_accuracy"])
           
       if epoch %100 == 0:
           save_mode_path = os.path.join(folder_path, 'epoch_' + str(epoch) + '.pth')
           if MULTI_GPU == True:
               torch.save(net.module.state_dict(), save_mode_path)
           else:    
               torch.save(net.state_dict(), save_mode_path)
          
         
    save_mode_path = os.path.join(folder_path, 'epoch_' + str(epoch) + '.pth') 
    if MULTI_GPU == True:
        torch.save(net.module.state_dict(), save_mode_path)
    else: 
        torch.save(net.state_dict(), save_mode_path)

if __name__ == '__main__':
  main()
              