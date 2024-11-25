# -*- coding: utf-8 -*-



"""## Define PyTorch dataset and dataloaders"""

from torch.utils.data import Dataset
import os
from PIL import Image
from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.nn.functional import one_hot
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import yaml
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

val_transforms = A.Compose([
    A.Normalize(p=1),
    ToTensorV2(),
])

def preprocess_train(examples):
  img = np.array(examples["pixel_values"].permute(1,2,0))
  mask_true = np.array(examples["labels"])
  augmented = train_transforms(image=img, mask=mask_true)

  examples["pixel_values"]  = augmented['image']
  examples["labels"]  = augmented['mask']
  return examples

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

def calc_loss(outputs, labels, ce_loss):
    logits = outputs.logits
    upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    loss_ce = ce_loss(upsampled_logits, labels.long())
    return loss_ce
    
def calc_BCloss(outputs, labels, bce_loss):
    logits = outputs.logits
    upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    label_onehot = one_hot(labels,num_classes=4).permute(0,3,1,2)
    loss_bce = bce_loss(upsampled_logits, label_onehot[:].float())
    
    return loss_bce
    
def calc_BCyIOUloss(outputs, labels, bce_loss, iou_weight:float=0.8):
    #Binary Cross Entropy
    
    bce_loss = calc_BCloss(outputs, labels, bce_loss)
    
    # IOU
    logits = outputs.logits
    upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    predicted = upsampled_logits.argmax(dim=1)
    my_iou = compute_metrics(labels.detach().cpu().numpy(),predicted.detach().cpu().numpy())
    iou_loss = torch.tensor(np.mean(my_iou))

    loss = (1 - iou_weight) * bce_loss + iou_weight * (1-iou_loss)
    return loss
    
def IOUloss(outputs, labels):
    
    logits = outputs.logits
    upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    predicted = upsampled_logits.argmax(dim=1)
    my_iou = compute_metrics(labels.detach().cpu().numpy(),predicted.detach().cpu().numpy())
    iou_loss = torch.tensor(np.mean(my_iou), )
    loss = 1- iou_loss
    
    return loss.requires_grad_()


from transformers import SegformerImageProcessor

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


    
    NET_VERSION = config_file["MODEL"]["VERSION"] #"nvidia/mit-b4"
    folder_path = config_file["MODEL"]["SAVE_MODEL"] #"/SegFormer_pytorch/segformer_b4v2_224_10_420_0001"

    
    BATCH_SIZE = config_file["TRAIN"]["BATCH_SIZE"]
    initial_learning_rate = config_file["TRAIN"]["LEARNING_RATE"]
    num_epochs = config_file["TRAIN"]["NUM_EPOCHS"]

    root_dir = DATA_DIR
    image_processor = SegformerImageProcessor(size = {"height": IMG_SIZE, "width": IMG_SIZE},do_reduce_labels=False, do_rescale= False, do_normalize= False)
    
    train_dataset = SemanticSegmentationDataset(root_dir=root_dir, image_processor=image_processor, transform=preprocess_train)
    valid_dataset = SemanticSegmentationDataset(root_dir=root_dir, image_processor=image_processor, train=False, transform=preprocess_val)
    
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(valid_dataset))
      
    from torch.utils.data import DataLoader
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
 
    batch = next(iter(train_dataloader))
        
    """## Define the model"""
    
    from transformers import SegformerForSemanticSegmentation
    
    id2label = {0: "background", 1: "water", 2: "cyano", 3: "rock"}
    label2id = {label: id for id, label in id2label.items()}
    num_labels = len(id2label)

    
    # define model
    model = SegformerForSemanticSegmentation.from_pretrained(NET_VERSION,
                                                             num_labels=num_labels,
                                                             id2label=id2label,
                                                             label2id=label2id,
                                                             ignore_mismatched_sizes=True,
    )
    
    """## Fine-tune the model"""
    
        
        
    import evaluate
    
    metric = evaluate.load("mean_iou")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
    
    num_training_stp = (len(train_dataloader) * num_epochs)
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps= int(0.1 * num_training_stp),
        num_training_steps= num_training_stp,
    )

    # move model to GPU
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    print("Device:", device)
    
    ce_loss = CrossEntropyLoss()
    bce_loss = BCEWithLogitsLoss()
     
    if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
               
    model.train()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
       
       print("Epoch:", epoch)
       for idx, batch in enumerate(tqdm(train_dataloader)):
            # get the inputs;
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].type(torch.long).to(device)
      
            # forward + backward + optimize
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits
            loss = calc_loss(outputs, labels, ce_loss)
    
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            
            # evaluate
            with torch.no_grad():
              upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
              predicted = upsampled_logits.argmax(dim=1)
    
              # note that the metric expects predictions + labels as numpy arrays
              metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
    
            # let's print loss and metrics every 100 batches
            if idx % 100 == 0:
              # currently using _compute instead of compute
              # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
              metrics = metric._compute(
                      predictions=predicted.cpu(),
                      references=labels.cpu(),
                      num_labels=len(id2label),
                      ignore_index=255,
                      reduce_labels=False, # we've already reduced the labels ourselves
                  )
    
              print("Loss:", loss.item())
              print("Mean_iou:", metrics["mean_iou"])
              print("Mean accuracy:", metrics["mean_accuracy"])
    
       if epoch %100 == 0:
           save_mode_path = os.path.join(folder_path, 'epoch_' + str(epoch))
           model.save_pretrained(save_mode_path)
           
    
    save_mode_path = os.path.join(folder_path, 'epoch_' + str(epoch))  
    model.save_pretrained(save_mode_path)
    
if __name__ == '__main__':
  main()
    
