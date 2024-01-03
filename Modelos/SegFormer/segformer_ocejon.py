# -*- coding: utf-8 -*-

import argparse
from tensorflow.keras import backend

import os
from glob import glob


import tensorflow as tf
import keras.backend as K

from tensorflow import keras
from tensorflow.keras import layers
from keras.api._v2.keras.layers import Normalization

import numpy as np
import itertools

from typing import Any, Optional


import matplotlib.pyplot as plt

import yaml

from transformers import TFSegformerForSemanticSegmentation

seed=42
IMAGE_SIZE = 224




def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])

    if mask:
        image = tf.image.decode_png(image, channels=1)
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])

    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = (image- mean) / tf.maximum(std, backend.epsilon())
        image = tf.transpose(image, (2, 0, 1))
    return image


def load_data(image_list, mask_list,ds_val):
    image = read_image(image_list)
    if ds_val:
      mask = read_image(mask_list, mask=True)
      mask = tf.squeeze(mask)
    else:
      mask = read_image(mask_list, mask=True)
    return {"pixel_values": image, "labels": mask }


def data_generator(image_list, mask_list,ds_val):

    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(lambda image, mask: load_data(image, mask, ds_val),num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()

        self.augment_inputs = [
            tf.keras.layers.RandomFlip(mode="horizontal", seed=seed),
            tf.keras.layers.RandomRotation(factor=0.25, fill_mode="wrap", seed=seed),
            tf.keras.layers.RandomTranslation(height_factor=0.25, width_factor=0.25, fill_mode="wrap", seed=seed),
            tf.keras.layers.RandomZoom(height_factor=(-0.3, -0.2), fill_mode="wrap", seed=seed)
        ]

        self.augment_labels = [
            tf.keras.layers.RandomFlip(mode="horizontal", seed=seed),
            tf.keras.layers.RandomRotation(factor=0.25, fill_mode="wrap", seed=seed),
            tf.keras.layers.RandomTranslation(height_factor=0.25, width_factor=0.25, fill_mode="wrap", seed=seed),
            tf.keras.layers.RandomZoom(height_factor=(-0.3, -0.2), fill_mode="wrap", seed=seed)
        ]

    def call(self, samples):
        augmented_inputs = [samples["pixel_values"]]
        augmented_labels = [samples["labels"]]

        for augment_input in self.augment_inputs:
            augmented_inputs.append(augment_input(samples["pixel_values"]))

        for augment_label in self.augment_labels:
            augmented_labels.append(augment_label(samples["labels"]))

        augmented_inputs = tf.stack(augmented_inputs)
        augmented_labels = tf.stack(augmented_labels)
        augmented_labels = tf.squeeze(augmented_labels)

        return {"pixel_values": augmented_inputs, "labels": augmented_labels}


def compute_metrics(y_true, y_pred):

  class_wise_iou = []
  class_wise_dice_score = []
  list_labels_id = []

  smoothening_factor = 0.00001

  list_id= list(np.unique(y_true))
  dicc_id= {0:'Fondo',1:'Agua', 2:'Cianobacterias', 3:'Rocas'}


  for i in list_id:
    intersection = np.sum((y_pred == i) * (y_true == i))
    y_true_area = np.sum((y_true == i))
    y_pred_area = np.sum((y_pred == i))
    combined_area = y_true_area + y_pred_area

    iou = (intersection + smoothening_factor) / (combined_area - intersection + smoothening_factor)
    class_wise_iou.append(iou)

    dice_score =  (2 *(intersection + smoothening_factor) / (combined_area + smoothening_factor))
    class_wise_dice_score.append(dice_score)
    list_labels_id.append(dicc_id[i])

  return class_wise_iou, class_wise_dice_score, list_labels_id


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=1)
    pred_mask = tf.expand_dims(pred_mask, -1)
    return pred_mask[0]


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



def plot_samples_matplotlib(display_list, display_string, figsize=(4, 3)):
    fig, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):

        if display_list[i].shape[-1] == 3:
            axes[i].title.set_text(title[i])

            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))

        else:
            axes[i].imshow(display_list[i])

    fig.text(0.5, 0.3,display_string, horizontalalignment='center',
     verticalalignment='bottom')
    #plt.show()
    plt.savefig("./pred_vs_ground_img.png")


colormap = np.array([[0,0,0],[255,50,50],[214,255,50],[50,255,132]])
colormap = colormap.astype(np.uint8)

def plot_predictions(image,ground_truth, prediction_mask,colormap):
    
    prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 4)
    ground_truth_colormap = decode_segmentation_masks(ground_truth, colormap, 4)
    
    iou_list, dice_score_list, labels_id = compute_metrics(ground_truth,prediction_mask)
    metrics_by_id = [(idx, iou, dice_score) for i, (idx,iou, dice_score) in enumerate(zip(iou_list, dice_score_list, labels_id)) if iou > 0.0]
    display_string_list = ["{}: IOU: {} Dice Score: {}".format(idx, iou, dice_score) for iou, dice_score, idx in metrics_by_id]
    display_string = "\n\n".join(display_string_list)
    plot_samples_matplotlib(
        [image,ground_truth_colormap,prediction_colormap], display_string, figsize=(15, 14)
    )




def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
     '--config_file', help='path of config file.', required=True)
    args = parser.parse_args()
    
    # Load the config file
    with open( args.config_file , "r") as ymlfile:
       config_file = yaml.load(ymlfile, Loader=yaml.Loader)
    
    BATCH_SIZE = config_file["TRAIN"]["BATCH_SIZE"]   
    NUM_CLASSES = config_file["DATASET"]["NUM_CLASSES"]
    DATA_DIR =  config_file["DATASET"]["DATA_DIR"]
    NUM_TRAIN_IMAGES= config_file["DATASET"]["NUM_TRAIN_IMAGES"]
    NUM_TEST_IMAGES =  config_file["DATASET"]["NUM_TEST_IMAGES"]

    train_images = sorted(glob(os.path.join(DATA_DIR, "train_img/*")))[:NUM_TRAIN_IMAGES]
    train_masks = sorted(glob(os.path.join(DATA_DIR, "train_mask_ids/*")))[:NUM_TRAIN_IMAGES]

    val_images = sorted(glob(os.path.join(DATA_DIR, "test_img/*")))[:NUM_TEST_IMAGES]
    val_masks = sorted(glob(os.path.join(DATA_DIR, "test_mask_ids/*")))[:NUM_TEST_IMAGES]
    
    train_dataset = data_generator(train_images, train_masks, ds_val=False)
    val_dataset = data_generator(val_images, val_masks, ds_val= True)
    
    
    train_batches = (
        train_dataset
        .map(Augment())
        .rebatch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE))
    
    
    test_batches = (
        val_dataset
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE))
    
    
    """# Load a pretrained SegFormer checkpoint"""
    model_checkpoint = "nvidia/mit-b0"
    id2label = {0: "fondo", 1: "agua", 2: "cyano", 3: "rocas"}
    label2id = {label: id for id, label in id2label.items()}
    num_labels = len(id2label)
    model = TFSegformerForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    """# Compile the model"""
    
    lr = config_file["TRAIN"]["LEARNING_RATE"]
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,
                  metrics= [
                  tf.keras.metrics.IoU(num_classes=NUM_CLASSES, target_class_ids=[1], sparse_y_true = True, sparse_y_pred = False),
                  tf.keras.metrics.IoU(num_classes=NUM_CLASSES, target_class_ids=[2], sparse_y_true = True, sparse_y_pred = False),
                  tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES, sparse_y_true = True, sparse_y_pred = False)]
                  )
    
    """# Train model"""


    epochs = config_file["TRAIN"]["NUM_EPOCHES"]
    VALIDATION_STEPS= NUM_TEST_IMAGES // BATCH_SIZE
  




    print('empezo el entranamiento')
    model.fit(
        train_batches, 
        validation_steps=VALIDATION_STEPS,
        validation_data=test_batches,
        epochs=epochs,
    )
    print('acabo el entrenamiento')
    model.save_pretrained(config_file["MODEL"]["SAVE_MODEL"])
    
    for sample in test_batches.take(1):
        images, masks = images, masks = sample["pixel_values"], sample["labels"]
        masks = tf.expand_dims(masks, -1)
        pred_masks = model.predict(images).logits
        pred_mask_interpolation= tf.keras.layers.UpSampling2D( size=(IMAGE_SIZE // pred_masks.shape[2], IMAGE_SIZE // pred_masks.shape[3]),
        data_format= "channels_first",
        interpolation="bilinear",
        )(pred_masks)
        images = tf.transpose(images, (0, 2, 3, 1))
        prd_mask= create_mask(pred_mask_interpolation)
        mask= np.squeeze(np.array(masks[0])).astype(np.uint8)
        pred = np.squeeze(np.array(prd_mask)).astype(np.uint8)
        plot_predictions(images[0], mask, pred,colormap)


if __name__ == '__main__':
  main()
