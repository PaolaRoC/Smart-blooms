#### Contenido del archivo config.yaml
DATASET:
- DATA_DIR: Ruta del directorio que contiene los datos.
- NUM_CLASSES: Número de clases en el conjunto de datos (nuestro caso 4 clases).
- NUM_TRAIN_IMAGES: Número de imágenes en el conjunto de entrenamiento (max: 323 imágenes).
- NUM_TEST_IMAGES: Número de imágenes en el conjunto de pruebas (max: 79 imágenes).

MODEL:
- NET_BACKBONE: Arquitectura de red utilizada como extractor de características (ResNet50v02, Xception, MobilNetLarge, EfficientNetV2S y DenseNet201).
- SAVE_MODEL: Ruta donde se guardará el modelo entrenado.

TRAIN:
- BATCH_SIZE: Tamaño del lote durante el entrenamiento.
- NUM_EPOCHS: Número de épocas de entrenamiento.
- LEARNING_RATE: Tasa de aprendizaje utilizada durante el entrenamiento.
