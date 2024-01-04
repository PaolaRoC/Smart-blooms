#### Contenido del archivo config.yaml

DATASET:
- DATA_DIR: Ruta del directorio que contiene los datos.
- NUM_CLASSES: Número de clases en el conjunto de datos (en nuestro caso 4 clases).
- NUM_TRAIN_IMAGES: Número de imágenes en el conjunto de entrenamiento (max: 323 imágenes).
- NUM_TEST_IMAGES: Número de imágenes en el conjunto de pruebas (max: 79 imágenes).

MODEL:
- ACTIVATION: Tipo de función de activación utilizada en el modelo (ReLU o GELU).
- NET_BACKBONE: Arquitectura de red utilizada como extractor de características (ResNet50V2 o DenseNet201).
- FREEZE_BACKBONE: Indicación de si se congela el extractor de características durante el entrenamiento (False o True).
- SAVE_MODEL: Ruta donde se guardará el modelo entrenado.

TRAIN:
- NUM_EPOCHES: Número de épocas de entrenamiento.
- BATCHES_PER_EPOCH: Número de lotes por época.
- SAMPLES_PER_BATCH: Número de ejemplos por lote.
- EARLY_STOPPING: Criterio de parada temprana que detiene el entrenamiento si no hay mejoras después de un cierto número de épocas.
- LEARNING_RATE: Tasa de aprendizaje utilizada durante el entrenamiento.
