Argumentos a introducir en `inferir_trt.py` :

- **`--model`**: Especificar la ubicación del archivo que contiene el modelo TRT.

- **`--data_dir`**: Ruta del directorio de datos.

- **`--batch_size`**: Tamaño del lote. Define el número de muestras de datos que se procesarán en cada iteración del modelo.

- **`--N_warmup_run`**: Número de ejecuciones de calentamiento. Indica cuántas veces se ejecutará el programa antes de realizar las mediciones reales.

- **`--N_run`**: Número de ejecuciones. Especifica cuántas veces se ejecutará el programa para realizar las mediciones.

- **`--name_csv`**: Ruta donde se guardará del archivo CSV. El archivo CSV contiene la siguiente información:

| Modelo | Placa | Precision | mIoU | AguaIoU | CyanoIoU | Batch | Tiempo min | Tiempo max | Tiempo mean | Img/s |
|--------|-------|-----------|------|---------|----------|-------|------------|------------|-------------|-------|
|        |       |           |      |         |          |       |            |            |             |       |

Si se quiere generar un archivo CSV con información de diferentes modelos, precisión y Batch usar `inferir_dif.sh`
