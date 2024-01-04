**Instrucciones para Convertir un Modelo TensorFlow a ONNX**

**Paso 1: Obtener el Modelo ONNX Temporal**

Utiliza el archivo `ONNX_model.sh` para obtener un modelo ONNX temporal que necesitará ajustes. En el script, modifica los siguientes parámetros:

- `--saved-model my_model`: Especifica la ruta del modelo TensorFlow guardado (SavedModel) que deseas convertir.

- `--output temp.onnx`: Indica la ubicación donde deseas guardar el archivo ONNX resultante después de la conversión.

**Paso 2: Modificar el Modelo ONNX para TensorRT**

Usa el script `Config_ONNX_File.py` para ajustar el modelo ONNX y hacerlo compatible con TensorRT. Introduce los siguientes argumentos:

- `--load_tempmodel`: Ruta al modelo ONNX temporal obtenido en el primer paso.

- `--save_onnxmodel`: Ruta donde se guardará el modelo ONNX final después de las modificaciones.
