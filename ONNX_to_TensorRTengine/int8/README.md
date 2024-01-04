#### Como crear el modelo con precisión in8
```
python build_engine.py \
    --onnx /path/to/model.onnx \
    --engine /path/to/engine.trt \
    --precision int8 \
    --calib_input /path/to/calibration/images \
    --calib_cache /path/to/calibration.cache
```
- ```--onnx ``` Ruta del modelo en formato ONNX que se utilizará para construir el motor TensorRT.
- ``` --engine ``` Ruta donde se guardará el motor TensorRT después de la construcción. 
- ``` --calib_input ```  Ruta del directorio que contiene las imágenes de calibración.
- ``` --calib_cache ``` Ruta donde se guardará la caché de calibración generada durante el proceso. 
- ``` --calib_num_images ```  Número de imágenes de calibración a utilizar.
