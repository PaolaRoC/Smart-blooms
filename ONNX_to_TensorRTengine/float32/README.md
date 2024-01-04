##### Este comando realiza la conversión de un modelo ONNX a un motor TensorRT con precisión float32

```
trtexec --onnx=model_gn.onnx --saveEngine=model_gn.trt --exportProfile=model_gn.json --useSpinWait > model_gn.log 

```

- **`--onnx=model_gn.onnx`**: Especifica la ruta del archivo ONNX del modelo que se utilizará para construir el motor TensorRT.

- **`--saveEngine=model_gn.trt`**: Indica la ruta donde se guardará el motor TensorRT resultante después de la conversión desde ONNX. El nombre del archivo debe ser ./nombremodelo_FP32_nombreplaca.trt

- **`--exportProfile=model_gn.json`**: Genera un archivo JSON (model_gn.json) que contiene información de perfilado del modelo, como tiempos de ejecución por capa.

- **`--useSpinWait`**: Es una opción que puede mejorar el rendimiento en ciertos casos al hacer que el hilo de ejecución espere activamente en lugar de bloquearse durante la ejecución.

- **`> model_gn.log`**: Redirige la salida del comando, que generalmente incluye información de registro y mensajes de tiempo de ejecución, a un archivo de registro llamado `model_gn.log`. Esto puede ser útil para el seguimiento y análisis.
