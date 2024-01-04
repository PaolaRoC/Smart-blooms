#!/bin/bash 
for name in FP32 FP16 INT8 ; do for model in MobilNetLarge EfficientNetV2S DenseNet201 ResNet50V2 Xception; do for i in {1,2,4,8,16,32,48}; do python3 inferir_trt.py --model /path/TRT_"$name"/"$model"_"$name"_orin2.trt --data_dir /path/Datos/ --batch_size $i --N_warmup_run 5 --N_run 10 --name_csv /path/info_metricas.csv; done; done; done
