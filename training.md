# ImageNet training

Requirement: torch, torchvision, numpy. As long as the version is not too old, it should be fine.

All possible model names are 
    BN models: ```resnet50```, ```resnet101```, ```resnet152```, ```resnext50_32x4d```, ```resnext101_32x8d```,
    non-normalization models:
    ```rescale50```, ```rescale101```, ```rescale152```, ```rescaleX50_32x4d```, ```rescaleX101_32x8d```, 
    ```fixup50```, ```fixup101```.



ResNet50
```
python imagenet.py --model_name=resnet50 --train_path=TRAIN_PATH --val_path=VAL_PATH --batch_size=1024 \
                   --drop_conv=0.03 --drop_fc=0.3 --alpha=0.0 --multi_step=[30,60,90]
```

Rescale50
```
python imagenet.py --model_name=rescale50 --train_path=TRAIN_PATH --val_path=VAL_PATH --batch_size=1024 \
                   --drop_conv=0.03 --drop_fc=0.3 --alpha=0.0 --multi_step=[30,60,90]
```

Using Mixup and no Dropout
```
python imagenet.py --model_name=rescale50 --train_path=TRAIN_PATH --val_path=VAL_PATH --batch_size=1024 \
                   --drop_conv=0.0 --drop_fc=0.0 --alpha=0.5 --multi_step=[30,60,90]
```

```
python imagenet.py --model_name=rescale50 --train_path=TRAIN_PATH --val_path=VAL_PATH --batch_size=1024 \
                   --drop_conv=0.0 --drop_fc=0.0 --alpha=0.7 --multi_step=[30,60,90]
```

No regularization
```
python imagenet.py --model_name=rescale50 --train_path=TRAIN_PATH --val_path=VAL_PATH --batch_size=1024 \
                   --drop_conv=0.0 --drop_fc=0.0 --alpha=0.0 --multi_step=[30,60,90]
```


Cosine Learning rate
```
python imagenet.py --model_name=rescale50 --train_path=TRAIN_PATH --val_path=VAL_PATH --batch_size=1024 \
                   --drop_conv=0.03 --drop_fc=0.3 --alpha=0.0 --multi_step=[]
```


rescale101
```
python imagenet.py --model_name=rescale101 --train_path=TRAIN_PATH --val_path=VAL_PATH --batch_size=1024 \
                   --drop_conv=0.03 --drop_fc=0.3 --alpha=0.0 --multi_step=[30,60,90]
```



rescaleX101_32x8d
```
python imagenet.py --model_name=rescaleX101_32x8d --train_path=TRAIN_PATH --val_path=VAL_PATH --batch_size=1024 \
                   --drop_conv=0.03 --drop_fc=0.3 --alpha=0.0 --multi_step=[30,60,90]
```

rescaleX101_32x8d + cosline
```
python imagenet.py --model_name=rescaleX101_32x8d --train_path=TRAIN_PATH --val_path=VAL_PATH --batch_size=1024 \
                   --drop_conv=0.03 --drop_fc=0.3 --alpha=0.0 --multi_step=[]
```

VGG19
```
python vgg_imgaenet.py --model_name=vgg19_noBN --train_path=TRAIN_PATH --val_path=VAL_PATH --batch_size=1024 \
                   --multi_step=[60, 90]
```

VGG19 + cosine
```
python vgg_imgaenet.py --model_name=vgg19_noBN --train_path=TRAIN_PATH --val_path=VAL_PATH --batch_size=1024 \
                   --multi_step=[] --bs256_lr=0.01
```


VGG19_BN 
```
python vgg_imgaenet.py --model_name=vgg19_bn --train_path=TRAIN_PATH --val_path=VAL_PATH --batch_size=1024 \
                   --multi_step=[30, 60, 90] 
```

