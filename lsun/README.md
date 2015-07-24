
To train your generators using the LSUN database, run the command:

```
DATA_ROOT=[path-to-lsun] th main.lua --dataset lsun
```


For imagenet, you'll have to [preprocess the dataset as described here.](https://github.com/soumith/imagenet-multiGPU.torch#data-processing)

To train your generators using the Imagenet database for the first time (for the caches to build up), run the command:

```
DATA_ROOT=[path-to-imagenet] th main.lua --dataset imagenet --nDonkeys 0
```

Subsequent runs can activate the multi-threaded data loader by changing the
command-line option to `--nDonkeys 4` (for 4 threads for example)


For more command-line options:

```
th main.lua --help


  --dataset          (default "imagenet")      imagenet | lsun
  --model            (default "large")      large | small | autogen
  -s,--save          (default "imgslogs")      subdirectory to save logs
  --saveFreq         (default 10)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRateD (default 0.01)        learning rate, for SGD only
  --learningRateG    (default 0.01)        learning rate, for SGD only
  -b,--batchSize     (default 128)         batch size
  -m,--momentum      (default 0)           momentum, for SGD only
  -w, --window       (default 3)           windsow id of sample image
  --nDonkeys         (default 10)           number of data loading threads
  --cache            (default "cache")     folder to cache metadata
  --epochSize        (default 5000)        number of samples per epoch
  --nEpochs          (default 25)
  --coarseSize       (default 16)
  --scaleUp          (default 2)          How much to upscale coarseSize
  --archgen          (default 1)
  --scratch          (default 0)
  --forceDonkeys     (default 0)
  
  ```
