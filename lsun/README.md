
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
```