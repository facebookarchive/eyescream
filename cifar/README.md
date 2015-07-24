Training a standard GAN on 32x32 cifar images:
```
th scripts/train_cifar.lua --scale 32
```

Training a coarse-to-fine GAN on 16 -> 32 images:
```
th  scripts/train_cifar_coarse_to_fine.lua --coarseScale 16 --fineScale 32
```


Training a class conditional GAN on 32x32 cifar images:
```
th scripts/train_cifar_classcond.lua --scale 32
```

Training a class conditional coarse-to-fine GAN on 16 -> 32 images:
```
th scripts/train_cifar_coarse_to_fine_classcond.lua --coarseScale 16 --fineScale 32
```

The default training hyperparameters should suffice for all cases.  
Add -g <gpu> to run on a specific gpu (default is cpu).   
If you have display (https://github.com/szym/display) installed, -p will plot sample generations and training curves after every epoch.  
The number of channels in the hidden layers of G and D can be specified with --hidden_G <numhidG> --hidden_D <numhidD>.  
To see model parameters for decent models at every scale please see https://gist.github.com/soumith/e3f722173ea16c1ea0d9.

