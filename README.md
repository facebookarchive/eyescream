# The Eyescream Project

Generating Natural Images using Neural Networks.

For our research summary on this work, please read the Arxiv paper: [http://arxiv.org/abs/1506.05751](http://arxiv.org/abs/1506.05751)

For a high-level blog post with a live demo, please go to this website: [http://soumith.ch/eyescream](http://soumith.ch/eyescream)

This repository contains the code to train neural networks and reproduce our results from scratch.

## Requirements
Eyescream requires or works with
* Mac OS X or Linux
* NVIDIA GPU with compute capability of 3.5 or above.

## Installing Dependencies
* Install [Torch](http://torch.ch)
* Install the nngraph and tds packages:

```
luarocks install tds
luarocks install nngraph
```

## Training your neural networks

* If you want to train the CIFAR-10 image generators, read the README in the cifar/ folder
* If you want to train the LSUN/Imagenet image generators, read the README in the lsun/ folder


## Discuss the paper/code at
* groups.google.com/forum/#!forum/torch7

See the CONTRIBUTING file for how to help out.

## License
Eyescream is BSD-licensed. We also provide an additional patent grant.