# Vision Transformer

[Vision Transformer](https://openreview.net/forum?id=YicbFdNTTy) (ICLR 2021 submission) in PyTorch.

**Real Talk**. This is not production ready, and not meant to be; it's a playground for ideas and implementations. Please don't use it for anything serious; please do use it for inspiration.
The vision transformer is a promising new direction for vision tasks. It manages to get rid of convolutions, using only the transformer architecture from [Attention Is All You Need](https://arxiv.org/abs/1706.03762) as building block. 

In addition to the vision transformer architecture, we provide tools to train it in self-supervised fashion using techniques from [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733).

Yannic Kilcher's amazing explanations
- https://www.youtube.com/watch?v=TrdevFK_am4
- https://www.youtube.com/watch?v=YPfUiOMYOEE


## Usage

For a self-contained and isolated dev environment

    docker-compose build

Enter it with

    docker-compose run dev

    $ visiont --help

### Sample command

```sh
visiont train --dataset /nas/3rd_party/openimagesV6/train
```


### Generate training samples

```sh
visiont generate -d /nas/3rd_party/openimagesV6/validation -l /nas/team-space/experiments/vision-t/09-11-2010/samples -n 100
```

### Validation step if the world is collapsing
```sh
 visiont val --dataset /nas/3rd_party/openimagesV6/validation --weights /nas/team-space/experiments/vision-t/13-11-2020-3770885/vt-051.pth
```


## License

Copyright © 2020 Daniel J. Hofmann

Distributed under the MIT License (MIT).
