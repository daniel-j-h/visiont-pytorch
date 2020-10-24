# Vision Transformer

[Vision Transformer](https://openreview.net/forum?id=YicbFdNTTy) (ICLR 2021 submission) in PyTorch.

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

    $ visiont
    It works!


## License

Copyright © 2020 Daniel J. Hofmann

Distributed under the MIT License (MIT).
