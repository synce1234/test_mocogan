# MoCoGAN: Decomposing Motion and Content for Video Generation

This repository contains an implementation and further details of [MoCoGAN: Decomposing Motion and Content for Video Generation](http://arxiv.org/abs/1707.04993) by Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz.

CVPR Poster:

[<img src="https://github.com/sergeytulyakov/mocogan/raw/master/poster/cvpr-poster-mocogan.jpg">](https://github.com/sergeytulyakov/mocogan/raw/master/poster/cvpr-poster-mocogan.pdf)

## Representation

MoCoGAN is a generative model for videos, which generates videos from random inputs. It features separated representations of motion and content, offering control over what is generated. For example, MoCoGAN can generate the same object performing different actions, as well as the same action performed by different objects

![MoCoGAN Representation](https://github.com/sergeytulyakov/mocogan/raw/master/doc/controlling-content-and-motion.png)

## Examples of generated videos

<!---
All videos in this section are generated by MoCoGAN.
-->

We trained MoCoGAN on the [MUG Facial Expression Database](https://mug.ee.auth.gr/fed/) to generate facial expressions. When fixing the content code and changing the motion code, it generated the same person performs different expressions. When fixing the motion code and changing the content code, it generated different people performs the same expression. In the figure shown below, each column has fixed identity, each row shows the same action:

![Facial expressions](https://github.com/sergeytulyakov/mocogan/raw/master/doc/faces.gif "Facial expressions")

<!---
We trained MoCoGAN on a synthetically generated dataset of moving shapes. The color, shape and size of each moving shape represent content. Action is a specific motion direction. The shapes move bottom-top and right-left along a random Bezier curve.

![Shape motion](https://github.com/sergeytulyakov/mocogan/raw/master/doc/shapes.gif "Shape motion")
-->

We trained MoCoGAN on a [human action dataset](http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html) where content is represented by the performer, executing several actions. When fixing the content code and changing the motion code, it generated the same person performs different actions. When fixing the motion code and changing the content code, it generated different people performs the same action. Each pair of images represents the same action executed by different people:

![Human actions](https://github.com/sergeytulyakov/mocogan/raw/master/doc/action.gif "Human actions")


We have collected a large-scale TaiChi dataset including 4.5K videos of TaiChi performers. Below are videos generated by MoCoGAN.

![TaiChi](https://github.com/sergeytulyakov/mocogan/raw/master/doc/taichi.gif "TaiChi")


## Training MoCoGAN

Please refer to a [wiki page](https://github.com/sergeytulyakov/mocogan/wiki/Training-MoCoGAN)

## Citation

If you use MoCoGAN in your research please cite our paper:

[Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz, "MoCoGAN: Decomposing Motion and Content for Video Generation"](https://arxiv.org/abs/1707.04993)

```
@inproceedings{Tulyakov:2018:MoCoGAN,
 title={{MoCoGAN}: Decomposing motion and content for video generation},
 author={Tulyakov, Sergey and Liu, Ming-Yu and Yang, Xiaodong and Kautz, Jan},
 booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 pages = {1526--1535},
 year={2018}
}
```

## Other implementations:
1. [Alternative pytorch implementation](https://github.com/DLHacks/mocogan)
2. [Chainer implementation](https://github.com/raahii/mocogan-chainer)
