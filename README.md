# Video Frame Interpolation via Residue Refinement (RRIN)
 [Paper](https://ieeexplore.ieee.org/document/9053987/)

Haopeng Li, Yuan Yuan, [Qi Wang](http://crabwq.github.io/#top)

IEEE Conference on Acoustics, Speech, and Signal Processing, ICASSP 2020

### WIP Fork

This is a forked version of the RRIN network, where i have implemented my own training, and a functions to convert videos as well. 
There is also used some code from Super-SloMo: https://github.com/avinashpaliwal/Super-SloMo

### This is a repository purely for educational use.

The original code repository, lacked a way to train and convert videos, only including a demo, so i decided to use this oppurtunity to try to make my own implementations. Currently i use some combination of loss calulating functions, however i believe that i also have included the loss criterion that is stated in the paper, and for those curious, they may try traning with just that. I just googled my way to a function to seem to be the right kind of calculation. 

I added some other convenience functions, like edge padding images to match the accepted dimentions by the UNet, and then cropping that away in the converted images. 

There are many things that can be improved on, notably, there is no random cropping (of larger images for example) nor flipping, which may help make training more robust. Created traning sequences should perhaps also be cropped when creating the dataset to not need padding during conversion, but presently they are just 720p and then halved to 360x368 during training.

There should also be a bit of cleanup, as conversion, training, data loading, and loss functions are all defined in the same file, which is a huge mess.

### License and Citation
If you find this useful, please check out the original, and if you use it within research, remember to cite the original paper. 

The use of the code is RESTRICTED to **non-commercial research and educational purposes**. 

```
@INPROCEEDINGS{RRIN, 
author={Haopeng, Li and Yuan, Yuan and Qi, Wang}, 
booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
title={Video Frame Interpolation Via Residue Refinement}, 
year={2020}, 
pages={2613-2617}
}
```
