# Video Frame Interpolation via Residue Refinement (RRIN)
 [Paper](https://ieeexplore.ieee.org/document/9053987/)

Haopeng Li, Yuan Yuan, [Qi Wang](http://crabwq.github.io/#top)

IEEE Conference on Acoustics, Speech, and Signal Processing, ICASSP 2020

### WIP Fork

This is a forked version of the RRIN network, where i have implemented my own training, and a functions to convert videos as well. 
There is also used some code from Super-SloMo: https://github.com/avinashpaliwal/Super-SloMo

Currently, this repository combines code from several neural network, like CAIN, BMBC, and RAFT (a motion estimation neural network). Along with some very basic combos of some of them in an attempt to make something better. 



### This is a repository purely for educational use.

The original code repository, lacked a way to train and convert videos, only including a demo, so i decided to use this oppurtunity to try to make my own implementations. Currently i use some combination of loss calulating functions, however i believe that i also have included the loss criterion that is stated in the paper, and for those curious, they may try traning with just that. I just googled my way to a function to seem to be the right kind of calculation. 

I added some other convenience functions, like edge padding images to match the accepted dimentions by the UNet, and then cropping that away in the converted images. 

### What can this repository do?

#### Training

I made my own implementation of training, inspired by other implementations. Can train RRIN, SRRIN and RAFT_RRIN (the latter to alterationsn of the first one.)
See the main.py file for arguments, but the basic use for training is:

    python main.py --model_name NAME_OF_CHECKPOINT --model_type MODEL train --train_folder TRAIN_FOLDER --batch_size 1 [--resume]
    
The model trains on 720p images by default, but it can easily be changed in dataloader.py. Also recommended to reduce the resize_dims to something less memory consuming. model_name is the name you want to the checkpoint, the trainer numbers the training points by default, and will resume on the latest one when the resume flag is included.

Example:

    python main.py --model_name RRIN_PTH --model_type RRIN train --train_folder .\train_data --batch_size 1 --resume
    


#### Testing / Converting
It can use RRIN, BMBC, SRRIN, RAFT_RRIN, and CAIN (untested for a long time, might need a fix or two.)
The model can both output images to a folder, and output to a video (in this case .webm), and support both a folder of images as input and a video, increasing both speed and storage savings as the need to split a video to images and then back is avoided. Sadly, no batching is implemented in the conversion process, so it's still somewhat slow. However, the default RRIN model is acceptably fast for shorter clips. RRIN also supports forward_chopping, which means it can split images into smaller sections to reduce memory requirement, basically a must on 4K videos. However, the performance is worse on large clips either way.

     python main.py --model_name NAME_OF_CHECKPOINT --model_type MODEL convert --input VIDEO_or_FOLDER_PATH --output WEBM_or_FOLDER_PATH --sf INTERMEDIATE_FRAMES --fps TARGET_FPS [--chop_forward]

If you interpolate a video, you can use --fps 2x to just double framerate. Any audio will also be included in the output, assuming webm supports it, otherwise an error will be thrown. If you want to specify framerate, you gotta use the --sf flag to specify intermediate frames. 

    python main.py --model_name RRIN_PTH --model_type RRIN convert --input test_video.mp4 --output interpolated.webm --fps 2x 

Note that this model performs best with only 2x and 3x framerate, accuracy drops quickly after that. 

### Requirements

Due to some issues, there  are a lot of packages needed to make this run, to cover for missing parts in others. Most notable, open-cv, pyav, pythorch, scipy, pillow. FFmpeg is also needed but i tihnk it may be autoinstalled for one framework. (Migth need to change the hardcoded path in utils.py)

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
