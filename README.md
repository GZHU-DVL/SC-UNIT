# SC-UNIT
This repository provides the official PyTorch implementation for the following paper:
**Unsupervised Image-to-Image Translation with Style Consistency**
> **Abstract:** *Unsupervised Image-to-Image Translation (UNIT) has gained significant attention due to its strong ability of data augmentation. UNIT aims to generate a visually pleasing image by synthesizing an image's content with another's style. However, current methods cannot ensure that the style of the generated image matches that of the input style image well. To overcome this issue, we present a new two-stage framework, called Unsupervised Image-to-Image Translation with Style Consistency (SC-UNIT), for improving the style consistency between the image of the style domain and the generated image. The key idea of SC-UNIT is to build a style consistency module to prevent the deviation of the learned style from the input one. Specifically, in the first stage, SC-UNIT trains a content encoder to extract the multiple-layer content features wherein the last-layer's feature can represent the abstract domain-shared content. In the second stage, we train a generator to integrate the content features with the style feature to generate a new image. During the generation process, dynamic skip connections and multiple-layer content features are used to build multiple-level content correspondences. Furthermore, we design a style reconstruction loss to make the style of the generated image consistent with that of the input style image. Numerous experimental results show that our SC-UNIT outperforms state-of-the-art methods in image quality, style diversity, and style consistency, even for domains with significant visual differences.*

## Installation
```bash
git clone https://github.com/GZHU-DVL/SC-UNIT.git
cd SC-UNIT
```
**Dependencies:**

We have tested on:
- CUDA 10.1
- PyTorch 1.7.0

All dependencies for defining the environment are provided in `environment/SC-UNIT.yaml`.

```bash
conda env create -f ./environment/SC-UNIT.yaml
```
## Dataset Preparation
| Translation Task | Used Dataset                                                                                                                                                                                                                                                                           | 
|:-----------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| 
| Male←→Female     | [CelebA-HQ](https://github.com/clovaai/stargan-v2#datasets-and-pre-trained-networks) ( divided into male and female subsets by [StarGANv2](https://github.com/clovaai/stargan-v2#datasets-and-pre-trained-networks))                                                                     |
| Dog←→Cat         | [AFHQ](https://github.com/clovaai/stargan-v2#datasets-and-pre-trained-networks) ( provided by [StarGANv2](https://github.com/clovaai/stargan-v2#datasets-and-pre-trained-networks))                                                                                                       |
| Face←→Cat        | [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ) and [AFHQ](https://github.com/clovaai/stargan-v2#datasets-and-pre-trained-networks)                                                                                                                                      |
| Bird←→Dog        | 4 classes of birds and 4 classes of dogs in [ImageNet291](https://github.com/williamyang1991/GP-UNIT/tree/main/data_preparation)
| Bird←→Car        | 4 classes of birds and 4 classes of cars in [ImageNet291](https://github.com/williamyang1991/GP-UNIT/tree/main/data_preparation)                                                                                          

## Image-to-Image Translation
Translate a content image to the target domain in the style of a style image by additionally specifying `--style`:
```python
python inference.py --generator_path PRETRAINED_GENERATOR_PATH --content_encoder_path PRETRAINED_ENCODER_PATH \ 
                    --content CONTENT_IMAGE_PATH --style STYLE_IMAGE_PATH --device DEVICE
```
## Training Content Encoder

We can train the content encoder to get the pre-trained model Content Encoder. (The Content Encoder should be in `./checkpoint/content_encoder.pt`.) Or you can download the supporting model at the [link](https://drive.google.com/file/d/1I7_IMMheihcIR57vInof5g6R0j8wEonx/view), which is provided by [GP-UNIT](https://github.com/williamyang1991/GP-UNIT/).
```python
python prior_distillation.py --unpaired_data_root UNPAIR_DATA --paired_data_root PAIR_DATA \
                             --unpaired_mask_root UNPAIR_MASK --paired_mask_root PAIR_MASK
```
## Training Image-to-Image Transaltion Network
```python
python train.py --task TASK --batch BATCH_SIZE --iter ITERATIONS \
                --source_paths SPATH1 SPATH2 ... SPATHS --source_num SNUM1 SNUM2 ... SNUMS \
                --target_paths TPATH1 TPATH2 ... TPATHT --target_num TNUM1 TNUM2 ... TNUMT
```  
## Acknowledgments

The code is developed based on GP-UNIT: https://github.com/williamyang1991/GP-UNIT