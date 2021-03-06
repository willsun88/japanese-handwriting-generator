# Japanese Handwriting Generator
A generative neural net to convert Japanese kanji into handwritten-style digits.

## Implementation Details
The model is an implementation of a Pix2Pix neural network (Citation 1), with modified structure to match the smaller image size (256x256 in the paper, 128x128 here). The data is from the ETL Character Database (Citation 2), specifically ETL8G. Everything was tested on Python 3.8 with CUDA 11.3.

## How to use
1. Get access to the ETL Character Database, and download ETL8.
2. Use the [ETLCDB Image Extractor](https://github.com/choo/etlcdb-image-extractor) to extract the images, then place them in the data/ folder in this repository.
3. Download the Takao Gothic font, and place the TakaoGothic.tff file in the root directory of this repository.
4. Run `python add_starting_images.py` to generate the font images.
5. Run `python accumulate_data.py` to aggregate the data into one .npy file. (It might require 21 GB of free ram...)
6. Run `python main.py --train --save` to actually train the model. (Add the flag `--progbar` to add a progress bar; requires tensorflow.)
6. Run `python main.py --gen --load` to generate using a trained model on a set of validation data.

## Results (WIP)
| ![Epoch 3](epoch3.png) |
|:--:|
| <b>Progress on 3 epochs</b>|

| ![Epoch 8](epoch8.png) |
|:--:|
| <b>Progress on 8 epochs</b>|

| ![Discriminator Overfit](discriminator_overfit.png) |
|:--:|
| <b>(orange is discriminator loss, blue is generator loss)</b>|
| <b>Evidence of discriminator overfitting after too many epochs</b>|

## Sources Used
1. [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
2. [ETL Character Database](http://etlcdb.db.aist.go.jp/)