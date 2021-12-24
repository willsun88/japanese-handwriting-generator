# Japanese Handwriting Generator
A variational neural net to convert Japanese kanji into handwritten-style digits. Also able to be used to generate new Japanese kanji.

## Implementation Details
The model is an implementation of a Variational RNN (Citation 1), using a unimodal Gaussian distribution. The data is from the ETL Character Database (Citation 2), specifically ETL8G. Everything was tested on Python 3.8 with CUDA 11.3.

## How to use
1. Get access to the ETL Character Database, and download ETL8.
2. Use the [ETLCDB Image Extractor](https://github.com/choo/etlcdb-image-extractor) to extract the images, then place them in the data/ folder in this repository.
3. Download the Takao Gothic font, and place the TakaoGothic.tff file in the root directory of this repository.
4. Run `python add_starting_images.py` to generate the font images.
5. Run `python accumulate_data.py` to aggregate the data into one .npy file. (It might require 21 GB of free ram...)
6. Run `python main.py --train` to actually train the model.

## Sources Used
1. [A Recurrent Latent Variable Model for Sequential Data](https://arxiv.org/abs/1506.02216)
2. [ETL Character Database](http://etlcdb.db.aist.go.jp/)