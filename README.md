# Japanese Handwriting Generator
A neural net to convert Japanese kanji into handwritten-style digits

## How to use
Tested on Python 3.8
1. Get access to the ETL Character Database, and download ETL8.
2. Use the [ETLCDB Image Extractor](https://github.com/choo/etlcdb-image-extractor) to extract the images, then place them in the data/ folder in this repository.
3. Download the Takao Gothic font, and place the TakaoGothic.tff file in the root directory of this repository.
4. Run `python add_starting_images.py` to generate the font images.
5. Run `python accumulate_data.py` to aggregate the data into one .npy file. (It might require 21 GB of free ram...)
6. Run `python main.py --train` to actually train the model.