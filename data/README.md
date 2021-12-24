# Data Processing Steps
1. Get access to the ETL Character Database, and download ETL8. This should be a zip file that has files (with no extension) in them.
2. Use the [ETLCDB Image Extractor](https://github.com/choo/etlcdb-image-extractor) to extract the images, which should result in a folder called ETL8G, which has folders within it, and each subfolder should have images and a ".char.txt" file in them. Then, place the ETL6G/ folder in this directory (the data/ folder).
3. Then, from the root, run `python add_starting_images.py` to generate the font images. This should add a "true.png" file to every subfolder.
4. From the root, then run `python accumulate_data.py`. This should generate a "data.npy" file in the root of the data/ folder. This is all for the data processing.