from PIL import Image, ImageDraw, ImageFont
import random
import os

# Our data
dataset = "ETL8G"

# Get curr path
absPath, _ = os.path.split(os.path.abspath(__file__))

# Function to generate image from kanji/kana
def gen_image_from_char(ch, filename):
    im = Image.new('RGB', (128, 127), color=(255, 255, 255))
    drawer = ImageDraw.Draw(im)
    y_ran = range(18, 33)
    x_ran = range(23, 38)
    drawer.text((random.choice(x_ran), random.choice(y_ran)), ch, 
                font=ImageFont.truetype('TakaoGothic.ttf', 70), fill=(0, 0, 0))
    im.save(filename, "png")

# Get all subfolders of our data
sub_folders = []
for dir, sub_dirs, files in os.walk(absPath+"/data/" + dataset):
    sub_folders.extend(sub_dirs)

# Generate and save all the text images
for folder in sub_folders:
    with open(absPath + "/data/" + dataset + "/" + folder + "/.char.txt") as f:
        a = f.readlines()
    gen_image_from_char(a[0], absPath + "/data/" + dataset + "/" + folder + "/true.png")