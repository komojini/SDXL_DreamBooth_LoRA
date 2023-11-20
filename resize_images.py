from PIL import Image
import os
import argparse

IMAGES_DIR = ""
SAVE_DIR = ""
IMG_TYPE = ""
WIDTH = 0
HEIGHT = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="./datasets/")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--img_type", type=str, default=".png")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)

    opt = parser.parse_args()

    IMAGES_DIR = opt.img_dir
    SAVE_DIR = opt.save_dir
    IMG_TYPE = opt.img_type
    WIDTH = opt.width
    HEIGHT = opt.height
    if not SAVE_DIR:
        SAVE_DIR = IMAGES_DIR + "resized/"
    if not os.path.exists(SAVE_DIR) or not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    images = os.listdir(IMAGES_DIR)

    for image in images:
        img = Image.open(IMAGES_DIR + image).convert("RGB")
        img_name = image.split(".")[0]
        resized_img = img.resize((WIDTH, HEIGHT))
        resized_img.save(SAVE_DIR + img_name + IMG_TYPE)

