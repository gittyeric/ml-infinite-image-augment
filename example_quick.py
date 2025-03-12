from augment import ImageAugmenter
import os
from os import path

# Fastest example for getting started generating images naively and only based on your training data, not the
# threshold of randomness your MODEL+dataset is tolerant to.
def main():
    # Setup 2 images to generate images from
    img_dir = path.join("examples", "basic", "train", "img")
    img_files = [path.join(img_dir, rel) for rel in os.listdir(img_dir)]

    img_aug = ImageAugmenter()
    # Find reasonable random intensity bounds with a dumb default vision model and your dataset of 2 switch images
    img_aug.search_randomization_boundries(img_files)
    # Generate 50 "reasonable" random permutations of the on/off switch to "generated" directory
    summary = img_aug.synthesize_more(img_files, count=50)
    print("Generated " + str(summary["count"]) + " images")

main()
