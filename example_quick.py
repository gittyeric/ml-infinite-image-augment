from augment import ImageAugmenter
import os
from os import path

# Fastest example for getting started generating images with naively and only based on your training data, not the
# threshold of randomness YOUR model is tolerant to.
def main():
    # Naively derive "reasonable" synthetic derivative images based on the on/off switch images.
    # "Reasonable" here is quick-and-dirty defined by a default one-shot image model under the hood
    img_dir = path.join("examples", "basic", "train", "img")
    img_files = [path.join(img_dir, rel) for rel in os.listdir(img_dir)]
    img_aug = ImageAugmenter()
    # Analyze training set for model weaknesses under different randomizations
    img_aug.searchRandomizationBoundries(img_files)
    # Generate 50 "reasonable" random permutations of the on/off switch to "generated" directory
    img_aug.synthesizeMore(img_files, count=50, output_dir="generated")

main()
