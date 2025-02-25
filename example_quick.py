from augment import ImageAugmenter
from examples.basic.model import SiftSimilarity
import cv2
import os
from os import path

# Fastest example for getting started generating images with naively and only based on your training data, not the
# threshold of randomness YOUR model is tolerant to.
def main():
    # Naively derive "reasonable" synthetic derivative images based on the on/off switch images.
    # "Reasonable" here is quick-and-dirty defined by a default one-shot image model under the hood
    training_img_dir = path.join("examples", "basic", "train", "img")
    training_set = [path.join(training_img_dir, rel) for rel in os.listdir(training_img_dir)]
    img_aug = ImageAugmenter()
    # Analyze training set for model weaknesses under different randomizations
    img_aug.searchRandomizationBoundries(training_set)
    # Generate 50 "reasonable" random permutations of the on/off switch
    synth_imgs, synth_labels = img_aug.synthesizeMore(training_set, count=50, output_dir="generated")

main()
