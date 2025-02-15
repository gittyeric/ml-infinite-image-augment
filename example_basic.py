from lib import ImageAugmenter
from examples.basic.model import SiftSimilarity
import cv2
import os
from os import path

# A simple binary classifier that predicts 0 if the switch is off otherwise 1.
# It also exposes the underlying confidence so we can tell future steps when 
# too much randomization has lowered model's confidence too via a custom diff_error().
training_img_dir = path.join("examples", "basic", "train", "img")
training_set = [path.join(training_img_dir, rel) for rel in os.listdir(training_img_dir)]
model = SiftSimilarity(training_set)
def on_off_predictor(img_file):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return model.predict(img)

# Calculate the "difference error" between an organic label and the model
# output of a synthetic spinoff by comparing output values or
# signaling error if the model's confidence drops due to synthetic mangling
def confidence_aware_diff_error(original_label, augmented_label):
    # Max error if confidence falls below threshold
    if augmented_label["γ"] < 0.1:
        return 1
    # Max error if outputs didn't match
    if augmented_label["out"] != original_label["out"]:
        return 1
    # Output matches, so only penalize now if low model confidence
    # No error for high confidence matches or higher confidence than original
    if augmented_label["γ"] >= 0.5 or original_label["γ"] < augmented_label["γ"]:
        return 0
    # Diff of confidence % is also error %, 1-to-1
    return original_label["γ"] - augmented_label["γ"]

# Spin up a predictor based on just 2 images and generate a validation
# set from the training set to validate that it works okay
def main():
    augmenter = ImageAugmenter(on_off_predictor, diff_error=confidence_aware_diff_error)
    # Analyze training set for model weaknesses under different randomizations
    augmenter.searchRandomizationBoundries(training_set, model.training_labels, step_size_percent=0.02)
    # Render a summary of how well the model behaved
    augmenter.renderBoundries(html_dir=path.join("examples", "basic", "analysis"))
    # Generate some images based on discovered randomiaation boundries and training data
    synth_imgs, synth_labels = augmenter.synthesizeMore(training_set, model.training_labels, count=50, realism=1, max_random_augmentations=10)
    # Use synthesized data as an improvised infinite validation set to test generality
    # and debug where your previously untested model tends to fail
    validation = augmenter.evaluate(synth_imgs, synth_labels)
    print(validation)

main()
