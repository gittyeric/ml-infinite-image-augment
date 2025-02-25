from augment import ImageAugmenter
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
# output from a synthetic spinoff image by comparing output values or
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

# This end-to-end example shows how to plumb a model up, search for randomization boundries,
# generate reports and finally generate some semi-realistic synthetic images and evaluate
# how the model performs on them.
def main():
    img_aug = ImageAugmenter(on_off_predictor, diff_error=confidence_aware_diff_error)
    # Analyze training set for model weaknesses under different randomizations
    img_aug.searchRandomizationBoundries(training_set, model.training_labels)
    # Render a summary of how well the model behaved
    img_aug.renderBoundries(html_dir=path.join("examples", "basic", "analysis"))
    # Control realism on a per-feature basis
    img_aug.set_augmentation_realism("SafeRotate", 0) # Unreal + max rotation since model is rotation-invariant
    img_aug.set_augmentation_weight("SafeRotate", 3) # 3x more likely to rotate than default
    img_aug.set_augmentation_realism("RandomSizedCrop", 0.5) # Only do small crop border cuts, but...
    img_aug.set_augmentation_weight("RandomSizedCrop", 2) # 2x more likely to crop at all
    # Generate some images based on discovered randomization boundries and training data
    synth_imgs, synth_labels = img_aug.synthesizeMore(training_set, model.training_labels, count=50, realism=0.5)
    # Use synthesized data as an improvised infinite validation set to test generality
    # and debug where your previously untested model tends to fail
    validation = img_aug.evaluate(synth_imgs, synth_labels)
    print("Average diff err on " + str(len(synth_imgs)) + " validation images: " + str(validation["avg_diff_error"]))

main()
