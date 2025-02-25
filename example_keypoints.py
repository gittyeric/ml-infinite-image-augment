from augment import ImageAugmenter
from examples.basic.model import SiftSimilarity
import cv2
import os
import math
from os import path

# A COCO segment model that predicts a 4 keypoint segment outlining either an on or off
# power switch class as well as either the 0 or 1 class for off or on respectively.
# It also exposes the underlying confidence so we can tell future steps when 
# too much randomization has lowered model's confidence too via a custom diff_error().

# Define the Albumentations schema of keypoint labels
import albumentations as A
label_format = { "keypoint_params": A.KeypointParams(format='xy', label_fields=['power'], remove_invisible=True) }

training_set = [
     path.join("examples", "keypoints", "train", "img", "off.png"), 
     path.join("examples", "keypoints", "train", "img", "on.png")]
# Class IDs for the images above, matching the label_format definition
img_power_labels = [0, 1] # Off or on switch state
# Hardcoded outline box pixel positions by looking at off.png and on.png 
# for switch corner positions in photo shop.
# Must be in Albumentations label_format as defined above
training_labels = [{
    (0, 0), (0, 0),
    (0, 0), (0, 0)
}, {
    (0, 0), (0, 0),
    (0, 0), (0, 0)
}]

# Model wiring for library usage
model = SiftSimilarity(training_set)
def on_off_predictor(img_file):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    keypoints, power, confidence = model.predict(img)
    return {
        "keypoints": keypoints, "power": power, "confidence": confidence
    }

# Calculate the "difference error" between an organic label and the model
# output from a synthetic spinoff image by comparing output boxes or
# signaling error if the model's confidence drops due to synthetic mangling
def confidence_aware_diff_error(true_label, predicted_label):
    if predicted_label["power"] != true_label["power"]:
        return 1.0
    # Max error if confidence falls below threshold
    if predicted_label["confidence"] < 0.1:
        return 1

    confidence_penalty = 0
    # Output matches, so only penalize now if low model confidence
    # No error for high confidence matches or higher confidence than original
    if predicted_label["confidence"] < 0.5 and true_label["confidence"] > predicted_label["confidence"]:
        confidence_penalty = true_label["confidence"] - predicted_label["confidence"]
    point_dist_penalty = 0.0
    for i in range(len(true_label["keypoints"])):
        x1, y1 = true_label["keypoints"][i]
        # TODO: could be None
        x2, y2 = predicted_label["keypoints"][i]
        point_dist_penalty = point_dist_penalty + math.sqrt()
    # Diff of confidence % is also error %, 1-to-1 with distance-between-points ratio penalty
    return 1 - confidence_penalty * 0.5 - (point_dist_penalty/len(predicted_label["keypoints"])) * 0.5

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
    # Generate some images based on discovered randomiaation boundries and training data
    synth_imgs, synth_labels = img_aug.synthesizeMore(training_set, model.training_labels, count=50, realism=0.5)
    # Use synthesized data as an improvised infinite validation set to test generality
    # and debug where your previously untested model tends to fail
    validation = img_aug.evaluate(synth_imgs, synth_labels)
    print("Average diff err on " + str(len(synth_imgs)) + " validation images: " + str(validation["avg_diff_error"]))

main()
