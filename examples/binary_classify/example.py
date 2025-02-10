from lib import ImageAugmenter
import numpy as np
import cv2

# Base images to train model on
training_set = ["examples/binary_classify/on.png, examples/binary_classify/off.png"]
# Labels are boolean but also includes γ (confidence) in order
# to provide a richer error signal beyond raw model output
training_labels = [{"γ": 1, "out": 1}, {"γ": 1, "out": 0}]

matcher = None
trained_features = None
sift = None

def get_model():
    global matcher
    global trained_features
    global sift
    if matcher is not None:
        return matcher, sift, trained_features

    # "Train" the BFMatcher by finding features of all input models
    imgs = [cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB) for x in training_set]
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    trained_features = [sift.detectAndCompute(img, None) for img in imgs]
    
    # BFMatcher with default params
    matcher = cv2.BFMatcher()
    return matcher, sift, trained_features

# A simple binary classifier that predicts 0 if the switch if off otherwise 1
# It also exposes the underlying confidence so we can tell future steps when 
# too much randomization has lowered model's confidence too via a custom diff_error().
def on_off_predictor(img_file):
    matcher, sift, trained_features = get_model()
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, predicted = sift.detectAndCompute(img, None)

    

# Calculate the "difference error" between an organic label and the model
# output of a syhnthetic spinoff by comparing output values or
# signing error if the model's confidence drops due to synthetic mangling
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
def validate_on_off_predictor():
    augmenter = ImageAugmenter(on_off_predictor, diff_error=confidence_aware_diff_error)
    # Analyze training set for model weaknesses under different randomizations
    augmenter.searchRandomizationBoundries(training_set, training_labels)
    # Render a summary of how well the model behaved
    augmenter.renderBoundries(html_dir="examples/binary_classify/analysis")
    # Generate some images based on discovered randomiaation boundries and training data
    synth = augmenter.synthesizeMore(training_set, training_labels)
    # Use synthesized data as an improvised infinite validation set to test generality
    # and debug where your previously untested model tends to fail
    validation = augmenter.evaluate(synth["images"], synth["labels"])
    print(validation)

