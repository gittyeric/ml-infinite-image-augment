import math
import os
from os import path
import json
import random
import albumentations as A
import cv2
import shutil

COLOR_AUGMENTATIONS = [
    "Brighten",
    "Contrast",
    "Darken",
    "Decontrast",
    "Desaturate",
    "Dehue",
    "Hue",
    "LessBlue",
    "LessGreen",
    "LessRed",
    "MoreBlue",
    "MoreGreen",
    "MoreRed",
    "Saturate",
]

DISTORTION_AUGMENTATIONS = [
    "ElasticTransform",
    "GaussianBlur",
    "MotionBlur",
    "SafeRotate",
    "Sharpen",
]

PIXEL_DROPOUT_AUGMENTATIONS = [
    "Downscale",
    "MultiplicitiveNoise",
    "PixelDropout",
    "RandomSizedCrop",
    "Superpixels"
]

ALL_AUGMENTATIONS=[
    *COLOR_AUGMENTATIONS,
    *DISTORTION_AUGMENTATIONS,
    *PIXEL_DROPOUT_AUGMENTATIONS
]

# Relative probability of picking an augmentation,
# most are 1 unless a dimension such as red shift is
# split among multiple augs (ex. MoreRed and LessRed)
AUG_PROB_DIST={
    "Brighten": 0.5,
    "Contrast": 0.5,
    "Darken": 0.5,
    "Decontrast": 0.5,
    "Desaturate": 0.5,
    "Dehue": 0.5,
    "Hue": 0.5,
    "LessBlue": 0.25,
    "LessGreen": 0.25,
    "LessRed": 0.25,
    "MoreBlue": 0.25,
    "MoreGreen": 0.25,
    "MoreRed": 0.25,
    "Saturate": 0.5,
    "ElasticTransform": 1,
    "GaussianBlur": 0.5,
    "MotionBlur": 0.5,
    "SafeRotate": 1,
    "Sharpen": 1,
    "Downscale": 0.5,
    "MultiplicitiveNoise": 0.5,
    "PixelDropout": 0.5,
    "RandomSizedCrop": 1,
    "Superpixels": 0.5
}

BANNED_PAIRS = {
    "Darken": ["Brighten"],
    "Brighten": ["Darken"],
    "Contrast": ["Decontrast"],
    "Decontrast": ["Contrast"],
    "Saturation": ["Desaturation"],
    "Desaturation": ["Saturation"],
    "LessRed": ["MoreRed"],
    "MoreRed": ["LessRed"],
    "LessGreen": ["MoreGreen"],
    "MoreGreen": ["LessGreen"],
    "LessBlue": ["MoreBlue"],
    "MoreBlue": ["LessBlue"],
    "PixelDropout": ["Downscale", "Superpixels"],
    "Downscale": ["PixelDropout", "Superpixels"],
    "Superpixels": ["PixelDropout", "Downscale"],
    "ElasticTransform": ["RandomSizedCrop"],
    "RandomSizedCrop": ["ElasticTransform"],
    "MotionBlur": ["GaussianBlur"],
    "GaussianBlur": ["MotionBlur"]
}

# Diff errors beyond 2% are considered non-negligible
MIN_NONNEGLIGABLE_ERR = 0.02
# Diff errors > 90% are considered critical failure
MIN_CRITICAL_ERR = 0.49

class SiftSimilarity():

    def __init__(self, training_img_filenames: list[str]):
        # "Train" the BFMatcher by finding features of all input models
        imgs = [cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB) for x in training_img_filenames]
        self.training_labels = [{"γ": 1, "out": i} for i in range(len(training_img_filenames))]
        # Initiate SIFT detector
        self.sift = cv2.SIFT_create()
        # "Train" by converting all images to SIFT features for fast similarity comparison
        self.trained_features = [self.sift.detectAndCompute(img, None) for img in imgs]
        # BFMatcher with default params
        self.matcher = cv2.BFMatcher()

    def sift_matches(self, baseline_features, img_features):
        kps1,desc1 = baseline_features
        kps2,desc2 = img_features
        if len(kps2) < 2 or len(desc2) < 2:
            return [], kps1, desc1, kps2, desc2
        matches = self.matcher.match(desc1,desc2)
        matches = sorted(matches,key= lambda x:x.distance)
        return matches,kps1,desc1,kps2,desc2

    def get_inliers_ratio(self, desc1,desc2,ratio):
        matches=self.matcher.knnMatch(desc1,desc2,2)
        inliers=[]
        for m in matches:
            if((m[0].distance/m[1].distance)<ratio):
                inliers.append(m[0])
        return inliers

    def predict(self, img):
        best_index = 0
        best_similarity = 0
        img_features = self.sift.detectAndCompute(img, None)
        for i in range(len(self.trained_features)): 
            baseline = self.trained_features[i]
            allMatches,kps1,desc1,kps2,desc2 = self.sift_matches(baseline, img_features)
            if len(allMatches) == 0:
                continue
            match=self.get_inliers_ratio(desc1,desc2,ratio=0.7)
            similarity=self.get_similarity(match,kps1,kps2)

            if similarity > best_similarity:
                best_similarity = similarity
                best_index = i
        return {"out": best_index, "γ": best_similarity}

    def get_similarity(self, inliers, kps1, kps2):
        similarity = len(inliers) / min(len(kps1), len(kps2))
        return similarity

def new_default_predictor(training_imgs: list[str]):
    model = SiftSimilarity(training_imgs)
    return lambda img_file: model.predict(cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB))

# Calculate the "difference error" between an organic label and the model
# output from a synthetic spinoff image by comparing output values or
# signaling error if the model's confidence drops due to synthetic mangling
def confidence_aware_diff_error(original_label, augmented_label):
    # Max error if outputs didn't match
    if augmented_label["out"] != original_label["out"]:
        return 1
    # Diff of confidence % is also error %, 1-to-1
    return abs(original_label["γ"] - augmented_label["γ"])

# Naively assume string equality is no error otherwise max error
def err_if_not_strict_eq(original_labels, aug_labels):
    return 0 if str(original_labels) == str(aug_labels) else 1

def verbose_synthetic_namer(organic_img_name: str, label, uid: int, applied_augs: list):
    aug_str = "".join([aug['__class_fullname__'].replace("Random", "")[0:6] for aug in applied_augs])
    img_basename_no_type = ".".join(organic_img_name.split(".")[0:-1])
    img_type = organic_img_name.split(".")[-1]
    return f'''{img_basename_no_type}_{aug_str}_{str(uid)}.{img_type}'''

class ImageAugmenter:
    def __init__(self, my_predict: lambda img_filename: str = None, diff_error: lambda orig, augmented: float=err_if_not_strict_eq, augmentations: list[str]=ALL_AUGMENTATIONS, label_format=None):
        """
        The main class for grid searching over a training dataset with a model to determine random augmentation limits that the model can tolerate, and store the results.
        Returns a JSON blob representing the raw results of each augmentation feature, the same as the contents of `analyze.json`.

        `my_predict`: A function that takes an absolute image filename and runs inference against it, returning prediction labels (the labels' type only has meaning to you as long as it's string-ifiable)

        `diff_error`: (Optional) Custom error/cost function in the range [0, 1.0] where 0 means both the original unaugmented image labels and augmented image output labels match perfectly or 1 meaning the labels match as little as possible.  Default behavior stringifies both raw and augment labels and assumes zero error only if strings are strictly equal, otherwise 1.

        `augmentations`: (Optional) List of augmentation string types to apply for all downstream operations, defaults to all supported augmentations that are mostly 1-to-1 with those provided in the Albumentations library.  You can pick-and-choose each individually if you know your model won't be able to handle certain augmentation types or want to prototype with a smaller/faster feature set.  Available types are:

        `label_format`: (Optional) Make dataset synthesis label-type aware so that for example bounding boxes are geometrically transformed to match the augmentations applied to the image.  This option is passed through as-is to the Albumentations library for it to figure out the label transformations, reference [their documentation](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/) for supported label formats.  COCO bounding box example: 
        """
        self.my_predict = my_predict
        self.predict = my_predict
        self.diff_error = diff_error
        self.augmentations = augmentations
        self.analytics = None
        self.label_format = label_format
        self.intensity_overrides = {}
        # Copy the aug prob distribution so user can overwrite per-augmentation
        self.probs = AUG_PROB_DIST.copy()

    def _pick_augs(self, min_random_augmentations, max_random_augmentations):
        # Pick augmentations according to static or user-overwritten probability distributions
        unpicked_augs = [key for key in self.analytics["augs"].keys()]
        picked_augs = []
        while len(unpicked_augs) > 0:
            weights = []
            for aug in unpicked_augs:
                weights.append(self.probs[aug])
            picked = random.choices(unpicked_augs, weights, k=1)
            picked_augs.append(picked[0])
            unpicked_augs.remove(picked[0])
        
        # Dice roll the augmentation count
        aug_count = random.randint(min_random_augmentations, max_random_augmentations)

        # Remove duplicate or contradictory aug types, preferring the first
        aug_set = {}
        for aug in picked_augs:
            bans = BANNED_PAIRS[aug] if aug in BANNED_PAIRS else []
            banned = False
            for ban in bans:
                if ban in aug_set:
                    banned = True
            if not banned:
                aug_set[aug] = True
            if len(aug_set) == aug_count:
                break
        return aug_set.keys()

    def _buildBoundryTestPipeline(self, img_width: int, img_height: int, intensity: float, aug_name: str):
        label_opts = {} if self.label_format is None else self.label_format
        step = None
        blur = round(intensity * 50) - (round(intensity * 50) % 2) + 1
        if aug_name == "Darken":
            step = A.RandomBrightnessContrast(p=1, brightness_limit=(-intensity, -intensity))
        elif aug_name == "Brighten":
            step = A.RandomBrightnessContrast(p=1, brightness_limit=(intensity, intensity))
        elif aug_name == "Contrast":
            step = A.RandomBrightnessContrast(p=1, contrast_limit=(intensity, intensity))
        elif aug_name == "Decontrast":
            step = A.RandomBrightnessContrast(p=1, contrast_limit=(-intensity, -intensity))
        elif aug_name == "Dehue":
            step = A.HueSaturationValue(p=1, hue_shift_limit=(-100 + intensity * 200, -100 + intensity * 200))
        elif aug_name == "Hue":
            step = A.HueSaturationValue(p=1, hue_shift_limit=(intensity * 100, intensity * 100))
        elif aug_name == "LessBlue":
            step = A.RGBShift(p=1, r_shift_limit=(-intensity, -intensity))
        elif aug_name == "LessGreen":
            step = A.RGBShift(p=1, g_shift_limit=(-intensity, -intensity))
        elif aug_name == "LessRed":
            step = A.RGBShift(p=1, b_shift_limit=(-intensity, -intensity))
        elif aug_name == "MoreBlue":
            step = A.RGBShift(p=1, r_shift_limit=(intensity, intensity))
        elif aug_name == "MoreGreen":
            step = A.RGBShift(p=1, g_shift_limit=(intensity, intensity))
        elif aug_name == "MoreRed":
            step = A.RGBShift(p=1, b_shift_limit=(intensity, intensity))
        elif aug_name == "Saturate":
            step = A.HueSaturationValue(p=1, sat_shift_limit=(intensity * 100, intensity * 100))
        elif aug_name == "Desaturate":
            step = A.HueSaturationValue(p=1, sat_shift_limit=(intensity * -100, intensity * -100))
        elif aug_name == "ElasticTransform":
            step = A.ElasticTransform(p=1.0, alpha=(intensity*150), sigma=10, approximate=False) 
        elif aug_name == "MultiplicitiveNoise":
            step = A.MultiplicativeNoise(p=1, multiplier=(1 + intensity * 4, 1 + intensity * 4), per_channel=True, elementwise=True)
        elif aug_name == "GaussianBlur":
            step = A.GaussianBlur(p=1, blur_limit=(blur, blur), sigma_limit=(intensity * 10, intensity * 10))
        elif aug_name == "Sharpen":
            step = A.Sharpen(p=1, alpha=(intensity, intensity), lightness=(1, 1))
        elif aug_name == "MotionBlur":
            step = A.MotionBlur(p=1, blur_limit=(blur, blur), allow_shifted=True)
        elif aug_name == "Downscale":
            step = A.Downscale(p=1, scale_range=(1 - intensity * 0.3, 1 - intensity * 0.3))
        elif aug_name == "SafeRotate":
            step = A.SafeRotate(p=1, limit=(intensity * 359.99, intensity * 359.99), border_mode=1)
        elif aug_name == "RandomSizedCrop":
            min_dim = min(img_width, img_height)
            max_cutoff = 0.5 * min_dim
            step = A.RandomSizedCrop(size=(img_height, img_width), min_max_height=(math.floor(min_dim - intensity * max_cutoff), math.ceil(min_dim - intensity * max_cutoff)), w2h_ratio=1.0)
        elif aug_name == "PixelDropout":
            step = A.PixelDropout(p=1, dropout_prob=intensity * 0.2, per_channel=1)
        elif aug_name == "Superpixels":
            step = A.Superpixels(p=1, p_replace=(intensity * 0.2, intensity * 0.2), n_segments=(500, 1000))
        else:
            raise Exception("Unknown augmentation type " + aug_name)
        return A.Compose([step], **label_opts)

    def _buildPipeline(self, realism: int, img_width: int, img_height: int, min_random_augmentations: int, max_random_augmentations: int):
        label_opts = {} if self.label_format is None else self.label_format
        steps = []
        augs = self._pick_augs(min_random_augmentations, max_random_augmentations)

        for aug_name in augs:
            # Compute max bound based on past err and realism as [0, 1] min/max_bound scalars
            histo = self.analytics["augs"][aug_name]
            first_nonnegligale_intensity = 1.0
            last_noncritical_intensity = 0.0
            # Swim thru the histogram to determine safe intensities
            for point in histo:
                intensity, err = point
                # Consider > 2% error to be when model first starts showing
                # non-negligible differences in output, this way minor issues
                # like slight IoU box area differences don't trigger this too soon
                if err > MIN_NONNEGLIGABLE_ERR and first_nonnegligale_intensity == 1.0:
                    first_nonnegligale_intensity = intensity
                # Consider > 49% error the breaking point for the model, the main
                # reason being that this is the lowest threshold that
                # still supports even the simplest boolean classifier, maybe
                # this should be parameterized for more complex models
                if err < MIN_CRITICAL_ERR:
                    last_noncritical_intensity = intensity
            if first_nonnegligale_intensity >= 1.0:
                first_nonnegligale_intensity = 0.0
                last_noncritical_intensity = 1.0
            half_step = self.analytics["steps"] * 0.5
            # Min bound is set by the first intensity that caused a non-trivial average error,
            # also shift left by half a step to conservatively assume the non-trivial error
            # started somewhere between intensity test X-axis ticks, scale by realism factor
            min_bound = (1 - realism) * max(first_nonnegligale_intensity - half_step, 0)
            # Max bound is set by the last intensity that didn't cause a non-critical average error,
            # also shift left by half a step to conservatively assume the critical error
            # started somewhere between intensity test X-axis ticks, scale by realism factor
            max_bound = (1 - realism) * max(last_noncritical_intensity - half_step, 0)

            # If no real successes for this aug, set to very tiny bounds
            if max_bound <= self.analytics["steps"]:
                min_bound = 0.0001
                max_bound = half_step

            # Potentially override min/max bounds with user override
            if aug_name in self.intensity_overrides:
                override = self.intensity_overrides[aug_name]
                if override["multiplier"] != 1.0:
                    min_bound *= override["multiplier"]
                    max_bound *= override["multiplier"]
                if override["min"] is not None:
                    min_bound = override["min"]
                if override["max"] is not None:
                    if not override["only_shrink_max"] or override["max"] < max_bound:
                        max_bound = override["max"]

            max_255 = min(255, max_bound * 255)
            min_255 = min(255, min_bound * 255)
            max_100 = min(100, max_bound * 100)
            min_100 = min(100, min_bound * 100)
            max_1 = min(1, max_bound)
            min_1 = min(1, min_bound)
            min_blur = round(min_1 * 50) - (round(min_1 * 50) % 2) + 1
            max_blur = round(max_1 * 50) - (round(max_1 * 50) % 2) + 1

            if aug_name == "Darken":
                steps.append(A.RandomBrightnessContrast(p=1, brightness_limit=(-max_1, -min_1)))
            elif aug_name == "Brighten":
                steps.append(A.RandomBrightnessContrast(p=1, brightness_limit=(min_1, max_1)))
            elif aug_name == "Contrast":
                steps.append(A.RandomBrightnessContrast(p=1, contrast_limit=(min_1, max_1)))
            elif aug_name == "Decontrast":
                steps.append(A.RandomBrightnessContrast(p=1, contrast_limit=(-max_1, -min_1)))
            elif aug_name == "Dehue":
                steps.append(A.HueSaturationValue(p=1, hue_shift_limit=(-max_100, -min_100)))
            elif aug_name == "Hue":
                steps.append(A.HueSaturationValue(p=1, hue_shift_limit=(min_100, max_100)))
            elif aug_name == "LessBlue":
                steps.append(A.RGBShift(p=1, r_shift_limit=(-max_255, -min_255)))
            elif aug_name == "LessGreen":
                steps.append(A.RGBShift(p=1, g_shift_limit=(-max_255, -min_255)))
            elif aug_name == "LessRed":
                steps.append(A.RGBShift(p=1, b_shift_limit=(-max_255, -min_255)))
            elif aug_name == "MoreBlue":
                steps.append(A.RGBShift(p=1, r_shift_limit=(min_255, max_255)))
            elif aug_name == "MoreGreen":
                steps.append(A.RGBShift(p=1, g_shift_limit=(min_255, max_255)))
            elif aug_name == "MoreRed":
                steps.append(A.RGBShift(p=1, b_shift_limit=(min_255, max_255)))
            elif aug_name == "Saturate":
                steps.append(A.HueSaturationValue(p=1, sat_shift_limit=(min_100, max_100)))
            elif aug_name == "Desaturate":
                steps.append(A.HueSaturationValue(p=1, sat_shift_limit=(-max_100, -min_100)))
            elif aug_name == "ElasticTransform":
                steps.append(A.ElasticTransform(p=1, alpha=min_1*150 + random.random() * (1 + (max_1 - min_1)*150), sigma=1 + 19 * random.random(), approximate=False))
            elif aug_name == "MultiplicitiveNoise":
                steps.append(A.MultiplicativeNoise(p=1, multiplier=(1 + min_1 * 4, 1 + max_1 * 4), per_channel=True, elementwise=True))
            elif aug_name == "GaussianBlur":
                steps.append(A.GaussianBlur(p=1, blur_limit=(min_blur, max_blur), sigma_limit=(min_1 * 10, max_1 * 10)))
            elif aug_name == "Sharpen":
                steps.append(A.Sharpen(p=1, alpha=(min_1, max_1), lightness=(1, 1)))
            elif aug_name == "MotionBlur":
                steps.append(A.MotionBlur(p=1, blur_limit=(min_blur, max_blur), allow_shifted=True))
            elif aug_name == "Downscale":
                steps.append(A.Downscale(p=1, scale_range=(1 - max_1 * 0.3, 1 - min_1 * 0.3)))
            elif aug_name == "SafeRotate":
                steps.append(A.SafeRotate(p=1, limit=((min_1 * 359.99) % 360, (max_1 * 359.99) % 360), border_mode=1))
            elif aug_name == "RandomSizedCrop":
                min_dim = min(img_width, img_height)
                max_cutoff = 0.5 * min_dim
                steps.append(A.RandomSizedCrop(size=(img_height, img_width), min_max_height=(math.floor(min_dim - max_1 * max_cutoff), math.ceil(min_dim - min_1 * max_cutoff)), w2h_ratio=1.0))
            elif aug_name == "PixelDropout":
                steps.append(A.PixelDropout(p=1, dropout_prob=max_bound * 0.2, per_channel=1))
            elif aug_name == "Superpixels":
                steps.append(A.Superpixels(p=1, p_replace=(min_bound * 0.2, max_bound * 0.2), n_segments=(500, 1000)))
            else:
                raise Exception("Unknown augmentation type " + aug_name)
        return A.ReplayCompose(steps, **label_opts)

    def _synthesize_one(self, img_file, label, pipeline):
        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_args = {} if self.label_format is None else label
        return pipeline(image=image, **label_args)

    def _write_temp(self, img_file: str, img):
         # Temp save locally
        temp_filename = "temp_" + str(random.randint(0, 10000)) + path.basename(img_file)
        cv2.imwrite(temp_filename, img)
        return temp_filename, lambda: os.remove(temp_filename)

    def _runTest(self, img_file, label, pipeline):
        synthetic = self._synthesize_one(img_file, label, pipeline)
        temp_filename, cleanup = self._write_temp(img_file, synthetic["image"])
        predicted = self.predict(temp_filename)
        cleanup()
        return synthetic, predicted

    def _get_dims_for_file(self, file: str):
        img = cv2.imread(file)
        return (img.shape[1], img.shape[0])

    # Returns a single scalar in [0, 1] of observed error sum divided by max possible error
    # to give a single percentage representation of how much this aug at this intensity
    # breaks my_predict
    def _measureErrorAtTickForAug(self, intensity: float, aug: str, img_filenames: list[str], labels):
        img_dims = self._get_dims_for_file(img_filenames[0])
        pipeline = self._buildBoundryTestPipeline(img_dims[0], img_dims[1], intensity, aug)
        first_err_img = None
        first_err_err = None
        first_err_label = None
        first_err_dims = None
        last_success_img = None
        last_success_label = None
        last_success_dims = None
        err_sum = 0.0
        for i in range(len(img_filenames)):
            file = img_filenames[i]
            test, prediction = self._runTest(file, labels[i], pipeline)
            # Re-use original unchanged labels unless format is set
            # in which case use the calculated synthetic labels
            test_labels = labels[i]
            if self.label_format is not None:
                # Remove the image and anything left over is labels
                test_labels = test.copy()
                del test_labels["image"]
            
            err = self.diff_error(test_labels, prediction)
            err_sum += err
            if err >= MIN_NONNEGLIGABLE_ERR and first_err_img is None:
                first_err_img = file
                first_err_err = err
                first_err_label = labels[i]
                first_err_dims = self._get_dims_for_file(file)
            if err < MIN_NONNEGLIGABLE_ERR:
                last_success_img = file
                last_success_label = labels[i]
        return {
            "first_err": {
                "img": first_err_img,
                "err": first_err_err,
                "label": first_err_label,
                "intensity": intensity,
                "dims": first_err_dims,
            },
            "last_success": {
                "img": last_success_img,
                "label": last_success_label,
                "intensity": intensity,
                "dims": None if last_success_img is None else self._get_dims_for_file(last_success_img)
            },
            "err": err_sum / len(img_filenames)
        }
    
    def _renderAugRow(self, aug_name: str, body: list[str], script: list[str], html_dir: str):
        aug_histogram = self.analytics["augs"][aug_name]
        summary = self.analytics["summaries"][aug_name]

        body.append("<tr>")
        body.append("<td>" + aug_name + "</td>")
        body.append("<td><div class='canvas'><canvas id=\"" + aug_name.lower() + "\"></canvas></div>")
        body.append("</td>")
        if summary["first_err"] is not None:
            first_err = summary["first_err"]
            # Draw the image to file
            pipeline = self._buildBoundryTestPipeline(first_err["dims"][0], first_err["dims"][1], first_err["intensity"], aug_name)
            test = self._synthesize_one(first_err["img"], first_err["label"], pipeline)
            synth_img_name = aug_name + "_" + str(round(first_err["intensity"] * 100)) + "_errors_start_" + path.basename(first_err["img"])
            rendered_img_path = path.join(html_dir, "imgs", synth_img_name)
            cv2.imwrite(rendered_img_path, test["image"])
            relative_img_path = "imgs/" + synth_img_name
            body.append("<td>" + str(round(first_err["intensity"] * 100)) + "%</td>")
            body.append("<td><img src=\"" + relative_img_path + "\" /></td>")
        else:
            body.append("<td>No Errors!</td>")
            body.append("<td>No Errors!</td>")
        if summary["last_success"] is not None:
            last_success = summary["last_success"]
            # Draw the image to file
            pipeline = self._buildBoundryTestPipeline(last_success["dims"][0], last_success["dims"][1], last_success["intensity"], aug_name)
            test = self._synthesize_one(last_success["img"], last_success["label"], pipeline)
            synth_img_name = aug_name + "_at_" + str(round(last_success["intensity"] * 100)) + "_still_good_" + path.basename(last_success["img"])
            rendered_img_path = path.join(html_dir, "imgs", synth_img_name)
            cv2.imwrite(rendered_img_path, test["image"])
            relative_img_path = "imgs/" + synth_img_name
            body.append("<td>" + (str(round(last_success["intensity"] * 100))) + "%</td>")
            body.append("<td><img src=\"" + relative_img_path + "\" /></td>")
        else:
            body.append("<td>No Successes</td>")
            body.append("<td>No Successes</td>") 
        body.append("</tr>")
        graph_data = {
            "labels": [point[0] for point in aug_histogram], 
            "datasets": [{"label": "Diff Error", "data": [point[1]*100 for point in aug_histogram], "borderColor": "#cc3333", "fill": False, "tension": 0.4}]}
        graph_config = {
            "type": 'line',
            "data": graph_data,
            "options": {
                "responsive": True,
                "plugins": {
                "title": {
                    "display": False,
                    "text": ''
                },
                },
                "interaction": {
                "intersect": False,
                },
                "scales": {
                "x": {
                    "display": True,
                    "title": {
                    "display": True,
                    "text": 'Intensity'
                    }
                },
                "y": {
                    "display": True,
                    "title": {
                    "display": True,
                    "text": "Output Diff Error"
                    },
                    "suggestedMin": 0,
                    "suggestedMax": 100
                }
                }
            }
        } 
        script.append("const " + aug_name.lower() + " = document.getElementById('" + aug_name.lower() + "');")
        script.append(f'''new Chart({aug_name.lower()}, {json.dumps(graph_config)});''')

    def _is_critical_contrast_drop(self, orig_img, synthetic_img):
        orig_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        synthetic_gray = cv2.cvtColor(synthetic_img, cv2.COLOR_BGR2GRAY)
        orig_std = orig_gray.std()
        synthetic_std = synthetic_gray.std()
        contrast_ratio = synthetic_std / orig_std

        # Filter if abs contrast diff dropped below threshold
        if synthetic_std < 5 and orig_std > 5:
            return True
        # Filter if relative contrast ratio dropped below 5% of original contrast
        if contrast_ratio < 0.05:
            return True
        return False

    def set_augmentation_intensity(self, augmentations: str | list[str] = ALL_AUGMENTATIONS, multiplier: float = 1.0, min: int = None, max: int = None, only_shrink_max: bool = False):
        """
        Sets the min/max [0, 1] scalar intensities of an augmentation(s) being applied.  You can view the webpage output from calling
        `render_boundries` to easily view what intensities affect your model and how and intelligently set this.

        `multiplier`: (Optional) Use a single scalar to boost or penalize the default min/max learned intensity boundries. You should prefer
        this over setting min/max since it's a single value and it still leverages the good intensity ranges found for your model+dataset.

        `min`: (Optional) The minimum amount of effect intensity to apply for `augmentations` when generating synthetics.
        Default: The intensity at which my_predict begins to show >= 2% label differences / error.

        `max`: (Optional) The maximum amount of effect intensity to apply for `augmentations` when generating synthetics.
        Default: The intensity at which my_predict begins to show >= 50% label differences / error.

        `augmentations`: (Optional) A list or single string of augmentation name(s) to apply the min and/or max intensity to.
        Default: ALL_AUGMENTATIONS

        `only_shrink_max`: (Optional) Whether this call should only shrink the range of possible intensity rather than accidently expand it.
        Use this if you want to make sure you don't accidently set a higher intensity than what your model can actually handle.
        Default: False, affected augmentations will strictly use your min/max values if set.
        """
        if min is not None and (min < 0 or min >= 1):
            raise Exception("set_augmentation_intensity min must be >= 0 and < 1")
        if max is not None and (max > 1 or max <= 0):
            raise Exception("set_augmentation_intensity max must be > 0 and <= 1")
        if multiplier is not None and multiplier <= 0:
            raise Exception("set_augmentation_intensity multiplier must be > 0")
        augs = [augmentations] if isinstance(augmentations, str) else augmentations
        for aug in augs:
            self.intensity_overrides[aug] = {"min": min, "max": max, "multiplier": multiplier, "only_shrink_max": only_shrink_max}

    def set_augmentation_weight(self, augmentations: str | list[str], weight: int):
        """
        Sets the probability of an augmentation being applied.  weight is relative
        to other augmentations which are typically 1 (uniform distribution), so a value 
        of 2 would double the odds of selection relative to others while 0.5 cuts in half

        `augmentations` Single str or list of str of augmentation names to apply this weight to.

        `weight`: How often this augmentation should be applied [0, inf], value of 1 is uniform, 2 would double the selection likelihood etc.
        """
        if weight <= 0:
            raise Exception("weight must be positive")
        augs = augmentations if hasattr(augmentations, "__len__") else [augmentations]
        for aug in augs:
            self.probs[aug] = weight

    def search_randomization_boundries(self, training_img_filenames: list[str], training_labels: list = None, step_size_percent: float=0.05, analytics_cache="analytics.json"):
        """
        The main method that examines all training sample images passed in, usually everything you've got.  Because it takes a long time to run, it stores intermediate and final results to `analytics.json` which you can delete manually to find boundries from scratch (ex. you collected more training data and want to re-run). You should generally feed in as many training_img_filenames as possible to strengthen boundry search confidence and ensure future generated data isn't too unrealistic.  On the other hand you may want to limit to ~50000 maximally diverse training samples so analysis completes faster but only if time is a virtue for you.

        All other class methods assume you've run this and already computed boundry state.

        `training_img_filenames`: List of image filenames that will later be passed to `my_predict`

        `training_labels`: (Optional) List of ground truth labels (type agnostic) that match 1-to-1 with `training_img_filenames`.  Required if you defined my_predict, otherwise defaults to labels derived under the hood.

        `step_size_percent`: (Optional) How big of steps to take when finding an augmentation feature's limit, default of 0.05 means each trial will increase augmentation intensity by 5% until `my_predict` starts to differ significantly in its output.  Lower values take longer for the 1-time cost of running search_randomization_boundries but will yield more accurate augmentation limit boundries for data generation and graphing, so going down to ~1% step_size granularity can sometimes be worth the investment.

        `analytics_cache`: The filename to use to store search cache calculations, defaults to analytics.json
        """

        labels = training_labels
        # If my_predict was not set, train a default SIFT predictor from this first seen training img batch
        if self.my_predict is None:
            labels = [{"out": i, "γ": 1.0} for i in range(len(training_img_filenames))]
            self.predict = new_default_predictor(training_img_filenames)
            self.diff_error = confidence_aware_diff_error
            print("Using default my_predict edge-based model for finding randomization boundries trained on " + str(len(training_img_filenames)) + " samples")
        set_size = len(training_img_filenames)
        temp_analytics = {"steps": step_size_percent, "set_size": set_size, "augs": {}, "summaries": {}}
        if path.exists(analytics_cache):
            with open(analytics_cache, 'r') as inputf:
                loaded = json.load(inputf)
                if loaded["steps"] == step_size_percent:
                    if loaded["set_size"] == set_size:
                        temp_analytics = loaded
                        print("Resuming progress from " + analytics_cache)
                    else:
                        print("Training set size changed, searching boundries from scratch")
                else:
                    print("Step size changed, searching boundries from scratch")
        else:
            print("Starting random boundry search using " + str(len(training_img_filenames)) + " samples")

        augs = temp_analytics["augs"]
        summaries = temp_analytics["summaries"]

        for aug_name in self.augmentations:
            first_err = None
            last_success = None
            if aug_name not in augs:
                augs[aug_name] = []
            else:
                continue
            aug_histogram = augs[aug_name]
            cur_intensity = step_size_percent
            while (cur_intensity <= 1):
                err_info = self._measureErrorAtTickForAug(cur_intensity, aug_name, training_img_filenames, labels)
                err_ratio = err_info["err"]
                if first_err is None and err_info["first_err"]["img"] is not None:
                    first_err = err_info["first_err"]
                if err_info["last_success"]["img"] is not None:
                    last_success = err_info["last_success"]
                aug_histogram.append([cur_intensity, err_ratio])
                cur_intensity = round(cur_intensity + step_size_percent, 6)
            # Store summaries for fast rendering
            summaries[aug_name] = {
                "first_err": first_err,
                "last_success": last_success,
            }
            # Checkpoint progress
            with open(analytics_cache, "w") as j:
                json.dump(temp_analytics, j, indent=2)
                print("Checkpoint: Analyzed " + aug_name + "'s acceptable intensity ranges")
        self.analytics = temp_analytics
        return self.analytics

    def render_boundries(self, html_dir="analytics"):
        """
        Render the results of `search_randomization_boundries` to HTML for easy visualization of how your model performs against varying degrees of augmention.

        `html_dir`: (Optional) The directory to write output HTML and image files to, defaults to "analytics" relative directory.
        """
        if self.analytics is None:
            raise Exception("Cannot call before search_randomization_boundries")

        script = []
        body = ["<h1>Model Resiliency Report</h1>"]
        body.append("<p>The table below summarizes how your model performs at different intensities of image augmentation.  You can view generated samples that began giving your model problems up to the last intensity where a majority of model outputs were the same.</p>")
        body.append("<table><tr><th>Augmentation</th><th>Intensity vs Diff Error</th><th>First Error Intensity</th><th>First Error</th><th>Last Success Intensity</th><th>Last Success</th></tr>")
        for aug_name in self.analytics["augs"].keys():
            self._renderAugRow(aug_name, body, script, html_dir)
        body.append("</table>")

        html = "<html><head><script src=\"chart.js\"></script><link rel=\"stylesheet\" href=\"style.css\"><title>Model Resiliency report</title></head><body>\n" + \
        "\n".join(body) + \
        "<script>" + \
        "\n".join(script) + \
        "</script>" + \
        "\n</body></html>"
        os.makedirs(html_dir, exist_ok=True)
        os.makedirs(path.join(html_dir, "imgs"), exist_ok=True)
        full_path = path.join(html_dir, "index.html")
        with open(full_path, "w") as f:
            f.write(html)
            print("Wrote " + full_path + ", open to view results")
        shutil.copy(path.join("assets", "style.css"), path.join(html_dir, "style.css"))
        shutil.copy(path.join("assets", "chart.js"), path.join(html_dir, "chart.js"))

    def _is_valid_synthetic(self, organic_img, organic_labels, synthetic, min_predicted_diff_error, max_predicted_diff_error):
        filter_by_prediction = min_predicted_diff_error > 0 or max_predicted_diff_error < 1
        if self._is_critical_contrast_drop(organic_img, synthetic["image"]):
            return False

        # Re-use original unchanged labels unless format is set
        # in which case use the calculated synthetic labels
        synthetic_labels = organic_labels
        if self.label_format is not None:
            # Remove the image+replay and anything left over is labels
            synthetic_labels = synthetic.copy()
            del synthetic_labels["image"]
            del synthetic_labels["replay"]

        if filter_by_prediction:
            temp_filename, cleanup = self._write_temp(organic_img, synthetic["image"])
            predicted = self.predict(temp_filename)
            cleanup()
            err = self.diff_error(organic_labels, predicted)
            if err > max_predicted_diff_error or err < min_predicted_diff_error:
                return False
        return True

    def _upsert_and_get_synthetic_log(self, synth_log_file: str, this_run: list[list]):
        gen_imgs, gen_labels, gen_origins, gen_origin_labels = this_run
        prev_json = [[], [], [], []]
        if path.exists(synth_log_file):
            with open(synth_log_file, "r") as f:
                prev_json = json.load(f)
        prev_imgs, prev_labels, prev_origins, prev_origin_labels = prev_json

        new_json = [
            prev_imgs + gen_imgs, 
            prev_labels + gen_labels, 
            prev_origins + gen_origins, 
            prev_origin_labels + gen_origin_labels]
        with open(synth_log_file, "w") as f:
            json.dump(new_json, f)
        return new_json

    def _raw_summary_to_human(self, raw_summary):
        count = len(raw_summary[0])
        return {
                "count": count,
                "generated": [{
                    "synth_filename": raw_summary[0][e],
                    "synth_label": raw_summary[1][e],
                    "origin_filename": raw_summary[2][e],
                    "origin_label": raw_summary[3][e]
                } for e in range(count)]
            }

    def get_synthesized_summary(self, log_file="", output_dir="generated"):
        """
        Get a summary of all images generated so far, useful for conditionally
        calling downstream methods only if needed or iterating over all historically
        generated image locations and label metadata.  

        `log_file`: (Optional) The filename of the JSON logs of what has been synthesized, useful for synthetic generation checkpointing.  Defaults to the output directory name but with ".json" appended, set to None to avoid writing this file at all however `count_by`="in_dir" will be disabled.

        `output_dir`: (Optional) The folder to save images and `__preview.html` to.
        
        Return object shape:

        ```
        {
            "count": int,
            "generated": [{
                "synth_filename": str,
                "synth_label": any,
                "origin_filename": str,
                "origin_label": any
            }]
        }
        ```
        """
        manifest = log_file if log_file != "" else (output_dir + ".json")
        with open(manifest, "r") as f:
            raw = json.load(f)
            return self._raw_summary_to_human(raw)

    def synthesize_more(self, organic_img_filenames: list[str], organic_labels: list = None, \
                       realism=0.5, count=None, count_by="per_call", min_random_augmentations=3, max_random_augmentations=6, \
                       min_predicted_diff_error=0, max_predicted_diff_error=1, \
                       image_namer = verbose_synthetic_namer, log_file="", output_dir="generated", preview_html="__preview.html"):
        """
        Generate synthetic training/validation samples based on some input set and only use as much randomization as `realism` demands.  Optionally generates a `__preview.html` file that previews all images in the generated output folder.

        Returns a quad-tuple of (generated image filenames, generated image labels, generated image's organic source file, generated image's organic label).  If `count_by`="in_dir" is set, the return values will reflect ALL synthesized images ever generated in output_dir.

        `organic_img_filenames`: Original (presumably real-world) training images from which to synthesize new datasets, each image will be used in equal quantity.

        `organic_labels` (Optional) List of ground-truth (presumably real-world) training labels that map 1-to-1 with `organic_img_filenames`.  Required if my_predict is specified otherwise defaults to derived labels under the hood.

        `realism`: (Optional) A float between [-∞, 1] to control generated images' realism based on what your model could handle during boundry search.  A value of 1 means to steer clear of more intense random values that your model has trouble with while a value of zero pushes to the very limit of what your model can tolerate.  Negative values push your current model well into failure territory but may be useful to generate synthetic training data for generalization of your model after retraining.

        `count`: (Optional) Number of synthetic images to generate, default of None signifies to use len(training_img_filenames)

        `count_by`: (Optional) Whether the `count` parameter should match `in_dir` in-directory total image count (checkpoint-friendly) or `per_call` absolute generate count. Defaults to `per_call`

        `min_random_augmentations`: (Optional) Randomly pick at least this many augmentations to apply.  Default: 3

        `max_random_augmentations`: (Optional) Randomly pick at most this many augmentations to apply.  Default: 6

        `min_predicted_diff_error`: (Optional) The minimum diff error between `my_predict` running on original image vs augmented image. Set this if you only want to keep generated images that your model fails at to force it to focus on the outliers it misses.  Defaults to zero so all synthesized data is kept.

        `max_predicted_diff_error`: (Optional) The maximum diff error between `my_predict` running on original image vs augmented image.  Set this if you want to filter out images that _may_ differ too wildly from the original image.  Useful for auto-removing images that end up for example too bright for _any_ model to process; such images can potentially weaken the synthetic dataset for training purposes or make validation on synthetics appear artifically poor.  Defaults to 1 so all synthesized data is kept.

        `image_namer`: (Optional) function that returns the relative image name based on: (raw input relative image path, matching label, uid, applied Albumentation transform summary)

        `log_file`: (Optional) The filename of the JSON logs of what has been synthesized, useful for synthetic generation checkpointing.  Defaults to the output directory name but with ".json" appended, set to None to avoid writing this file at all however `count_by`="in_dir" will be disabled then.

        `output_dir`: (Optional) The folder to save images and `__preview.html` to.

        `preview_html`: (Optional) The name of the HTML file that will summarize synthetic images in `output_dir`, defaults to `__preview.html`.  Set to None to disable summarization.

        Returned summary object shape:

        ```
        {
            "count": int,
            "generated": [{
                "synth_filename": str,
                "synth_label": any,
                "origin_filename": str,
                "origin_label": any
            }]
        }
        ```
        """
        if self.analytics is None:
            raise Exception("Cannot call before search_randomization_boundries")
        os.makedirs(output_dir, exist_ok=True)
        uid = 0
        existing_count = 0

        # Merge in previous synth runs if need be
        manifest = log_file if log_file != "" else (output_dir + ".json")
        if count_by == "in_dir" and path.exists(manifest):
            with open(manifest, "r") as f:
               past_gen_imgs = json.load(f)[0]
               existing_count = len(past_gen_imgs)
               uid = existing_count

        gen_count = count if count is not None else len(organic_img_filenames)
        gen_imgs = []
        gen_labels = []
        gen_origins = []
        gen_origin_labels = []
        # Lazy eval labels if using default my_predict model
        derived_label_cache = {}
        cur_original_index = 0

        while ((len(gen_imgs) + existing_count) < gen_count):
            organic_file = organic_img_filenames[cur_original_index]
            organic_label = organic_labels[cur_original_index] if organic_labels is not None else \
                (derived_label_cache[organic_file] if organic_file in derived_label_cache else self.predict(organic_file))
            derived_label_cache[organic_file] = organic_label
            label_args = {} if self.label_format is None else organic_label

            # For each input image roll the dice up to N times
            # to find an acceptable augmented image before warning
            # and moving on to the next input image
            max_dice_rolls = 100
            synthetic_img = None
            synthetic_label = None
            synthetic_name = None
            for __ in range(max_dice_rolls):
                base_img = cv2.imread(organic_file)
                base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
                pipeline = self._buildPipeline(realism, base_img.shape[1], base_img.shape[0], min_random_augmentations, max_random_augmentations)
                synthetic = pipeline(image=base_img, **label_args)

                if self._is_valid_synthetic(base_img, organic_label, synthetic, min_predicted_diff_error, max_predicted_diff_error):
                    synthetic_img = synthetic["image"]
                    result_augs = [aug for aug in synthetic['replay']["transforms"]]
                    applied_augs = filter(lambda aug: aug["applied"], result_augs)
                    synthetic_name = image_namer(path.basename(organic_file), organic_label, uid, applied_augs)

                    # Re-use original unchanged labels unless format is set
                    # in which case use the calculated synthetic labels
                    synthetic_label = organic_label
                    if self.label_format is not None:
                        # Remove the image+replay and anything left over is labels
                        synthetic_label = synthetic.copy()
                        del synthetic_label["image"]
                        del synthetic_label["replay"]
                    
                    uid += 1
                    break
            
            cur_original_index = (cur_original_index+1) % len(organic_img_filenames)
            if synthetic_img is None:
                print("Warn: Could not generate valid synthetic from " + organic_file + ", skipping file this iteration")
            else:
                synthetic_path = path.join(output_dir, synthetic_name)
                cv2.imwrite(synthetic_path, synthetic_img)
                gen_imgs.append(synthetic_path)
                gen_labels.append(synthetic_label)
                gen_origins.append(organic_file)
                gen_origin_labels.append(organic_label)

        if preview_html is not None:
            try:
                os.unlink(preview_html)
            except:
                pass
            html = ["<html><head><title>Synthetic Image Grid</title></head><body><div class=\"grid\">"]
            discovered_file_count = 0
            for file in os.listdir(output_dir):
                if file.endswith(".html"):
                    continue
                html.append("<div class=\"square\">")
                html.append(f'''<a href="{file}"><img src="{file}" loading="lazy" /></a>''')
                html.append("</div>")
                discovered_file_count += 1
            css = ""
            with open(path.join("assets", "style.css"), "r") as css_file:
                css = css_file.read()
            html.append("</div>")
            html.append("<style>\n" + css + "\n</style>")
            html.append("</body></html>")
            preview_path = path.join(output_dir, preview_html)
            with open(preview_path, "w") as preview_file:
                preview_file.write("\n".join(html))
            print("Open " + preview_path + " to view all " + str(discovered_file_count) + " synthetic images")
        this_run = [gen_imgs, gen_labels, gen_origins, gen_origin_labels]
        if log_file is not None:
            log_file = log_file if log_file != "" else (output_dir + ".json")
            all_logs = self._upsert_and_get_synthetic_log(log_file, this_run)
            if count_by == "in_dir":
                return self._raw_summary_to_human(all_logs)
        return self._raw_summary_to_human(this_run)

    def evaluate(self, img_filenames, img_labels):
        """
        Run `my_predict` against all img_filenames which are 
        typically generated by `synthesize_more` as well as the 
        matching synthetic truth labels and compare to the model's 
        output labels ran against img_filenames.  This is 
        convenient to test synthetic data against different 
        versions of your model, presumably your model before and 
        after training on the synthetic dataset.  Can also be used 
        to compare real-world vs synthetic model performance.  If 
        your new model performs poorly on a synthetic batch it 
        was trained on, it suggests your `realism` hyperparameter 
        may be too high and you're randomizing training data to 
        the point of mangling it for even the best model (ex. so 
        much extra brightness the image is pure white).  If your 
        model performs extremely well on a synthetic batch it was 
        trained on while retaining real-world accuracy, consider 
        increasing `realism` to handle more real-world edge cases 
        by training on even stranger synthetic samples.  Also 
        consider setting `max_predicted_diff_error` to < 1 to 
        task your model with filtering out overly unrealistic 
        synthetic samples.

        Returns an object containing:

        ```
        {
            "avg_diff_error": Average diff error across all evaluated samples,
            "output_differs_count": Count of outputs that differed from the label significantly,
            "differing_output_errs": All output errors
        }
        ```

        `img_filenames`: list of string filenames to use in evaluation.

        `img_labels`: The 1-to-1 matching labels of `img_filenames`.
        """
        err_sum = 0.0
        output_differs_count = 0
        differing_outputs = {}
        for i in range(len(img_filenames)):
            img = img_filenames[i]
            truth = img_labels[i]
            predicted = self.predict(img)
            err = self.diff_error(truth, predicted)
            err_sum += err
            if err > 0:
                output_differs_count += 1
                differing_outputs[img] = err
        return {
            "avg_diff_error": err_sum / len(img_filenames),
            "output_differs_count": output_differs_count,
            "differing_output_errs": differing_outputs
        }
