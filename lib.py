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
    # TODO "ElasticTransform",
    "GaussianBlur",
    "MotionBlur",
    "SafeRotate",
    "Sharpen",
]

PIXEL_DROPOUT_AUGMENTATIONS = [
    "Downscale",
    "MultiplicitiveNoise",
    "PixelDropout",
    "RandomCropFromBorders",
    "Superpixels"
]

ALL_AUGMENTATIONS=[
    *COLOR_AUGMENTATIONS,
    *DISTORTION_AUGMENTATIONS,
    *PIXEL_DROPOUT_AUGMENTATIONS
]

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
    "ElasticTransform": ["RandomCropFromBorders"],
    "RandomCropFromBorders": ["ElasticTransform"],
    "MotionBlur": ["GaussianBlur"],
    "GaussianBlur": ["MotionBlur"]
}

# Diff errors beyond 2% are considered non-negligible
MIN_NONNEGLIGABLE_ERR = 0.02
# Diff errors > 90% are considered critical failure
MIN_CRITICAL_ERR = 0.49

# Naively assume string equality is no error otherwise max error
def err_if_not_strict_eq(original_labels, aug_labels):
    return 0 if str(original_labels) == str(aug_labels) else 1

class ImageAugmenter:
    def __init__(self, my_predict: lambda img_filename: str, diff_error: lambda orig, augmented: float=err_if_not_strict_eq, label_format=None, augmentations: list[str]=ALL_AUGMENTATIONS):
        self.my_predict = my_predict
        self.diff_error = diff_error
        self.augmentations = augmentations
        self.analytics = None
        self.label_format = label_format
        self.realism_overrides = {}

    def _buildBoundryTestPipeline(self, intensity: float, aug_name: str):
        label_opts = {} if self.label_format is None else self.label_format
        step = None
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
            step = A.ElasticTransform(p=1.0, alpha=intensity, sigma=intensity*9, alpha_affine=intensity*50, border_mode=1, approximate=False) 
        elif aug_name == "MultiplicitiveNoise":
            step = A.MultiplicativeNoise(p=1, multiplier=(1 + intensity * 4, 1 + intensity * 4), per_channel=True, elementwise=True)
        elif aug_name == "GaussianBlur":
            step = A.GaussianBlur(p=1, blur_limit=(round(intensity * 50), round(intensity * 50)), sigma_limit=(intensity * 10, intensity * 10))
        elif aug_name == "Sharpen":
            step = A.Sharpen(p=1, alpha=(intensity, intensity), lightness=(1, 1))
        elif aug_name == "MotionBlur":
            step = A.MotionBlur(p=1, blur_limit=(round(intensity * 50), round(intensity * 50)), allow_shifted=True)
        elif aug_name == "Downscale":
            step = A.Downscale(p=1, scale_min=(1 - intensity * 0.1), scale_max=(1 - intensity * 0.1))
        elif aug_name == "SafeRotate":
            step = A.SafeRotate(p=1, limit=(intensity * 359.99, intensity * 359.99), border_mode=1)
        elif aug_name == "RandomCropFromBorders":
            step = A.RandomCropFromBorders(p=1, crop_left=intensity * 0.5, crop_right=intensity * 0.5, crop_top=intensity * 0.5, crop_bottom=intensity * 0.5)
        elif aug_name == "PixelDropout":
            step = A.PixelDropout(p=1, dropout_prob=intensity * 0.5, per_channel=1)
        elif aug_name == "Superpixels":
            step = A.Superpixels(p=1, p_replace=(intensity * 0.5, intensity * 0.5), n_segments=(500, 1000))
        else:
            raise Exception("Unknown augmentation type " + aug_name)
        return A.Compose([step], **label_opts)

    def _buildPipeline(self, realism: int, min_random_augmentations: int, max_random_augmentations: int):
        label_opts = {} if self.label_format is None else self.label_format
        steps = []

        shuffled_augs = [x for x in self.analytics["augs"].keys()]
        random.shuffle(shuffled_augs)
        aug_count = random.randint(min_random_augmentations, max_random_augmentations)

        # Remove duplicate or contradictory aug types, preferring the first
        aug_set = {}
        for aug in shuffled_augs:
            bans = BANNED_PAIRS[aug] if aug in BANNED_PAIRS else []
            banned = False
            for ban in bans:
                if ban in aug_set:
                    banned = True
            if not banned:
                aug_set[aug] = True
            if len(aug_set) == aug_count:
                break
        augs = aug_set.keys()

        for aug_name in augs:
            realism_to_use = realism if aug_name not in self.realism_overrides else self.realism_overrides[aug_name]
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
            min_bound = (1 - realism_to_use) * max(first_nonnegligale_intensity - half_step, 0)
            # Max bound is set by the last intensity that didn't cause a non-critical average error,
            # also shift left by half a step to conservatively assume the critical error
            # started somewhere between intensity test X-axis ticks, scale by realism factor
            max_bound = (1 - realism_to_use) * max(last_noncritical_intensity - half_step, 0)

            # If no real successes for this aug, set to very tiny bounds
            if max_bound <= self.analytics["steps"]:
                min_bound = 0.0001
                max_bound = half_step

            if aug_name == "Darken":
                steps.append(A.RandomBrightnessContrast(p=1, brightness_limit=(-max_bound, -min_bound)))
            elif aug_name == "Brighten":
                steps.append(A.RandomBrightnessContrast(p=1, brightness_limit=(min_bound, max_bound)))
            elif aug_name == "Contrast":
                steps.append(A.RandomBrightnessContrast(p=1, contrast_limit=(min_bound, max_bound)))
            elif aug_name == "Decontrast":
                steps.append(A.RandomBrightnessContrast(p=1, contrast_limit=(-max_bound, -min_bound)))
            elif aug_name == "Dehue":
                steps.append(A.HueSaturationValue(p=1, hue_shift_limit=(max_bound * -100, min_bound * -100)))
            elif aug_name == "Hue":
                steps.append(A.HueSaturationValue(p=1, hue_shift_limit=(min_bound * 100, max_bound * 100)))
            elif aug_name == "LessBlue":
                steps.append(A.RGBShift(p=1, r_shift_limit=(max_bound * -255, min_bound * -255)))
            elif aug_name == "LessGreen":
                steps.append(A.RGBShift(p=1, g_shift_limit=(max_bound * -255, min_bound * -255)))
            elif aug_name == "LessRed":
                steps.append(A.RGBShift(p=1, b_shift_limit=(max_bound * -255, min_bound * -255)))
            elif aug_name == "MoreBlue":
                steps.append(A.RGBShift(p=1, r_shift_limit=(min_bound * 255, max_bound * 255)))
            elif aug_name == "MoreGreen":
                steps.append(A.RGBShift(p=1, g_shift_limit=(min_bound * 255, max_bound * 255)))
            elif aug_name == "MoreRed":
                steps.append(A.RGBShift(p=1, b_shift_limit=(min_bound * 255, max_bound * 255)))
            elif aug_name == "Saturate":
                steps.append(A.HueSaturationValue(p=1, sat_shift_limit=(min_bound * 100, max_bound * 100)))
            elif aug_name == "Desaturate":
                steps.append(A.HueSaturationValue(p=1, sat_shift_limit=(max_bound * -100, min_bound * -100)))
            elif aug_name == "ElasticTransform":
                steps.append(A.ElasticTransform(always_apply=False, p=1.0, alpha=(0, max_bound), sigma=(min_bound*9, max_bound*9), alpha_affine=(min_bound*50, max_bound*50), interpolation=0, border_mode=1, approximate=False, same_dxdy=False))
            elif aug_name == "MultiplicitiveNoise":
                steps.append(A.MultiplicativeNoise(p=1, multiplier=(1 + min_bound * 4, 1 + max_bound * 4), per_channel=True, elementwise=True))
            elif aug_name == "GaussianBlur":
                steps.append(A.GaussianBlur(p=1, blur_limit=(round(min_bound * 50), round(max_bound * 50)), sigma_limit=(min_bound * 10, max_bound * 10)))
            elif aug_name == "Sharpen":
                steps.append(A.Sharpen(p=1, alpha=(min_bound, max_bound), lightness=(1, 1)))
            elif aug_name == "MotionBlur":
                steps.append(A.MotionBlur(p=1, blur_limit=(round(min_bound * 50), round(max_bound * 50)), allow_shifted=True))
            elif aug_name == "Downscale":
                steps.append(A.Downscale(p=1, scale_min=(1 - max_bound * 0.1), scale_max=(1 - min_bound * 0.1)))
            elif aug_name == "SafeRotate":
                steps.append(A.SafeRotate(p=1, limit=((min_bound * 359.99) % 360, (max_bound * 359.99) % 360), border_mode=1))
            elif aug_name == "RandomCropFromBorders":
                steps.append(A.RandomCropFromBorders(p=1, crop_left=max_bound * 0.5, crop_right=max_bound * 0.5, crop_top=max_bound * 0.5, crop_bottom=max_bound * 0.5))
            elif aug_name == "PixelDropout":
                steps.append(A.PixelDropout(p=1, dropout_prob=max_bound * 0.5, per_channel=1))
            elif aug_name == "Superpixels":
                steps.append(A.Superpixels(p=1, p_replace=(min_bound * 0.5, max_bound * 0.5), n_segments=(500, 1000)))
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
        temp_filename = "temp." + path.splitext(img_file)[-1]
        cv2.imwrite(temp_filename, img)
        return temp_filename, lambda: os.remove(temp_filename)

    def _runTest(self, img_file, label, pipeline):
        synthetic = self._synthesize_one(img_file, label, pipeline)
        temp_filename, cleanup = self._write_temp(img_file, synthetic["image"])
        predicted = self.my_predict(temp_filename)
        cleanup()
        return synthetic, predicted

    # Returns a single scalar in [0, 1] of observed error sum divided by max possible error
    # to give a single percentage representation of how much this aug at this intensity
    # breaks my_predict
    def _measureErrorAtTickForAug(self, intensity: float, aug: str, img_filenames: list[str], labels):
        pipeline = self._buildBoundryTestPipeline(intensity, aug)
        first_err_img = None
        first_err_err = None
        first_err_label = None
        last_success_img = None
        last_success_label = None
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
            if err < MIN_NONNEGLIGABLE_ERR:
                last_success_img = file
                last_success_label = labels[i]
        return {
            "first_err": {
                "img": first_err_img,
                "err": first_err_err,
                "label": first_err_label,
                "intensity": intensity
            },
            "last_success": {
                "img": last_success_img,
                "label": last_success_label,
                "intensity": intensity
            },
            "err": err_sum / len(img_filenames)
        }

    def set_realism_for(self, augmentation_name: str, realism: float):
        self.realism_overrides[augmentation_name] = realism

    def searchRandomizationBoundries(self, training_img_filenames: list[str], training_labels, step_size_percent: float=0.05):
        print("Starting random boundry search using " + str(len(training_img_filenames)) + " samples")
        set_size = len(training_img_filenames)
        temp_analytics = {"steps": step_size_percent, "set_size": set_size, "augs": {}, "summaries": {}}
        if path.exists("analytics.json"):
            with open('analytics.json', 'r') as inputf:
                loaded = json.load(inputf)
                if loaded["steps"] == step_size_percent:
                    if loaded["set_size"] == set_size:
                        temp_analytics = loaded
                        print("Resuming progress from analytics.json")
                    else:
                        print("Training set size changed, searching boundries from scratch")
                else:
                    print("Step size changed, searching boundries from scratch")

        augs = temp_analytics["augs"]
        summaries = temp_analytics["summaries"]

        for aug_name in self.augmentations:
            first_err = None
            last_success = None
            if aug_name not in augs:
                augs[aug_name] = []
            else:
                print("Skipping " + aug_name + ", already processed")
                continue
            aug_histogram = augs[aug_name]
            cur_intensity = step_size_percent
            while (cur_intensity <= 1):
                err_info = self._measureErrorAtTickForAug(cur_intensity, aug_name, training_img_filenames, training_labels)
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
            with open("analytics.json", "w") as j:
                json.dump(temp_analytics, j, indent=2)
                print("Checkpoint: Analyzed " + aug_name + "'s acceptable intensity ranges")
        self.analytics = temp_analytics
        return self.analytics

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
            pipeline = self._buildBoundryTestPipeline(first_err["intensity"], aug_name)
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
            pipeline = self._buildBoundryTestPipeline(last_success["intensity"], aug_name)
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
        # TODO: Draw graph
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

    def renderBoundries(self, html_dir="analytics"):
        if self.analytics is None:
            raise Exception("Cannot call before searchRandomizationBoundries")

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
        if not os.path.exists(html_dir):
            os.makedirs(html_dir)
        if not os.path.exists(path.join(html_dir, "imgs")):
            os.makedirs(path.join(html_dir, "imgs"))
        full_path = path.join(html_dir, "index.html")
        with open(full_path, "w") as f:
            f.write(html)
            print("Wrote " + full_path + ", open to view results")
        shutil.copy(path.join("assets", "style.css"), path.join(html_dir, "style.css"))
        shutil.copy(path.join("assets", "chart.js"), path.join(html_dir, "chart.js"))

    def is_critical_contrast_drop(self, orig_img, synthetic_img):
        orig_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        synthetic_gray = cv2.cvtColor(synthetic_img, cv2.COLOR_BGR2GRAY)
        orig_std = orig_gray.std()
        synthetic_std = synthetic_gray.std()
        contrast_ratio = synthetic_std / orig_std

        # Filter if abs contrast diff dropped below threshold
        if synthetic_std < 3 and orig_std > 3:
            return True
        # Filter if relative contrast ratio dropped below 5% of original contrast
        if contrast_ratio < 0.05:
            return True
        return False

    def synthesizeMore(self, organic_img_filenames, organic_labels, realism=0.5, count=None, min_random_augmentations=3, max_random_augmentations=8, min_predicted_diff_error=0, max_predicted_diff_error=1, output_dir="generated", preview_html="__preview.html"):
        if self.analytics is None:
            raise Exception("Cannot call before searchRandomizationBoundries")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        gen_count = count if count is not None else len(organic_img_filenames)
        gen_imgs = []
        gen_labels = []

        cur_original_index = 0
        filter_by_prediction = min_predicted_diff_error > 0 or max_predicted_diff_error < 1
        uid = len(os.listdir(output_dir))+1
        while (len(gen_imgs) < gen_count):
            origin_file = organic_img_filenames[cur_original_index]
            origin_labels = organic_labels[cur_original_index]
            label_args = {} if self.label_format is None else origin_labels

            # For each input image roll the dice up to N times
            # to find an acceptable augmented image before warning
            # and moving on to the next input image
            max_dice_rolls = 100
            synthetic_img = None
            synthetic_labels = None
            synthetic_name = None
            for i in range(max_dice_rolls):
                pipeline = self._buildPipeline(realism, min_random_augmentations, max_random_augmentations)
                base_img = cv2.imread(origin_file)
                base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
                synthetic = pipeline(image=base_img, **label_args)
                synthetic_img = synthetic["image"]

                if self.is_critical_contrast_drop(base_img, synthetic_img):
                    continue

                result_augs = [aug for aug in synthetic['replay']["transforms"]]
                applied_augs = filter(lambda aug: aug["applied"], result_augs)
                aug_str = "".join([aug['__class_fullname__'].replace("Random", "")[0:6] for aug in applied_augs])
                img_basename_no_type = ".".join(path.basename(origin_file).split(".")[0:-1])
                img_type = path.basename(origin_file).split(".")[-1]
                synthetic_name = f'''{img_basename_no_type}_{aug_str}_{str(uid)}.{img_type}'''
                uid += 1

                # Re-use original unchanged labels unless format is set
                # in which case use the calculated synthetic labels
                synthetic_labels = origin_labels
                if self.label_format is not None:
                    # Remove the image+replay and anything left over is labels
                    synthetic_labels = synthetic.copy()
                    del synthetic_labels["image"]
                    del synthetic_labels["replay"]

                if not filter_by_prediction:
                    break

                temp_filename, cleanup = self._write_temp(origin_file, synthetic["image"])
                predicted = self.my_predict(temp_filename)
                cleanup()
                err = self.diff_error(origin_labels, predicted)
                if err <= max_predicted_diff_error and err >= min_predicted_diff_error:
                    break
            
            cur_original_index = (cur_original_index+1) % len(organic_img_filenames)
            if synthetic_img is None:
                print("Warn: Could not generate valid synthetic from " + origin_file + ", skipping this round")
            else:
                synthetic_path = path.join(output_dir, synthetic_name)
                cv2.imwrite(synthetic_path, synthetic_img)
                gen_imgs.append(synthetic_path)
                gen_labels.append(synthetic_labels)

        if preview_html is not None:
            try:
                os.unlink(preview_html)
            except:
                pass
            html = ["<html><head><title>Synthetic Image Grid</title></head><body><div class=\"grid\">"]
            discovered_file_count = 0
            for file in os.listdir(output_dir):
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
        return (gen_imgs, gen_labels)

    def evaluate(self, img_filenames, img_labels):
        err_sum = 0.0
        output_differs_count = 0
        differing_outputs = {}
        for i in range(len(img_filenames)):
            img = img_filenames[i]
            truth = img_labels[i]
            predicted = self.my_predict(img)
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
