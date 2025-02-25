import cv2
import numpy as np
from os import path

# A model that, given an input image, finds the closest matching image from a training_img_filenames set and outputs
# 4 COCO segmentation point labels representing the switch bounding box and class of on or off.
class SiftSimilarity():

    def __init__(self, training_img_filenames: list[str], training_coco_segment_labels: list):
        # "Train" the BFMatcher by finding features of all input models
        imgs = [cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB) for x in training_img_filenames]
        self.training_labels = [{"confidence": 1, "out": i} for i in range(len(training_img_filenames))]
        # Initiate SIFT detector
        self.sift = cv2.SIFT_create()
        # "Train" by converting all images to SIFT features for fast similarity comparison
        self.trained_features = [self.sift.detectAndCompute(img, None) for img in imgs]
        self.training_labels = training_coco_segment_labels

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

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
        best_segment = None
        best_similarity = 0
        img_features = self.sift.detectAndCompute(img, None)
        for i in range(len(self.trained_features)):
            baseline = self.trained_features[i]
            allMatches,kps1,desc1,kps2,desc2 = self.sift_matches(baseline, img_features)
            if len(allMatches) == 0:
                continue

            
            matches = self.flann.knnMatch(desc1,desc2,k=2)
        
            # store all the good matches as per Lowe's ratio test.
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)

            similarity = len(good) / len(matches)

            #match=self.get_inliers_ratio(desc1,desc2,ratio=0.7)
            #similarity=self.get_similarity(match,kps1,kps2)

            if similarity > best_similarity or best_segment is None:
                best_similarity = similarity
                best_segment = self.training_labels[i].copy()
                # Also transform segment points to match homography,
                # aka realign the segment to where the switch is
                homog, status = self.find_homography(None, kps2, kps1, None)
                
        return {"out": best_segment, "confidence": best_similarity}

    def get_similarity(self, inliers, kps1, kps2):
        similarity = len(inliers) / min(len(kps1), len(kps2))
        return similarity

    def find_homography(self, inliers, kpsA, kpsB, reprojThresh):
        s = np.float32([kpsA[i.queryIdx].pt for i in inliers]).reshape(-1, 1, 2)
        d = np.float32([kpsB[i.trainIdx].pt for i in inliers]).reshape(-1, 1, 2)
        (H, status) = cv2.findHomography(s, d, cv2.RANSAC, reprojThresh)
        return (H, status)