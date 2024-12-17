import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import tkinter as tk
from tkinter import filedialog

class Detect(object):
    def __init__(self, image):
        self.image = image

    def siftDetector(self):
        sift = cv2.SIFT_create()  # Corrected SIFT creation method
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.key_points, self.descriptors = sift.detectAndCompute(gray, None)
        return self.key_points, self.descriptors

    def showSiftFeatures(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sift_image = cv2.drawKeypoints(self.image, self.key_points, self.image.copy())
        return sift_image

    def locateForgery(self, eps=40, min_sample=2):
        clusters = DBSCAN(eps=eps, min_samples=min_sample).fit(self.descriptors)
        size = np.unique(clusters.labels_).shape[0] - 1
        forgery = self.image.copy()
        
        if (size == 0) and (np.unique(clusters.labels_)[0] == -1):
            print('No Forgery Found!!')
            return None
        
        if size == 0:
            size = 1
            
        cluster_list = [[] for i in range(size)]
        for idx in range(len(self.key_points)):
            if clusters.labels_[idx] != -1:
                cluster_list[clusters.labels_[idx]].append((int(self.key_points[idx].pt[0]), int(self.key_points[idx].pt[1])))

        for points in cluster_list:
            if len(points) > 1:
                for idx1 in range(1, len(points)):
                    cv2.line(forgery, points[0], points[idx1], (255, 0, 0), 5)
        
        return forgery

def choose_image():
    # Create a Tkinter root window and hide it
    root = tk.Tk()
    root.withdraw()

    # Open file dialog to choose an image
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    # Check if an image was selected
    if image_path:
        image = cv2.imread(image_path)  # Load the selected image
        return image
    else:
        print("No image selected. Exiting.")
        return None

def check_forgery(image):
    detect_obj = Detect(image)
    detect_obj.siftDetector()
    forgery_image = detect_obj.locateForgery()

    if forgery_image is not None:
        cv2.imshow("Forgery Detection", forgery_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No forgery detected!")

if __name__ == "__main__":
    # Choose an image via file dialog
    image = choose_image()

    if image is not None:
        check_forgery(image)
