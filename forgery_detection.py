import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import tkinter as tk
from tkinter import filedialog

class Detect(object):
    def __init__(self, image):
        self.image = image  #assume image is loaded and stired in numpy array

    def siftDetector(self):
        sift = cv2.SIFT_create()  # Corrected SIFT creation method detect key point based on local intensity and compute their descriptors
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) #convert image from BGR to grayscale
        self.key_points, self.descriptors = sift.detectAndCompute(gray, None)
        return self.key_points, self.descriptors

    def showSiftFeatures(self):
        # gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sift_image = cv2.drawKeypoints(self.image, self.key_points, self.image.copy())
        return sift_image

    def locateForgery(self, eps=40, min_sample=2):
        clusters = DBSCAN(eps=eps, min_samples=min_sample).fit(self.descriptors)
        size = np.unique(clusters.labels_).shape[0] - 1
        forgery = self.image.copy()
        
        if (size == 0) and (np.unique(clusters.labels_)[0] == -1):
            print('No Forgery Found!!')   ##if image is not forged
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
#my name is ashika

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
    # Get screen dimensions
    screen_width = 800  # Adjust these values according to your screen
    screen_height = 600

    # Resize image if it exceeds screen dimensions
    h, w, _ = image.shape
    scale_factor = min(screen_width / w, screen_height / h, 1)  # Ensures the image fits within the screen
    if scale_factor < 1:
        new_width = int(w * scale_factor)
        new_height = int(h * scale_factor)
        image = cv2.resize(image, (new_width, new_height))

    detect_obj = Detect(image)
    detect_obj.siftDetector()
    forgery_image = detect_obj.locateForgery()

    if forgery_image is not None:
        # Resize the forgery image similarly to fit the screen
        h, w, _ = forgery_image.shape
        scale_factor = min(screen_width / w, screen_height / h, 1)
        if scale_factor < 1:
            new_width = int(w * scale_factor)
            new_height = int(h * scale_factor)
            forgery_image = cv2.resize(forgery_image, (new_width, new_height))
        
        cv2.imshow("Forgery Detection", forgery_image)  # Display the image in a window
        cv2.waitKey(0)  # Wait for any key press
        cv2.destroyAllWindows()  # Close all OpenCV windows
    else:
        print("No forgery detected!")


if __name__ == "__main__":
    # Choose an image via file dialog
    image = choose_image()

    if image is not None:
        check_forgery(image)
