import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

class Detect(object):
    def __init__(self, image):
        self.image = image  # Assume image is loaded and stored in a numpy array

    def siftDetector(self):
        sift = cv2.SIFT_create()  # Create SIFT detector
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
        self.key_points, self.descriptors = sift.detectAndCompute(gray, None)
        return self.key_points, self.descriptors

    def locateForgery(self, eps=40, min_sample=2):
        clusters = DBSCAN(eps=eps, min_samples=min_sample).fit(self.descriptors)
        size = np.unique(clusters.labels_).shape[0] - 1
        forgery = self.image.copy()

        if (size == 0) and (np.unique(clusters.labels_)[0] == -1):
            return None, None  # No forgery detected

        if size == 0:
            size = 1

        cluster_list = [[] for _ in range(size)]
        for idx in range(len(self.key_points)):
            if clusters.labels_[idx] != -1:
                cluster_list[clusters.labels_[idx]].append(
                    (int(self.key_points[idx].pt[0]), int(self.key_points[idx].pt[1]))
                )

        forgery_parts = []
        for points in cluster_list:
            if len(points) > 1:
                forgery_parts.append(points)
                for idx1 in range(1, len(points)):
                    cv2.line(forgery, points[0], points[idx1], (255, 0, 0), 5)

        return forgery, forgery_parts

class ForgeryDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Copy-Move Forgery Detection")
        self.image = None

        # UI Elements
        self.label = Label(root, text="Select an image to detect forgery")
        self.label.pack(pady=10)

        self.choose_button = Button(root, text="Choose Image", command=self.choose_image)
        self.choose_button.pack(pady=5)

        self.detect_button = Button(root, text="Detect Forgery", command=self.detect_forgery, state=tk.DISABLED)
        self.detect_button.pack(pady=5)

        self.image_label = Label(root)
        self.image_label.pack(pady=10)

        self.result_label = Label(root, text="", fg="green")
        self.result_label.pack(pady=10)

    def choose_image(self):
        image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if image_path:
            self.image = cv2.imread(image_path)
            self.display_image(self.image)
            self.detect_button.config(state=tk.NORMAL)

    def display_image(self, image):
        # Convert BGR image to RGB for displaying
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk

    def detect_forgery(self):
        if self.image is not None:
            detect_obj = Detect(self.image)
            detect_obj.siftDetector()
            forgery_image, forgery_parts = detect_obj.locateForgery()

            if forgery_image is not None:
                self.display_image(forgery_image)
                parts_text = "\n".join([f"Forgery cluster at: {points}" for points in forgery_parts])
                self.result_label.config(text=f"Forgery Detected:\n{parts_text}", fg="red")
            else:
                self.result_label.config(text="No forgery detected!", fg="green")

if __name__ == "__main__":
    root = tk.Tk()
    app = ForgeryDetectionApp(root)
    root.mainloop()
