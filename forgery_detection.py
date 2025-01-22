import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import os

class Detect(object):
    """
    A class for detecting copy-move forgery using SIFT and DBSCAN.
    
    Attributes:
        image: The input image as a numpy array.
    Methods:
        siftDetector(): Detects keypoints and descriptors using SIFT.
        locateForgery(eps, min_sample): Identifies clusters of keypoints indicating forgery.
    """
    def __init__(self, image):
        self.image = image  # Assume image is loaded and stored in a numpy array

    def siftDetector(self):
        """Detect keypoints and descriptors using SIFT."""
        sift = cv2.SIFT_create()  # Create SIFT detector
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
        self.key_points, self.descriptors = sift.detectAndCompute(gray, None)
        return self.key_points, self.descriptors

    def locateForgery(self, eps=40, min_sample=2):
        """Locate forgery using DBSCAN clustering on SIFT descriptors."""
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
    """
    A GUI application for copy-move forgery detection using SIFT and DBSCAN.
    
    Methods:
        choose_image(): Opens a file dialog to select an image.
        display_image(image): Displays the given image on the GUI.
        detect_forgery(): Performs forgery detection and displays results.
    """
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

        # Save Results Button
        self.save_button = Button(root, text="Save Results", command=self.save_results, state=tk.DISABLED)
        self.save_button.pack(pady=5)

        self.forgery_parts = None

    def choose_image(self):
        """Opens a file dialog to select an image."""
        image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if image_path:
            self.image = cv2.imread(image_path)
            self.image = self.resize_image(self.image)  # Resize image if it exceeds screen dimensions
            self.display_image(self.image)
            self.detect_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.DISABLED)

    def resize_image(self, image, max_width=800, max_height=600):
        """Resize the image to fit within the screen dimensions."""
        h, w, _ = image.shape
        scale_factor = min(max_width / w, max_height / h, 1)
        if scale_factor < 1:
            new_width = int(w * scale_factor)
            new_height = int(h * scale_factor)
            image = cv2.resize(image, (new_width, new_height))
        return image

    def display_image(self, image):
        """Displays the given image on the GUI."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk

    def detect_forgery(self):
        """Performs forgery detection and displays results."""
        if self.image is not None:
            detect_obj = Detect(self.image)
            detect_obj.siftDetector()
            forgery_image, self.forgery_parts = detect_obj.locateForgery()

            if forgery_image is not None:
                forgery_image = self.resize_image(forgery_image)  # Resize forgery image if needed
                self.display_image(forgery_image)
                parts_text = "\n".join([f"Forgery cluster at: {points}" for points in self.forgery_parts])
                self.result_label.config(text=f"Forgery Detected:\n{parts_text}", fg="red")
                self.save_button.config(state=tk.NORMAL)
            else:
                self.result_label.config(text="No forgery detected!", fg="green")

    def save_results(self):
        """Saves the results (image and textual description) to the disk."""
        if self.image is not None and self.forgery_parts is not None:
            save_dir = filedialog.askdirectory(title="Select Directory to Save Results")
            if save_dir:
                # Save the forgery image
                forgery_image_path = os.path.join(save_dir, "forgery_detected.png")
                cv2.imwrite(forgery_image_path, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))

                # Save the forgery details
                forgery_details_path = os.path.join(save_dir, "forgery_details.txt")
                with open(forgery_details_path, "w") as f:
                    for idx, points in enumerate(self.forgery_parts):
                        f.write(f"Cluster {idx + 1}: {points}\n")

                self.result_label.config(text="Results saved successfully!", fg="blue")

if __name__ == "__main__":
    root = tk.Tk()
    app = ForgeryDetectionApp(root)
    root.mainloop()
