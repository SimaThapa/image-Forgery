import cv2
import numpy as np
from scipy.fftpack import dct
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import time  # To create unique output filenames using timestamps

# Load the image
def load_image(image_path, resize_factor=0.8):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    # Resize the image to speed up processing
    h, w = img.shape
    img_resized = cv2.resize(img, (int(w * resize_factor), int(h * resize_factor)))
    return img_resized

# Divide the image into overlapping blocks with stride to reduce block overlap
def divide_into_blocks(img, block_size, stride=2):
    h, w = img.shape
    blocks = []
    positions = []
    for i in range(0, h - block_size + 1, stride):
        for j in range(0, w - block_size + 1, stride):
            block = img[i:i + block_size, j:j + block_size]
            blocks.append(block)
            positions.append((i, j))
    return blocks, positions

# Apply DCT on a block
def compute_dct(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

# Extract features from blocks, using more DCT coefficients
def extract_features(blocks, max_blocks=1000):
    features = []
    for idx, block in enumerate(blocks):
        if idx >= max_blocks:  # Limit the number of blocks to process
            break
        dct_block = compute_dct(block)
        features.append(dct_block.flatten()[:30])  # Use first 30 coefficients (more information)
    features = np.array(features)
    return features

# Detect duplicate blocks and classify them as original and forged
def detect_copy_move_forgery(features, positions, threshold=0.2):
    matches = []
    dist_matrix = cdist(features, features, 'euclidean')
    np.fill_diagonal(dist_matrix, np.inf)  # Ignore self-comparisons

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if dist_matrix[i][j] < threshold:  # Similarity threshold
                if abs(positions[i][0] - positions[j][0]) > 1 or abs(positions[i][1] - positions[j][1]) > 1:
                    matches.append((positions[i], positions[j]))
    return matches


# Visualize detected forgery regions with clear labeling
def visualize_forgery(img, matches, block_size, output_path):
    img_copy = img.copy()  # Keep the original color image

    # Highlight forged regions with yellow circles and label them
    for pos1, pos2 in matches:
        center1 = (pos1[1] + block_size // 2, pos1[0] + block_size // 2)
        center2 = (pos2[1] + block_size // 2, pos2[0] + block_size // 2)
        radius = block_size

        # Draw yellow circles around original and copied parts
        cv2.circle(img_copy, center1, radius, (0, 255, 255), 2)  # Yellow circle for original
        cv2.circle(img_copy, center2, radius, (0, 255, 255), 2)  # Yellow circle for copied

        # Add labels
        cv2.putText(img_copy, 'Original', (center1[0] - 20, center1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(img_copy, 'Copied', (center2[0] - 20, center2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Labeling the image
    cv2.putText(img_copy, 'Copy-Move Forgery Detection', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Convert from BGR to RGB before displaying with matplotlib
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.title("Detected Forgery")
    plt.axis("off")
    plt.savefig(output_path)  # Save the output image
    plt.show()

# Main function to process the image and detect forgery
def process_image(image_path, block_size=8, output_folder="output"):
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        img = load_image(image_path, resize_factor=0.8)
        blocks, positions = divide_into_blocks(img, block_size, stride=2)
        features = extract_features(blocks, max_blocks=1000)
        matches = detect_copy_move_forgery(features, positions, threshold=0.2)

        if matches:
            messagebox.showinfo("Forgery Detected", f"Copy-move forgery detected! Found {len(matches)} matching regions.")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_folder, f"detected_forgery_{timestamp}.jpg")
            visualize_forgery(img, matches, block_size, output_path)
        else:
            messagebox.showinfo("No Forgery", "No copy-move forgery detected.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create a Tkinter GUI for user input and display
def create_gui():
    root = tk.Tk()
    root.title("Image Forgery Detection")

    label = tk.Label(root, text="Choose an image for forgery detection:")
    label.pack(pady=10)

    image_path_entry = tk.Entry(root, width=40)
    image_path_entry.pack(pady=10)

    def browse_image():
        filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        image_path_entry.delete(0, tk.END)
        image_path_entry.insert(0, filename)

    browse_button = tk.Button(root, text="Browse", command=browse_image)
    browse_button.pack(pady=10)

    def detect_forgery():
        image_path = image_path_entry.get()
        if image_path:
            process_image(image_path, block_size=8, output_folder="output")
        else:
            messagebox.showwarning("Input Error", "Please select an image first.")

    detect_button = tk.Button(root, text="Detect Forgery", command=detect_forgery)
    detect_button.pack(pady=20)

    root.mainloop()

# Run the program
if __name__ == "__main__":
    create_gui()