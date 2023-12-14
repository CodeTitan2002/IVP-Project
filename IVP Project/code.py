import cv2
import numpy as np
from sklearn.cluster import KMeans
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import tkinter as tk

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")
        self.root.configure(bg='#e6e6e6')  # Set background color

        self.image_path = None
        self.original_image = None
        self.processed_image = None

        # Buttons
        self.load_button = tk.Button(root, text="Load Image", command=self.load_image, bg='#99ccff')  # Light Blue
        self.quantize_button = tk.Button(root, text="Color Quantization", command=self.quantize_colors, bg='#ffcc99')  # Light Orange
        self.resize_button = tk.Button(root, text="Resize Image", command=self.resize_image, bg='#99ff99')  # Light Green
        self.blur_button = tk.Button(root, text="Apply Blur", command=self.apply_blur, bg='#ff6666')  # Light Red
        self.edge_detection_button = tk.Button(root, text="Edge Detection", command=self.apply_edge_detection, bg='#ffccff')  # Light Pink
        self.save_button = tk.Button(root, text="Save Processed Image", command=self.save_image, bg='#cccccc')  # Light Gray

        # Labels
        self.image_label = tk.Label(root, bg='#e6e6e6')  # Set label background color

        # Layout
        self.load_button.grid(row=0, column=0, pady=10)
        self.quantize_button.grid(row=0, column=1, pady=10)
        self.resize_button.grid(row=0, column=2, pady=10)
        self.blur_button.grid(row=0, column=3, pady=10)
        self.edge_detection_button.grid(row=0, column=4, pady=10)
        self.save_button.grid(row=0, column=5, pady=10)
        self.image_label.grid(row=1, column=0, columnspan=6)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            self.display_image(self.original_image)

    def display_image(self, image):
        # Convert the image to RGB format for displaying in Tkinter
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        # Update the image label
        self.image_label.configure(image=image)
        self.image_label.image = image

    def save_image(self):
        if self.processed_image is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                       filetypes=[("PNG files", "*.png"),
                                                                  ("JPEG files", "*.jpg;*.jpeg")])
            if save_path:
                cv2.imwrite(save_path, cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
                messagebox.showinfo("Save Successful", f"Processed image saved at {save_path}")
        else:
            messagebox.showwarning("Save Warning", "No processed image to save.")

    def quantize_colors(self):
        if self.original_image is not None:
            # Ask the user for the number of clusters (colors) for quantization
            k = simpledialog.askinteger("Input", "Enter the number of colors (clusters) for quantization:")
            if k is not None:
                pixels = self.original_image.reshape((-1, 3))
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(pixels)
                centers = np.uint8(kmeans.cluster_centers_)
                labels = kmeans.labels_
                self.processed_image = centers[labels].reshape(self.original_image.shape)
                self.display_image(self.processed_image)
        else:
            messagebox.showwarning("Image Warning", "Please load an image first.")

    def resize_image(self):
        if self.original_image is not None:
            # Ask the user for the new dimensions
            new_width = simpledialog.askinteger("Input", "Enter the new width:")
            new_height = simpledialog.askinteger("Input", "Enter the new height:")
            if new_width is not None and new_height is not None:
                self.processed_image = cv2.resize(self.original_image, (new_width, new_height))
                self.display_image(self.processed_image)
        else:
            messagebox.showwarning("Image Warning", "Please load an image first.")

    def apply_blur(self):
        if self.original_image is not None:
            # Ask the user for the blur kernel size
            kernel_size = simpledialog.askinteger("Input", "Enter the blur kernel size:")
            if kernel_size is not None:
                self.processed_image = cv2.GaussianBlur(self.original_image, (kernel_size * 2 + 1, kernel_size * 2 + 1), 0)
                self.display_image(self.processed_image)
        else:
            messagebox.showwarning("Image Warning", "Please load an image first.")

    def apply_edge_detection(self):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.processed_image = cv2.Canny(gray_image, 50, 150)
            self.display_image(self.processed_image)
        else:
            messagebox.showwarning("Image Warning", "Please load an image first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
