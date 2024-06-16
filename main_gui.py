import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import torchvision.transforms as transforms
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='PIL')

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Load the gender detection model
gender_detection_model = load_model('gender_pedestrian_model.keras')

# Define transformation for gender detection model
gender_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to get HSV limits for color detection
def get_limits():
    return {
        'yellow': ([20, 100, 100], [30, 255, 255]),
        'blue': ([100, 150, 0], [140, 255, 255]),
        'red1': ([0, 100, 100], [10, 255, 255]),
        'red2': ([160, 100, 100], [180, 255, 255]),
        'white': ([0, 0, 200], [180, 20, 255]),
        'black': ([0, 0, 0], [180, 255, 30]),
        'gray': ([0, 0, 50], [180, 50, 200]),
    }

# Function to preprocess the image for gender detection
def preprocess_image(image):
    image = cv2.resize(image, (48, 48))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  # Create a batch dimension
    return image

# Function to open file dialog and load image
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        if img is not None:
            display_image(img)
            results = model.predict(img)[0]
            display_results(img, results)
        else:
            messagebox.showerror("Error", "Failed to load image")

# Function to display image in Tkinter window
def display_image(img):
    # Resize the image to medium size for display
    resized_img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    panel.img_tk = img_tk  # Keep reference to avoid garbage collection
    panel.config(image=img_tk)

# Function to display results on the image
def display_results(img, results):
    detected_objects.clear()  # Clear the list for each new image
    detected_car_colors.clear()  # Clear the list for each new image
    detected_genders.clear()  # Clear the list for each new image

    for result in results.boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        conf = result.conf[0]
        cls = int(result.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"
        
        # Draw bounding box for detected objects
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # If a person is detected, detect gender for the entire body
        if model.names[cls] == 'person':
            body_img = img[y1:y2, x1:x2]
            if body_img.size != 0:  # Check if the cropped image is not empty
                detect_gender(body_img, img, x1, y1)
        # If a car is detected, detect car color
        elif model.names[cls] == 'car':
            car_img = img[y1:y2, x1:x2]
            if car_img.size != 0:  # Check if the cropped image is not empty
                detect_car_color(car_img, img, x1, y1)
                detected_objects.append('car')
        else:
            detected_objects.append(model.names[cls])

    display_image(img)
    update_summary()  # Call update_summary here

# Function to detect gender in the entire body image using the gender detection model
def detect_gender(body_img, original_img, x, y):
    try:
        preprocessed_img = preprocess_image(body_img)
        prediction = gender_detection_model.predict(preprocessed_img)[0]
        confidence = prediction[0]
        gender = "Male" if confidence < 0.5 else "Female"
        
        detected_genders.append((gender, confidence))
        
        # Draw gender label with confidence on the original image
        confidence_percentage = (1 - confidence if gender == "Male" else confidence) * 100
        label = f"{gender} ({confidence_percentage:.2f}%)"
        cv2.putText(original_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    except Exception as e:
        print(f"Gender detection failed: {str(e)}")

# Function to detect car color using HSV color detection within the bounding box of the car
def detect_car_color(car_img, original_img, x, y):
    try:
        hsv_car_img = cv2.cvtColor(car_img, cv2.COLOR_BGR2HSV)
        detected_color = None
        max_color_area = 0
        
        color_ranges = get_limits()
        
        for color_name, (lower_limit, upper_limit) in color_ranges.items():
            lower_limit = np.array(lower_limit, dtype=np.uint8)
            upper_limit = np.array(upper_limit, dtype=np.uint8)
            
            if color_name == 'red1':
                mask1 = cv2.inRange(hsv_car_img, lower_limit, upper_limit)
            elif color_name == 'red2':
                mask2 = cv2.inRange(hsv_car_img, lower_limit, upper_limit)
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv_car_img, lower_limit, upper_limit)
            
            color_area = cv2.countNonZero(mask)
            
            if color_area > max_color_area:
                max_color_area = color_area
                detected_color = color_name.replace('1', '').replace('2', '')
        
        if detected_color:
            detected_car_colors.append((detected_color, x, y))
            
            # Determine custom color
            custom_color = None
            if detected_color == 'blue':
                custom_color = 'red'
            elif detected_color == 'red':
                custom_color = 'blue'
            else:
                custom_color = 'None'
            
            # Draw car color label on the original image
            cv2.putText(original_img, f"{detected_color} car", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.putText(original_img, f"Custom: {custom_color}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Print detected and custom car color in the terminal
            print(f"Detected car color: {detected_color}, Custom car color: {custom_color}")

    except Exception as e:
        print(f"Car color detection failed: {str(e)}")

def update_summary():
    # Update gender counts
    num_male = sum(1 for gender, _ in detected_genders if gender == "Male")
    num_female = sum(1 for gender, _ in detected_genders if gender == "Female")
    
    # Update car count
    num_cars = sum(1 for obj in detected_objects if obj == "car")
    
    num_other_vehicles = sum(1 for obj in detected_objects if obj not in ["car", "traffic light", "handbag"])

    
    # Update car colors detected
    car_colors_detected = set(color for color, _, _ in detected_car_colors)

    # Update the summary labels
    gender_summary = f"Genders detected: Male ({num_male}), Female ({num_female})"
    gender_summary_label.config(text=gender_summary)

    car_summary = f"Cars detected: {num_cars}"
    other_vehicle_summary = f"Other vehicles detected: {num_other_vehicles}"
    car_color_summary = f"Car colors detected: {', '.join(car_colors_detected)}"
    
    summary_text = '\n'.join([car_summary, other_vehicle_summary, car_color_summary])
    summary_label.config(text=summary_text)

        # Update car color details
    car_color_details = "\n".join([f"Detected car color: {color}, Custom car color: {'red' if color == 'blue' else 'blue' if color == 'red' else 'None'}" for color, _, _ in detected_car_colors])
    car_color_details_label.config(text=car_color_details)

    # Print summaries in the terminal
    print(gender_summary)
    print(car_summary)
    print(other_vehicle_summary)
    print(car_color_summary)
    

# Create the main window
window = tk.Tk()
window.title("YOLOv8 and Gender & Car Color Detection")

# Create panel to display image
panel = tk.Label(window)
panel.pack(padx=10, pady=10)

# Create label to display gender summary
gender_summary_label = tk.Label(window, text="")
gender_summary_label.pack(padx=10, pady=10)

# Create label to display car color details
car_color_details_label = tk.Label(window, text="")
car_color_details_label.pack(padx=10, pady=10)

# Create label to display the summary
summary_label = tk.Label(window, text="")
summary_label.pack(padx=10, pady=10)

# Create button to upload image
btn = tk.Button(window, text="Upload Image", command=load_image)
btn.pack(side="bottom", fill="both", padx=10, pady=10)

# Initialize the lists to store detected genders, objects, and car colors
detected_genders = []
detected_objects = []
detected_car_colors = []

# Start Tkinter event loop
window.mainloop()

