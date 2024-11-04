import tkinter as tk
from tkinter import filedialog, messagebox
# from tensorflow.keras.models import load_model
import joblib as load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('cervical_proj12_model.h5')

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = image.load_img(file_path, target_size=(174,122))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return img, file_path
    else:
        return None, None

def predict_image(name, age):
    img, file_path = load_image()
    if img is None or file_path is None:
        return
    
    # Scale the pixel values to [0, 1]
    input_image_scaled = img / 255

    # Reshape the input image to match the model's input shape
    input_image_reshaped = np.reshape(input_image_scaled, [1, 174, 122, 3])

    # Make predictions
    input_prediction = model.predict(input_image_reshaped)

    # Assuming it's a binary classification task
    threshold = 0.8

    # Extract the predicted probability
    predicted_probability = input_prediction[0][1]

    if predicted_probability >= threshold:
        result_label.config(text=f'Patient: {name}\nAge: {age} \nCancer cell detected', fg='red')
    else:
        result_label.config(text=f'Patient: {name}\nAge: {age} \nCancer cells not detected.', fg='green')

def main():
    root = tk.Tk()
    root.title("Cervical Cancer Prediction")
    root.geometry("400x400")

    # Set black background
    background_color = "black"
    root.configure(background=background_color)

    name_label = tk.Label(root, text="Enter Name:", bg=background_color, fg="white")
    name_label.pack(pady=5)
    name_entry = tk.Entry(root)
    name_entry.pack(pady=5)

    age_label = tk.Label(root, text="Enter Age:", bg=background_color, fg="white")
    age_label.pack(pady=5)
    age_entry = tk.Entry(root)
    age_entry.pack(pady=5)

    load_button = tk.Button(root, text="Load Image", command=lambda: predict_image(name_entry.get(), age_entry.get()))
    load_button.pack(pady=10)

    global result_label
    result_label = tk.Label(root, text='', font=('Helvetica', 12), bg=background_color, fg="white")
    result_label.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
