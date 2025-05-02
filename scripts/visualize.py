import idx2numpy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import subprocess

train_images_path = 'C:/CLionProjects/nn/data/train-images.idx3-ubyte'
train_labels_path = 'C:/CLionProjects/nn/data/train-labels.idx1-ubyte'
test_images_path = 'C:/CLionProjects/nn/data/t10k-images.idx3-ubyte'
test_labels_path = 'C:/CLionProjects/nn/data/t10k-labels.idx1-ubyte'


def load_data_set():
    train_img = idx2numpy.convert_from_file(train_images_path)
    train_labels = idx2numpy.convert_from_file(train_labels_path)
    test_img = idx2numpy.convert_from_file(test_images_path)
    test_labels = idx2numpy.convert_from_file(test_labels_path)

    print(f'Train images shape: {train_img.shape}')
    print(f'Train labels shape: {train_labels.shape}')
    print(f'Test images shape: {test_img.shape}')
    print(f'Test labels shape: {test_labels.shape}')
    
    return train_img, train_labels, test_img, test_labels


def to_img(image_array):
    img = Image.fromarray(image_array)
    return img


def send_to_cpp(grid_data):
    flattened_data = grid_data.flatten().tolist()
    
    data_str = ' '.join(map(str, flattened_data))
    
    try:
        cpp_process = subprocess.Popen(
            [r"C:\CLionProjects\nn\neural_network\cmake-build-debug\predictor.exe"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = cpp_process.communicate(input=data_str)
        
        if stderr:
            print(f"Error from C++ program: {stderr}")
            return "Error"
        
        return stdout.strip()
    
    except Exception as e:
        print(f"Error communicating with C++ program: {e}")
        return "Error"


def show_image(image_array, label, prediction):
    img = to_img(image_array)
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {label}, Prediction: {prediction}")
    plt.show()


def navigate_images(image_data, label_data, start_idx=0):
    current_idx = start_idx

    while True:
        image = image_data[current_idx]
        label = label_data[current_idx]
        
        prediction = send_to_cpp(image)
        
        show_image(image, label, prediction)
        
        user_input = input("Press 'n' for next, 'p' for previous, 'q' to quit: ").strip().lower()
        
        if user_input == 'n': 
            if current_idx < len(image_data) - 1:
                current_idx += 1
        elif user_input == 'p':  
            if current_idx > 0:
                current_idx -= 1
        else:
            print("Invalid input. Please try again.")

# Załadowanie danych
train_img, train_labels, test_img, test_labels = load_data_set()

# Rozpocznij nawigację od pierwszego obrazu w zbiorze testowym
navigate_images(test_img, test_labels, start_idx=0)
