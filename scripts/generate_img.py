import idx2numpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import subprocess
import random
import os
from matplotlib.colors import LinearSegmentedColormap
from config import MNIST_TEST_IMAGES, MNIST_TEST_LABELS, PREDICTOR_EXE, FASHION_MNIST_LABELS

test_images_path = MNIST_TEST_IMAGES
test_labels_path = MNIST_TEST_LABELS
fashion_mnist_labels = FASHION_MNIST_LABELS


def load_test_data():
    """Load the MNIST test dataset"""
    test_img = idx2numpy.convert_from_file(test_images_path)
    test_labels = idx2numpy.convert_from_file(test_labels_path)
    print(f'Test images shape: {test_img.shape}')
    print(f'Test labels shape: {test_labels.shape}')
    return test_img, test_labels

def send_to_cpp(image_data):
    """Send image data to the C++ model and get prediction"""
    flattened_data = image_data.flatten().tolist()
    data_str = ' '.join(map(str, flattened_data))
    
    try:
        cpp_process = subprocess.Popen(
            [PREDICTOR_EXE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = cpp_process.communicate(input=data_str)
        
        if stderr:
            print(f"Error from C++ program: {stderr}")
            return "Error"
        
        return int(stdout.strip())
    
    except Exception as e:
        return "Error"

def create_prediction_grid(num_rows=10, num_cols=15, random_seed=42):
    """Create a grid visualization of MNIST test images with model predictions"""
    test_images, test_labels = load_test_data()
    
    random.seed(random_seed)
    
    num_samples = num_rows * num_cols
    selected_indices = random.sample(range(len(test_images)), num_samples)
    
    predictions = []
    actual_labels = []
    
    cell_size = 32  
    padding = 4    
    
    header_height = 50  
    border = 20     
    
    grid_width = num_cols * (cell_size + padding) - padding + 2 * border
    grid_height = num_rows * (cell_size + padding) - padding + 2 * border + header_height
    
    grid_image = Image.new('RGB', (grid_width, grid_height), color=(245, 245, 250))
    draw = ImageDraw.Draw(grid_image)
    
    try:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    except IOError:
        print("Using default font")
        font = None
        small_font = None
    
    for i, idx in enumerate(selected_indices):
        row = i // num_cols
        col = i % num_cols
        
        image = test_images[idx]
        label = test_labels[idx]
        actual_labels.append(label)
        
        prediction = send_to_cpp(image)
        predictions.append(prediction)
        
        if image.max() <= 1.0:
            display_image = (image * 255).astype(np.uint8)
        else:
            display_image = image.astype(np.uint8)
            
        enhanced_image = Image.fromarray(display_image).convert('L')
        
        rgb_image = enhanced_image.convert('RGB')
        
        x = col * (cell_size + padding) + border
        y = row * (cell_size + padding) + border + header_height 
        
        rgb_image = rgb_image.resize((cell_size, cell_size), Image.LANCZOS)
        
        draw.rectangle([x-1, y-1, x+cell_size, y+cell_size], outline=(200, 200, 210))
        
        grid_image.paste(rgb_image, (x, y))

        if prediction != label:

             
            red_overlay = Image.new('RGBA', rgb_image.size, color=(255, 100, 100, 150))
            rgb_image = Image.alpha_composite(rgb_image.convert('RGBA'), red_overlay).convert('RGB')

            grid_image.paste(rgb_image, (x, y))


            true_label_text = str(prediction)
            text_position = (x + cell_size - 7, y)
            draw.text(text_position, true_label_text, fill=(255, 255, 255), font=small_font)


        
    
    correct = sum(1 for p, l in zip(predictions, actual_labels) if p == l)
    accuracy = correct / len(predictions) * 100
    
    title = "MNIST Predictions Visualization"
    subtitle = f"Accuracy: {accuracy:.1f}%"
    
    draw.text((border, border), title, fill=(50, 50, 100), font=font)
    draw.text((border, border + 20), subtitle, fill=(80, 80, 120), font=small_font)
    
    draw.line([(border, header_height + border - 10), 
               (grid_width - border, header_height + border - 10)], 
              fill=(200, 200, 210), width=1)
    
    output_path = "mnist_predictions_grid.png"
    grid_image.save(output_path)
    
    print(f"Grid created with {num_rows}x{num_cols} images")
    print(f"Model accuracy on selected samples: {accuracy:.2f}%")
    print(f"Grid saved to: {output_path}")
    
    # Display the grid
    plt.figure(figsize=(15, 10))
    plt.imshow(np.array(grid_image))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return grid_image

if __name__ == "__main__":
    grid_image = create_prediction_grid(num_rows=15, num_cols=15, random_seed=9)


# import idx2numpy
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
# import matplotlib.pyplot as plt
# import subprocess
# import random
# import os
# from matplotlib.colors import LinearSegmentedColormap

# test_images_path = 'C:/CLionProjects/nn/data/fashion_mnist/fashion-test-img'
# test_labels_path = 'C:/CLionProjects/nn/data/fashion_mnist/fashion-test-labels'
# fashion_mnist_labels = [
#     "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
#     "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
# ]


# def load_test_data():
#     """Load the MNIST test dataset"""
#     test_img = idx2numpy.convert_from_file(test_images_path)
#     test_labels = idx2numpy.convert_from_file(test_labels_path)
#     print(f'Test images shape: {test_img.shape}')
#     print(f'Test labels shape: {test_labels.shape}')
#     return test_img, test_labels

# def send_to_cpp(image_data):
#     """Send image data to the C++ model and get prediction"""
#     flattened_data = image_data.flatten().tolist()
#     data_str = ' '.join(map(str, flattened_data))
    
#     try:
#         cpp_process = subprocess.Popen(
#             [r"C:\CLionProjects\nn\neural_network\cmake-build-debug\predictor.exe"],
#             stdin=subprocess.PIPE,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True
#         )
        
#         stdout, stderr = cpp_process.communicate(input=data_str)
        
#         if stderr:
#             print(f"Error from C++ program: {stderr}")
#             return "Error"
        
#         return int(stdout.strip())
    
#     except Exception as e:
#         print(f"Error communicating with C++ program: {e}")
#         return "Error"

# def create_prediction_grid(num_rows=10, num_cols=15, random_seed=42):
#     """Create a grid visualization of MNIST test images with model predictions"""
#     test_images, test_labels = load_test_data()
    
#     random.seed(random_seed)
    
#     num_samples = num_rows * num_cols
#     selected_indices = random.sample(range(len(test_images)), num_samples)
    
#     predictions = []
#     actual_labels = []
    
#     cell_size = 32  
#     padding = 4     
    
#     header_height = 50  
#     border = 20    
    
#     label_height = 15
    
#     grid_width = num_cols * (cell_size + padding) - padding + 2 * border
#     grid_height = num_rows * (cell_size + padding + label_height) - padding + 2 * border + header_height
    
#     grid_image = Image.new('RGB', (grid_width, grid_height), color=(245, 245, 250))
#     draw = ImageDraw.Draw(grid_image)
    
#     try:
#         font = ImageFont.load_default()
#         small_font = ImageFont.load_default()
#     except IOError:
#         print("Using default font")
#         font = None
#         small_font = None
    
#     for i, idx in enumerate(selected_indices):
#         row = i // num_cols
#         col = i % num_cols
        
#         image = test_images[idx]
#         label = test_labels[idx]
#         actual_labels.append(label)
        
#         prediction = send_to_cpp(image)
#         predictions.append(prediction)
    

#         if image.max() <= 1.0:
#             display_image = (image * 255).astype(np.uint8)
#         else:
#             display_image = image.astype(np.uint8)
            
#         enhanced_image = Image.fromarray(display_image).convert('L')
        
#         rgb_image = enhanced_image.convert('RGB')
        
#         x = col * (cell_size + padding + label_height) + border
#         y = row * (cell_size + padding + label_height) + border + header_height 
        
#         rgb_image = rgb_image.resize((cell_size, cell_size), Image.LANCZOS)
        
#         draw.rectangle([x-1, y-1, x+cell_size, y+cell_size], outline=(200, 200, 210))
        
#         grid_image.paste(rgb_image, (x, y))

#         if prediction != label:
#             draw.rectangle([x-2, y-2, x+cell_size+1, y+cell_size+1], outline=(255, 0, 0), width=2)
            
#             pred_label_text = fashion_mnist_labels[prediction]
#             draw.text((x, y + cell_size + 2), pred_label_text, fill=(255, 0, 0), font=small_font)
    
#     correct = sum(1 for p, l in zip(predictions, actual_labels) if p == l)
#     accuracy = correct / len(predictions) * 100
    
#     title = "Fashion MNIST Predictions Visualization"
#     subtitle = f"Accuracy: {accuracy:.1f}%"
    
#     draw.text((border, border), title, fill=(50, 50, 100), font=font)
#     draw.text((border, border + 20), subtitle, fill=(80, 80, 120), font=small_font)
    
#     draw.line([(border, header_height + border - 10), 
#                (grid_width - border, header_height + border - 10)], 
#               fill=(200, 200, 210), width=1)
    
#     output_path = "fashion_mnist_predictions_grid.png"
#     grid_image.save(output_path)
    
#     print(f"Grid created with {num_rows}x{num_cols} images")
#     print(f"Model accuracy on selected samples: {accuracy:.2f}%")
#     print(f"Grid saved to: {output_path}")
    
#     plt.figure(figsize=(15, 10))
#     plt.imshow(np.array(grid_image))
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()
    
#     return grid_image

# if __name__ == "__main__":
#     grid_image = create_prediction_grid(num_rows=15, num_cols=15, random_seed=2)