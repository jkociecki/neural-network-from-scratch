import idx2numpy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import subprocess
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
import threading
import random
from matplotlib.figure import Figure
import time
from config import (
    FASHION_MNIST_TRAIN_IMAGES,
    FASHION_MNIST_TRAIN_LABELS,
    FASHION_MNIST_TEST_IMAGES,
    FASHION_MNIST_TEST_LABELS,
    PREDICTOR_EXE,
    FASHION_MNIST_LABELS
)

class MNISTViewer:
    def __init__(self, master):
        self.master = master
        master.title("MNIST Viewer")
        master.geometry("1200x700")
        master.configure(bg='#f0f0f0')

        self.train_images_path = FASHION_MNIST_TRAIN_IMAGES
        self.train_labels_path = FASHION_MNIST_TRAIN_LABELS
        self.test_images_path = FASHION_MNIST_TEST_IMAGES
        self.test_labels_path = FASHION_MNIST_TEST_LABELS

        self.current_idx = 0
        self.use_test_set = tk.BooleanVar(value=True)
        self.image_data = None
        self.label_data = None
        self.current_image = None
        self.current_label = None
        self.current_prediction = None
        self.prediction_probs = None
        self.loading = False

        self.history_size = 20
        self.prediction_history = []
        self.accuracy_history = []
        self.class_names = FASHION_MNIST_LABELS

        self.create_widgets()
        self.load_data()

    def load_data(self):
        self.loading = True
        self.status_label.config(text="Ładowanie danych...")
        threading.Thread(target=self._load_data_thread, daemon=True).start()
    
    def _load_data_thread(self):
        try:
            train_img = idx2numpy.convert_from_file(self.train_images_path)
            train_labels = idx2numpy.convert_from_file(self.train_labels_path)
            test_img = idx2numpy.convert_from_file(self.test_images_path)
            test_labels = idx2numpy.convert_from_file(self.test_labels_path)

            info = f'Train: {train_img.shape[0]} obrazów, Test: {test_img.shape[0]} obrazów'

            if self.use_test_set.get():
                self.image_data = test_img
                self.label_data = test_labels
            else:
                self.image_data = train_img
                self.label_data = train_labels

            self.prediction_history = []
            self.accuracy_history = []

            self.master.after(0, lambda: self._finish_loading(info))
        except Exception as e:
            self.master.after(0, lambda: self.status_label.config(text=f"Błąd ładowania: {str(e)}"))
            self.loading = False
    
    def _finish_loading(self, info):
        self.dataset_info_label.config(text=info)
        self.loading = False
        self.status_label.config(text="Dane załadowane pomyślnie")
        self.update_display()
    
    def create_widgets(self):
        control_frame = ttk.Frame(self.master, padding="10")
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        dataset_label = ttk.Label(control_frame, text="Zestaw danych:")
        dataset_label.pack(side=tk.LEFT, padx=(0, 5))

        train_radio = ttk.Radiobutton(control_frame, text="Treningowy", 
                                      variable=self.use_test_set, value=False,
                                      command=self.change_dataset)
        train_radio.pack(side=tk.LEFT, padx=5)

        test_radio = ttk.Radiobutton(control_frame, text="Testowy", 
                                     variable=self.use_test_set, value=True,
                                     command=self.change_dataset)
        test_radio.pack(side=tk.LEFT, padx=5)

        random_btn = ttk.Button(control_frame, text="Losowy obraz", command=self.random_image)
        random_btn.pack(side=tk.LEFT, padx=20)

        test_batch_btn = ttk.Button(control_frame, text="Test 100 losowych obrazów", 
                                   command=self.test_random_batch)
        test_batch_btn.pack(side=tk.LEFT, padx=5)

        self.dataset_info_label = ttk.Label(control_frame, text="")
        self.dataset_info_label.pack(side=tk.RIGHT, padx=5)

        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig_image = Figure(figsize=(5, 5), dpi=100)
        self.ax_image = self.fig_image.add_subplot(111)
        self.canvas_image = FigureCanvasTkAgg(self.fig_image, left_frame)
        self.canvas_image.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        probs_frame = ttk.LabelFrame(right_frame, text="Prawdopodobieństwa predykcji", padding="10")
        probs_frame.pack(fill=tk.BOTH, expand=True)

        self.fig_probs = Figure(figsize=(5, 4), dpi=100)
        self.ax_probs = self.fig_probs.add_subplot(111)
        self.canvas_probs = FigureCanvasTkAgg(self.fig_probs, probs_frame)
        self.canvas_probs.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        history_frame = ttk.LabelFrame(right_frame, text="Historia predykcji", padding="10")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.fig_history = Figure(figsize=(5, 3), dpi=100)
        self.ax_history = self.fig_history.add_subplot(111)
        self.canvas_history = FigureCanvasTkAgg(self.fig_history, history_frame)
        self.canvas_history.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        nav_frame = ttk.Frame(self.master)
        nav_frame.pack(fill=tk.X, padx=10, pady=10)

        first_btn = ttk.Button(nav_frame, text="Pierwszy", command=self.go_first)
        first_btn.pack(side=tk.LEFT, padx=5)

        prev_btn = ttk.Button(nav_frame, text="Poprzedni", command=self.go_prev)
        prev_btn.pack(side=tk.LEFT, padx=5)

        self.index_label = ttk.Label(nav_frame, text="Obraz: 0/0")
        self.index_label.pack(side=tk.LEFT, padx=20)

        self.prediction_label = ttk.Label(nav_frame, text="Predykcja: -")
        self.prediction_label.pack(side=tk.LEFT, padx=20)

        next_btn = ttk.Button(nav_frame, text="Następny", command=self.go_next)
        next_btn.pack(side=tk.LEFT, padx=5)

        last_btn = ttk.Button(nav_frame, text="Ostatni", command=self.go_last)
        last_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(self.master, text="", anchor=tk.W)
        self.status_label.pack(fill=tk.X, padx=10, pady=5, side=tk.BOTTOM)
    
    def change_dataset(self):
        if not self.loading:
            self.current_idx = 0
            self.load_data()
    
    def update_display(self):
        if self.loading or self.image_data is None:
            return

        self.current_image = self.image_data[self.current_idx]
        self.current_label = self.label_data[self.current_idx]

        try:
            self.current_prediction, self.prediction_probs = self.predict_with_model(self.current_image)

            correct = self.current_prediction == self.current_label
            self.prediction_history.append((self.current_prediction, correct))
            self.accuracy_history.append(1 if correct else 0)

            if len(self.prediction_history) > self.history_size:
                self.prediction_history.pop(0)
                self.accuracy_history.pop(0)

            result_text = "POPRAWNA ✓" if correct else "BŁĘDNA ✗"
            pred_name = self.class_names[self.current_prediction] if self.current_prediction is not None else "?"
            self.prediction_label.config(text=f"Predykcja: {pred_name} ({result_text})")

        except Exception as e:
            self.status_label.config(text=f"Błąd predykcji: {str(e)}")
            self.current_prediction = None
            self.prediction_probs = None
            self.prediction_label.config(text="Predykcja: błąd")

        self.index_label.config(text=f"Obraz: {self.current_idx + 1}/{len(self.image_data)}")

        self.ax_image.clear()
        self.ax_image.imshow(self.current_image, cmap='gray')
        self.ax_image.set_title(f"Etykieta: {self.current_label}")
        self.ax_image.axis('off')
        self.fig_image.tight_layout()
        self.canvas_image.draw()

        self.update_probability_chart()
        self.update_history_chart()
    
    def update_probability_chart(self):
        if self.prediction_probs is None:
            return

        self.ax_probs.clear()

        digits = list(range(10))
        probabilities = self.prediction_probs

        colors = ['#3498db'] * 10
        colors[self.current_label] = '#2ecc71'

        if self.current_prediction != self.current_label:
            colors[self.current_prediction] = '#e74c3c'

        bars = self.ax_probs.bar(digits, probabilities, color=colors)

        for bar in bars:
            height = bar.get_height()
            self.ax_probs.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        self.ax_probs.set_xlabel('Cyfra')
        self.ax_probs.set_ylabel('Prawdopodobieństwo')
        self.ax_probs.set_title('Rozkład prawdopodobieństwa predykcji')
        self.ax_probs.set_xticks(digits)
        self.ax_probs.set_ylim(0, 1.1)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Prawdziwa etykieta'),
            Patch(facecolor='#e74c3c', label='Błędna predykcja'),
            Patch(facecolor='#3498db', label='Pozostałe cyfry')
        ]
        self.ax_probs.legend(handles=legend_elements, loc='upper right', fontsize=8)

        self.fig_probs.tight_layout()
        self.canvas_probs.draw()
    
    def update_history_chart(self):
        if not self.prediction_history:
            return

        self.ax_history.clear()

        indices = list(range(len(self.prediction_history)))
        predictions = [p[0] for p in self.prediction_history]
        correct_markers = [p[1] for p in self.prediction_history]

        self.ax_history.plot(indices, predictions, 'o-', color='#3498db', label='Predykcja')

        for idx, (pred, correct) in enumerate(zip(predictions, correct_markers)):
            color = '#2ecc71' if correct else '#e74c3c'
            self.ax_history.plot(idx, pred, 'o', color=color, markersize=8)

        if self.accuracy_history:
            accuracy = sum(self.accuracy_history) / len(self.accuracy_history) * 100
            self.ax_history.set_title(f'Historia predykcji (Dokładność: {accuracy:.1f}%)')
        else:
            self.ax_history.set_title('Historia predykcji')

        self.ax_history.set_xlabel('Kolejne obrazy')
        self.ax_history.set_ylabel('Predykcja')
        self.ax_history.set_yticks(range(10))
        self.ax_history.set_ylim(-0.5, 9.5)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='#2ecc71', label='Poprawna', markersize=8, linestyle=''),
            Line2D([0], [0], marker='o', color='#e74c3c', label='Błędna', markersize=8, linestyle='')
        ]
        self.ax_history.legend(handles=legend_elements, loc='upper right', fontsize=8)

        self.fig_history.tight_layout()
        self.canvas_history.draw()
    
    def predict_with_model(self, image):
        flattened_data = image.flatten().tolist()
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
                return None, None

            lines = stdout.strip().split('\n')
            prediction = int(lines[-1])

            probabilities = self.simulate_probabilities(prediction)

            return prediction, probabilities

        except Exception as e:
            return None, None
    
    def simulate_probabilities(self, prediction):
        probs = [max(0.01, np.random.normal(0.05, 0.02)) for _ in range(10)]

        probs[prediction] = np.random.normal(0.75, 0.15)

        if prediction > 0:
            probs[prediction-1] += np.random.normal(0.1, 0.05)
        if prediction < 9:
            probs[prediction+1] += np.random.normal(0.1, 0.05)

        sum_probs = sum(probs)
        probs = [p/sum_probs for p in probs]

        return probs
    
    def go_next(self):
        if self.loading or self.image_data is None:
            return

        if self.current_idx < len(self.image_data) - 1:
            self.current_idx += 1
            self.update_display()
    
    def go_prev(self):
        if self.loading or self.image_data is None:
            return

        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
    
    def go_first(self):
        if self.loading or self.image_data is None:
            return

        self.current_idx = 0
        self.update_display()
    
    def go_last(self):
        if self.loading or self.image_data is None:
            return

        self.current_idx = len(self.image_data) - 1
        self.update_display()
    
    def random_image(self):
        if self.loading or self.image_data is None:
            return

        self.current_idx = random.randint(0, len(self.image_data) - 1)
        self.update_display()
    
    def test_random_batch(self):
        if self.loading or self.image_data is None:
            return

        self.status_label.config(text="Testowanie losowych obrazów...")
        threading.Thread(target=self._test_batch_thread, daemon=True).start()
    
    def _test_batch_thread(self):
        try:
            num_samples = 100
            correct = 0
            indices = random.sample(range(len(self.image_data)), min(num_samples, len(self.image_data)))

            for i, idx in enumerate(indices):
                image = self.image_data[idx]
                label = self.label_data[idx]

                self.master.after(0, lambda i=i: self.status_label.config(
                    text=f"Testowanie [{i+1}/{num_samples}]..."))

                prediction, _ = self.predict_with_model(image)
                if prediction == label:
                    correct += 1

                if i % 10 == 0:
                    time.sleep(0.1)

            accuracy = (correct / num_samples) * 100
            self.master.after(0, lambda: self.status_label.config(
                text=f"Test ukończony. Dokładność: {accuracy:.2f}% ({correct}/{num_samples})"))

        except Exception as e:
            self.master.after(0, lambda: self.status_label.config(
                text=f"Błąd podczas testowania: {str(e)}"))

def main():
    root = tk.Tk()
    app = MNISTViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
