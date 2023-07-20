import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tkinter as tk
from tkinter import ttk, messagebox, font

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import keras


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Number Classifier")
        self.root.geometry("620x900")  # Adjusted window size
        self.center_window()  # Center the main window
        self.root.resizable(True, True)

        self.threshold_label = tk.Label(self.root, text="Reference Number (between 0 and 1):")
        self.threshold_entry = tk.Entry(self.root)
        self.threshold_entry.insert(0, "0.5")

        self.number_label = tk.Label(self.root, text="Number to predict (between 0 and 1):")
        self.number_entry = tk.Entry(self.root)
        self.number_entry.insert(0, "0.52")

        self.layers_nodes_label = tk.Label(self.root, text="Layers/Nodes (Brain size):")
        self.layers_nodes_entry = tk.Entry(self.root)
        self.layers_nodes_entry.insert(0, "16")

        self.random_count_label = tk.Label(self.root, text="'Random number count' for training:")
        self.random_count_entry = tk.Entry(self.root)
        self.random_count_entry.insert(0, "1000")

        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict)

        self.reset_button = tk.Button(self.root, text="Reset Graph", command=self.reset_graph)

        self.guide_button = tk.Button(self.root, text="Guide", command=self.show_guide)

        self.exit_button = tk.Button(self.root, text="Exit", command=self.exit_program)

        self.threshold_label.grid(row=0, column=0, padx=10, pady=10)
        self.threshold_entry.grid(row=0, column=1, padx=10, pady=10)
        self.number_label.grid(row=1, column=0, padx=10, pady=10)
        self.number_entry.grid(row=1, column=1, padx=10, pady=10)
        self.layers_nodes_label.grid(row=2, column=0, padx=10, pady=10)
        self.layers_nodes_entry.grid(row=2, column=1, padx=10, pady=10)
        self.random_count_label.grid(row=3, column=0, padx=10, pady=10)
        self.random_count_entry.grid(row=3, column=1, padx=10, pady=10)
        self.predict_button.grid(row=4, column=0, padx=10, pady=10)
        self.reset_button.grid(row=4, column=1, padx=10, pady=10)
        self.guide_button.grid(row=11, column=0, padx=(10, 20), pady=10)

        self.progress_label = tk.Label(self.root, text="Training Progress:")
        self.progress_bar = ttk.Progressbar(self.root, mode="determinate", length=500)

        self.fig, self.ax = plt.subplots(2, 1, figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()

        self.progress_label.grid(row=5, column=0, columnspan=2, padx=10, pady=10)
        self.progress_bar.grid(row=6, column=0, columnspan=2, padx=10, pady=10)
        self.canvas.get_tk_widget().grid(row=7, column=0, columnspan=2, padx=10, pady=10)

        self.prediction_label = tk.Label(self.root, text="Prediction:")
        self.prediction_entry = tk.Entry(self.root, state="readonly")

        self.prediction_label.grid(row=8, column=0, columnspan=2, padx=10, pady=10)
        self.prediction_entry.grid(row=8, column=1, columnspan=2, padx=10, pady=10)

        self.final_loss_label = tk.Label(self.root, text="Final Loss:")
        self.final_loss_entry = tk.Entry(self.root, state="readonly")

        self.final_accuracy_label = tk.Label(self.root, text="Final Accuracy:")
        self.final_accuracy_entry = tk.Entry(self.root, state="readonly")

        self.final_loss_label.grid(row=9, column=0, columnspan=2, padx=10, pady=10)
        self.final_loss_entry.grid(row=9, column=1, columnspan=2, padx=10, pady=10)
        self.final_accuracy_label.grid(row=10, column=0, columnspan=2, padx=10, pady=10)
        self.final_accuracy_entry.grid(row=10, column=1, columnspan=2, padx=10, pady=10)

        self.exit_button.grid(row=11, column=1, padx=(10, 20), pady=10)

    def center_window(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 620  # Adjust the window width here
        window_height = 900  # Adjust the window height here
        x = int((screen_width / 2) - (window_width / 2))
        y = int((screen_height / 2) - (window_height / 2))
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def predict(self):
        threshold = float(self.threshold_entry.get())
        number = float(self.number_entry.get())
        layers_nodes_input = int(self.layers_nodes_entry.get())
        random_count = int(self.random_count_entry.get())

        X_train = np.random.uniform(0, 1, size=(random_count, 1)).round(8)
        y_train = (X_train >= threshold).astype(int)

        model = Sequential()
        layers_nodes = [layers_nodes_input] * 3
        model.add(Dense(layers_nodes[0], input_dim=1, activation='relu'))
        for nodes in layers_nodes[1:]:
            model.add(Dense(nodes, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        progress_callback = self.ProgressCallback()
        progress_callback.model = self
        history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0, callbacks=[progress_callback])

        prediction = model.predict(np.array([[number]]))
        prediction_value = f"{prediction[0][0]:.4f}"
        self.prediction_entry.configure(state="normal")
        self.prediction_entry.delete(0, tk.END)
        self.prediction_entry.insert(0, prediction_value)
        self.prediction_entry.configure(state="readonly")

        self.final_loss_entry.configure(state="normal")
        self.final_loss_entry.delete(0, tk.END)
        self.final_loss_entry.insert(0, f"{history.history['loss'][-1]:.4f}")
        self.final_loss_entry.configure(state="readonly")

        self.final_accuracy_entry.configure(state="normal")
        self.final_accuracy_entry.delete(0, tk.END)
        self.final_accuracy_entry.insert(0, f"{history.history['accuracy'][-1]:.4f}")
        self.final_accuracy_entry.configure(state="readonly")

        self.ax[0].plot(history.history['loss'], label='Loss')
        self.ax[1].plot(history.history['accuracy'], label='Accuracy', color='green')
        self.ax[0].legend()
        self.ax[1].legend()

        # Change line color for each prediction
        lines = self.ax[1].get_lines()
        last_line = lines[-1]
        last_line.set_color(np.random.rand(3))

        self.fig.tight_layout()
        self.canvas.draw()

    def reset_graph(self):
        self.ax[0].clear()
        self.ax[1].clear()
        self.canvas.draw()

    def exit_program(self):
        answer = messagebox.askyesno("Exit Confirmation", "Are you sure you want to exit?")
        if answer:
            self.root.quit()

    def show_guide(self):
        guide = """
        Number Classifier - User Guide

        1. Reference Number (between 0 and 1):
           - Enter a decimal number between 0 and 1.
           - This number serves as a reference for the classification.

        2. Number to predict (between 0 and 1):
           - Enter a decimal number between 0 and 1.
           - This number will be classified as higher or lower than the reference number.

        3. Layers/Nodes (Brain size):
           - Enter the number of nodes to use in each layer.
           - The model uses a three-layer architecture with the same number of nodes in each layer.

        4. 'Random number count' for training:
           - Enter the count of random numbers to generate for training the model.

        5. Predict Button:
           - Click this button to perform the prediction based on the provided inputs.

        6. Reset Graph Button:
           - Click this button to clear the training progress graph.

        Note: The prediction value closer to 1 indicates a higher prediction, while closer to 0 indicates a lower prediction.
        """

        guide_window = tk.Toplevel(self.root)
        guide_window.title("User Guide")
        guide_window.geometry("+{}+{}".format(self.root.winfo_x() + 50, self.root.winfo_y() + 50))
        font_style = font.Font(family="Arial", size=10)  # Adjust the font size here

        text = tk.Text(guide_window, font=font_style)
        text.insert(tk.END, guide)
        text.pack()

    class ProgressCallback(keras.callbacks.Callback):
        def __init__(self):
            self.progress = 0

        def on_epoch_end(self, epoch, logs=None):
            self.progress += 1
            self.model.progress_bar["value"] = self.progress
            self.model.progress_bar.update()

        def set_model(self, model):
            pass


gui = GUI()
gui.root.mainloop()
