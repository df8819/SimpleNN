import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Number Classifier")
        self.root.geometry("620x680")  # Adjusted window size
        # self.root.resizable(False, False)
        self.root.resizable(True, True)

        # Entry field for threshold value
        self.threshold_label = tk.Label(self.root, text="Reference Number:")
        self.threshold_entry = tk.Entry(self.root)
        self.threshold_entry.insert(0, "")

        # Entry field for number to predict
        self.number_label = tk.Label(self.root, text="Number to predict:")
        self.number_entry = tk.Entry(self.root)
        self.number_entry.insert(0, "")

        # Entry field for changing layers/nodes
        self.layers_nodes_label = tk.Label(self.root, text="Layers/Nodes:")
        self.layers_nodes_entry = tk.Entry(self.root)
        self.layers_nodes_entry.insert(0, "64")

        # Entry field for the amount of randomly generated numbers
        self.random_count_label = tk.Label(self.root, text="Random count:")
        self.random_count_entry = tk.Entry(self.root)
        self.random_count_entry.insert(0, "1000")

        # Button to start prediction
        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict)

        # Button to exit the program
        self.exit_button = tk.Button(self.root, text="Exit", command=self.exit_program)

        # Arrange the widgets in the window
        self.threshold_label.grid(row=0, column=0, padx=10, pady=10)
        self.threshold_entry.grid(row=0, column=1, padx=10, pady=10)
        self.number_label.grid(row=1, column=0, padx=10, pady=10)
        self.number_entry.grid(row=1, column=1, padx=10, pady=10)
        self.layers_nodes_label.grid(row=2, column=0, padx=10, pady=10)
        self.layers_nodes_entry.grid(row=2, column=1, padx=10, pady=10)
        self.random_count_label.grid(row=3, column=0, padx=10, pady=10)
        self.random_count_entry.grid(row=3, column=1, padx=10, pady=10)
        self.predict_button.grid(row=4, column=0, padx=10, pady=10)
        self.exit_button.grid(row=4, column=1, padx=10, pady=10)

        # Figure for training loss and accuracy plot
        self.fig, self.ax = plt.subplots(2, 1, figsize=(6, 4))

        # Canvas for displaying the plot
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()

        # Arrange the widgets in the window
        self.canvas.get_tk_widget().grid(row=5, column=0, columnspan=2, padx=10, pady=10)

        # Prediction label and entry field
        self.prediction_label = tk.Label(self.root, text="Prediction:")
        self.prediction_entry = tk.Entry(self.root, state="readonly")

        # Arrange the prediction widgets in the window
        self.prediction_label.grid(row=6, column=0, padx=10, pady=10)
        self.prediction_entry.grid(row=6, column=1, padx=10, pady=10)

    def predict(self):
        threshold = float(self.threshold_entry.get())
        number = float(self.number_entry.get())
        layers_nodes_input = int(self.layers_nodes_entry.get())
        random_count = int(self.random_count_entry.get())

        # Prepare the data
        X_train = np.random.rand(random_count, 1)
        y_train = (X_train >= threshold).astype(int)

        # Define the model
        model = Sequential()
        layers_nodes = [layers_nodes_input] * 3  # Set all layers to the same number
        model.add(Dense(layers_nodes[0], input_dim=1, activation='relu'))
        for nodes in layers_nodes[1:]:
            model.add(Dense(nodes, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

        # Predict the class of the number
        prediction = model.predict(np.array([[number]]))
        prediction_value = f"{prediction[0][0]:.4f}"
        self.prediction_entry.configure(state="normal")
        self.prediction_entry.delete(0, tk.END)
        self.prediction_entry.insert(0, prediction_value)
        self.prediction_entry.configure(state="readonly")

        # Show training loss and accuracy
        messagebox.showinfo("Training Info", f"Final Loss: {history.history['loss'][-1]:.4f}\nFinal Accuracy: {history.history['accuracy'][-1]:.4f}")

        # Update training loss and accuracy plot
        self.ax[0].plot(history.history['loss'], label='Loss')
        self.ax[1].plot(history.history['accuracy'], label='Accuracy', color='green')
        self.ax[0].legend()
        self.ax[1].legend()
        self.fig.tight_layout()
        self.canvas.draw()

    def exit_program(self):
        answer = messagebox.askyesno("Exit Confirmation", "Are you sure you want to exit?")
        if answer:
            self.root.quit()

gui = GUI()
gui.root.mainloop()
