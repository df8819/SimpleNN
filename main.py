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
        self.root.geometry("600x580")  # Adjusted window size
        self.root.resizable(False, False)


        # Entry field for threshold value
        self.threshold_label = tk.Label(self.root, text="Reference Number:")
        self.threshold_entry = tk.Entry(self.root)
        self.threshold_entry.insert(0, "")

        # Entry field for number to predict
        self.number_label = tk.Label(self.root, text="Number to predict:")
        self.number_entry = tk.Entry(self.root)
        self.number_entry.insert(0, "")

        # Entry field for displaying prediction
        self.result_label = tk.Label(self.root, text="Prediction:")
        self.result_entry = tk.Entry(self.root)

        # Button to start prediction
        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict)

        # Button to exit the program
        self.exit_button = tk.Button(self.root, text="Exit", command=self.exit_program)

        # Arrange the widgets in the window
        self.threshold_label.grid(row=0, column=0, padx=10, pady=10)
        self.threshold_entry.grid(row=0, column=1, padx=10, pady=10)
        self.number_label.grid(row=1, column=0, padx=10, pady=10)
        self.number_entry.grid(row=1, column=1, padx=10, pady=10)
        self.result_label.grid(row=2, column=0, padx=10, pady=10)
        self.result_entry.grid(row=2, column=1, padx=10, pady=10)
        self.predict_button.grid(row=3, column=0, padx=10, pady=10)
        self.exit_button.grid(row=3, column=1, padx=10, pady=10)

        # Figure for training loss and accuracy plot
        self.fig, self.ax = plt.subplots(2, 1, figsize=(6, 4))

        # Canvas for displaying the plot
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=4, column=0, columnspan=2)

    def predict(self):
        threshold = float(self.threshold_entry.get())
        number = float(self.number_entry.get())

        # Prepare the data
        X_train = np.random.rand(1000, 1)
        y_train = (X_train >= threshold).astype(int)

        # Define the model
        model = Sequential()
        model.add(Dense(64, input_dim=1, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

        # Predict the class of the number
        prediction = model.predict(np.array([[number]]))
        self.result_entry.delete(0, tk.END)
        self.result_entry.insert(0, f"{prediction[0][0]:.4f}")

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
