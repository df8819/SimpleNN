import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tkinter as tk

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Number Classifier")
        self.root.geometry("400x200")

        # Entry field for threshold value
        self.threshold_label = tk.Label(self.root, text="Reference Number:")
        self.threshold_entry = tk.Entry(self.root)
        self.threshold_entry.insert(0, "")

        # Entry field for number to predict
        self.number_label = tk.Label(self.root, text="Number to predict:")
        self.number_entry = tk.Entry(self.root)
        self.number_entry.insert(0, "")

        # Button to start prediction
        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict)

        # Button to exit the program
        self.exit_button = tk.Button(self.root, text="Exit", command=self.root.quit)

        # Label to display prediction result
        self.result_label = tk.Label(self.root, text="")

        # Arrange the widgets in the window
        self.threshold_label.pack()
        self.threshold_entry.pack()
        self.number_label.pack()
        self.number_entry.pack()
        self.predict_button.pack()
        self.exit_button.pack()
        self.result_label.pack()

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
        model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

        # Predict the class of the number
        prediction = model.predict(np.array([[number]]))
        self.result_label.config(text=f"Prediction: {prediction[0][0]:.4f}")

gui = GUI()
gui.root.mainloop()
