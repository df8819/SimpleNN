import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

class NeuralNetwork:
    def __init__(self):
        np.random.seed(42)  # For reproducibility
        self.weights = 2 * np.random.random((4, 4)) - 1  # Initialization of weights

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, outputs, iterations):
        loss_history = []
        for iteration in range(iterations):
            output = self.predict(inputs)
            error = outputs - output
            adjustment = np.dot(inputs.T, error * self.sigmoid_derivative(output))
            self.weights += adjustment
            loss = np.mean(np.square(outputs - output))  # Calculate mean squared error for visualization
            loss_history.append(loss)
        return loss_history

    def predict(self, inputs):
        return self.sigmoid(np.dot(inputs, self.weights))

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Neural Network Visualization")
        self.canvas = tk.Canvas(self.root, width=400, height=300, bg="white")  # Creating a canvas to draw the loss graph
        self.label_loss = tk.Label(self.root, text="Loss:", font=("Arial", 12))
        self.label_loss_val = tk.Label(self.root, text="", font=("Arial", 12))
        self.button = tk.Button(self.root, text="Start Training", command=self.start_training)  # Button to start training
        self.canvas.pack()
        self.label_loss.pack(side=tk.LEFT)
        self.label_loss_val.pack(side=tk.LEFT)
        self.button.pack()

    def draw_loss_graph(self, loss_history):
        self.canvas.delete("all")  # Clear canvas
        num_iterations = len(loss_history)
        max_loss = max(loss_history)
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        for i in range(num_iterations - 1):  # Draw lines connecting the points
            x1 = i * canvas_width / num_iterations
            y1 = canvas_height - (loss_history[i] / max_loss * canvas_height)
            x2 = (i+1) * canvas_width / num_iterations
            y2 = canvas_height - (loss_history[i+1] / max_loss * canvas_height)
            self.canvas.create_line(x1, y1, x2, y2, fill="blue")

    def start_training(self):
        inputs = np.array([[0, 0, 1, 1],
                           [0, 1, 0, 1],
                           [1, 1, 1, 1],
                           [0, 0, 0, 1]])

        outputs = np.array([[0, 1, 1, 0]]).T
        neural_network = NeuralNetwork()
        loss_history = neural_network.train(inputs, outputs, iterations=10000)
        self.draw_loss_graph(loss_history)
        predicted_output = neural_network.predict(inputs)
        self.label_loss_val.config(text=f"MSE: {loss_history[-1]:.4f}")
        print(f"Predicted Output: {predicted_output.T}")  # Printing the final predicted output

gui = GUI()
gui.root.mainloop()  # Start the main Tkinter loop
