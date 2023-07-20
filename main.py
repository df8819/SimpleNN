import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tkinter as tk
from tkinter import ttk, messagebox, font

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import keras


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Number Classifier - 'Feed forward neural Network' for educational purposes")
        self.center_window()  # Center the main window

        self.threshold_label = tk.Label(self.root, text="Reference Number - Threshold (between 0 and 1):")
        self.threshold_entry = tk.Entry(self.root)
        self.threshold_entry.insert(0, "0.5")

        self.number_label = tk.Label(self.root, text="Number to predict (between 0 and 1):")
        self.number_entry = tk.Entry(self.root)
        self.number_entry.insert(0, "0.527")

        self.layers_label = tk.Label(self.root, text="Layers:")
        self.layers_entry = tk.Entry(self.root)
        self.layers_entry.insert(0, "3")

        self.nodes_label = tk.Label(self.root, text="Nodes per Layer:")
        self.nodes_entry = tk.Entry(self.root)
        self.nodes_entry.insert(0, "4")

        self.random_count_label = tk.Label(self.root, text="Random number count for training:")
        self.random_count_entry = tk.Entry(self.root)
        self.random_count_entry.insert(0, "1000")

        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict)

        self.reset_button = tk.Button(self.root, text="Reset Graph", command=self.reset_graph)

        self.visualize_button = tk.Button(self.root, text="Visualize Brain", command=self.visualize_brain)

        self.guide_button = tk.Button(self.root, text="Guide", command=self.show_guide)

        self.exit_button = tk.Button(self.root, text="Exit", command=self.exit_program)

        self.threshold_label.grid(row=0, column=0, padx=10, pady=10)
        self.threshold_entry.grid(row=0, column=1, padx=10, pady=10)

        self.number_label.grid(row=1, column=0, padx=10, pady=10)
        self.number_entry.grid(row=1, column=1, padx=10, pady=10)

        self.layers_label.grid(row=2, column=0, padx=10, pady=10)
        self.layers_entry.grid(row=2, column=1, padx=10, pady=10)

        self.nodes_label.grid(row=3, column=0, padx=10, pady=10)
        self.nodes_entry.grid(row=3, column=1, padx=10, pady=10)

        self.random_count_label.grid(row=4, column=0, padx=10, pady=10)
        self.random_count_entry.grid(row=4, column=1, padx=10, pady=10)

        self.predict_button.grid(row=5, column=0, padx=10, pady=10)
        self.reset_button.grid(row=5, column=1, padx=10, pady=10)
        self.visualize_button.grid(row=6, column=0, padx=10, pady=10)
        self.guide_button.grid(row=13, column=0, padx=(10, 20), pady=10)
        self.exit_button.grid(row=13, column=1, padx=(10, 20), pady=10)

        self.progress_label = tk.Label(self.root, text="Training Progress:")
        self.progress_bar = ttk.Progressbar(self.root, mode="determinate", length=500)

        self.fig, self.ax = plt.subplots(2, 1, figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()

        self.progress_label.grid(row=7, column=0, columnspan=2, padx=10, pady=10)
        self.progress_bar.grid(row=8, column=0, columnspan=2, padx=10, pady=10)
        self.canvas.get_tk_widget().grid(row=9, column=0, columnspan=2, padx=10, pady=10)

        self.prediction_label = tk.Label(self.root, text="Prediction certainty for 'Number to predict':")
        self.prediction_entry = tk.Entry(self.root, state="readonly")

        self.prediction_label.grid(row=10, column=0, columnspan=2, padx=10, pady=10)
        self.prediction_entry.grid(row=10, column=1, columnspan=2, padx=10, pady=10)

        self.final_loss_label = tk.Label(self.root, text="Overall Loss:")
        self.final_loss_entry = tk.Entry(self.root, state="readonly")

        self.final_accuracy_label = tk.Label(self.root, text="Overall Accuracy:")
        self.final_accuracy_entry = tk.Entry(self.root, state="readonly")

        self.final_loss_label.grid(row=11, column=0, columnspan=2, padx=10, pady=10)
        self.final_loss_entry.grid(row=11, column=1, columnspan=2, padx=10, pady=10)
        self.final_accuracy_label.grid(row=12, column=0, columnspan=2, padx=10, pady=10)
        self.final_accuracy_entry.grid(row=12, column=1, columnspan=2, padx=10, pady=10)

    def visualize_brain(self):
        # Display a messagebox and get the user's response
        response = tk.messagebox.askyesno("WARNING - High CPU demand",
                                          "WARNING:\n\nA big brain size (16 * 128 Neurons for example) can crash"
                                          " the app or take VERY long to load the visualization. (And will"
                                          " probably lag harder than 'Redfall' on release) \n\nContinue?")

        # If the user clicked 'No', return without doing anything
        if not response:
            return

        plt.close('all')  # Close all existing figures

        # Get the number of layers and nodes per layer from the input fields
        layers = int(self.layers_entry.get()) + 1  # Add one more hidden layer
        nodes_per_layer = int(self.nodes_entry.get())

        # Create a new graph
        G = nx.DiGraph()

        # Add node for input layer
        G.add_node((0, 0), pos=(0, 0.5), color='g')

        # Add nodes and edges for each hidden layer
        for layer in range(1, layers + 1):
            if layer == layers:  # output layer
                G.add_node((layer, 0), pos=(layer, 0.5), color='b')
                for prev_node in range(nodes_per_layer):
                    G.add_edge((layer - 1, prev_node), (layer, 0))
            else:  # hidden layers
                for node in range(nodes_per_layer):
                    # Calculate the position of the node
                    pos = (layer, node / (nodes_per_layer - 1) if nodes_per_layer > 1 else 0.5)

                    # Add the node to the graph
                    G.add_node((layer, node), pos=pos, color='y')

                    # Connect the node to all nodes in the previous layer
                    for prev_node in range(nodes_per_layer if layer > 1 else 1):
                        G.add_edge((layer - 1, prev_node), (layer, node))

        # Get node positions and colors
        pos = nx.get_node_attributes(G, 'pos')
        colors = [color for _, color in nx.get_node_attributes(G, 'color').items()]

        # Create a new figure and draw the graph
        fig, ax = plt.subplots()
        nx.draw(G, pos, node_color=colors, with_labels=True, ax=ax)

        # Show the plot
        plt.show()

    def center_window(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 620  # Adjust the window width here
        window_height = 980  # Adjust the window height here
        self.root.resizable(False, False)
        x = int((screen_width / 2) - (window_width / 2))
        y = int((screen_height / 2) - (window_height / 2))
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def predict(self):
        threshold = float(self.threshold_entry.get())
        number = float(self.number_entry.get())
        layers_input = int(self.layers_entry.get())
        nodes_input = int(self.nodes_entry.get())
        random_count = int(self.random_count_entry.get())

        x_train = np.random.uniform(0, 1, size=(random_count, 1)).round(8)
        y_train = (x_train >= threshold).astype(int)

        model = Sequential()
        layers_nodes = [nodes_input] * layers_input
        model.add(Dense(layers_nodes[0], input_dim=1, activation='relu'))
        for nodes in layers_nodes[1:]:
            model.add(Dense(nodes, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        progress_callback = self.ProgressCallback()
        progress_callback.model = self
        history = model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=0, callbacks=[progress_callback])

        prediction = model.predict(np.array([[number]]))
        prediction_value = "{:.3%}".format(prediction[0][0])
        self.prediction_entry.configure(state="normal")
        self.prediction_entry.delete(0, tk.END)
        self.prediction_entry.insert(0, prediction_value)
        self.prediction_entry.configure(state="readonly")

        self.final_loss_entry.configure(state="normal")
        self.final_loss_entry.delete(0, tk.END)
        loss_value = "{:.3%}".format(history.history['loss'][-1])
        self.final_loss_entry.insert(0, loss_value)
        self.final_loss_entry.configure(state="readonly")

        self.final_accuracy_entry.configure(state="normal")
        self.final_accuracy_entry.delete(0, tk.END)
        accuracy_value = "{:.3%}".format(history.history['accuracy'][-1])
        self.final_accuracy_entry.insert(0, accuracy_value)
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

        # Clear the prediction certainty field
        self.prediction_entry.configure(state='normal')
        self.prediction_entry.delete(0, tk.END)
        self.prediction_entry.configure(state='readonly')

        # Clear the overall loss field
        self.final_loss_entry.configure(state='normal')
        self.final_loss_entry.delete(0, tk.END)
        self.final_loss_entry.configure(state='readonly')

        # Clear the overall accuracy field
        self.final_accuracy_entry.configure(state='normal')
        self.final_accuracy_entry.delete(0, tk.END)
        self.final_accuracy_entry.configure(state='readonly')

    def exit_program(self):
        answer = messagebox.askyesno("Exit Confirmation", "Are you sure you want to exit?")
        if answer:
            self.root.quit()

    def show_guide(self):
        guide = """
        Number Classifier - 'Feed forward neural Network' - User Guide

        1. Reference Number - Threshold (between 0 and 1):
           - Enter a decimal number between 0 and 1.
           - This number serves as a reference for the classification.

        2. Number to predict if >= 'Reference number' (between 0 and 1):
           - Enter a decimal number between 0 and 1.
           - This number will be classified as higher or lower than the reference number.
           - The neural network will be trained to predict if this number is >= 'Reference Number'.

        3. Layers:
           - Enter the number of layers.
           - {layers} * {nodes} = Neurons.
           
        4. Nodes:
            - Enter the numbers of nodes in each layer.
            - {layers} * {nodes} = Neurons.

        5. Random number count for training:
           - Enter the count of random numbers to generate for training the model.

        6. Predict Button:
           - Click this button to perform the prediction based on the provided inputs.

        7. Reset Graph Button:
           - Click this button to clear the training progress graph.
           
        8. Visualize Brain Button:
            - Opens a window for a visual representation of the current model.   

        Note:
        - This app is for educational purposes and people who
            want to have a small insight in the world of neural networks.
        - The prediction value closer to 1 indicates a higher prediction,
            while closer to 0 indicates a lower prediction.
        - This model is set to 100 epochs and a batch size of 10.     
        """

        guide_window = tk.Toplevel(self.root)
        guide_window.title("User Guide")
        guide_window.geometry("+{}+{}".format(self.root.winfo_x() + 50, self.root.winfo_y() + 50))
        guide_window.resizable(False, False)
        font_style = font.Font(family="Arial", size=10)  # Adjust the font size here

        text = tk.Text(guide_window, font=font_style, width=80, height=40)  # Adjust the height to your desired value
        text.insert(tk.END, guide)
        text.pack()

    class ProgressCallback(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.progress = 0

        def on_epoch_end(self, epoch, logs=None):
            self.progress += 1
            self.model.progress_bar["value"] = self.progress
            self.model.progress_bar.update()

        def set_model(self, model):
            pass


gui = GUI()
gui.root.mainloop()
