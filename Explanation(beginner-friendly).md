# Neural Network Learning Process
- The "Reference Number" and the "Number to Predict" play vital roles in the learning process of the neural network. 

## 1. Reference Number
- This is a threshold value that you set. In the context of your application, you're using this threshold to categorize numbers into two groups: numbers that are greater than or equal to this reference number, and numbers that are less than this reference number.

## 2. Number to Predict
- This is the new number that you want the neural network to categorize. After the neural network has been trained, you feed this number into the network, and it will output a prediction. The output is a number between 0 and 1. Numbers closer to 0 indicate that the network thinks the input is less than the reference number, while numbers closer to 1 indicate that the network thinks the input is greater than or equal to the reference number.

- The network "learns" by adjusting its internal weights based on the error it made on its predictions during the training process.

## 3. Data Preparation
- You generate a set of random numbers and categorize them based on whether they're greater than or equal to the reference number. These categorized numbers (0s and 1s) serve as the "ground truth" labels during the training process.

## 4. Model Definition
- You define a model and each layer has a certain number of nodes (neurons). Each node is responsible for learning different aspects of the data.

## 5. Training
- During training, the model takes the random numbers (features) and tries to predict their categories. Initially, the model's predictions are random because it hasn't learned anything yet.

## 6. Loss Calculation
- The model calculates the error (loss) between its predictions and the actual categories (ground truth labels). The aim is to minimize this error.

## 7. Backpropagation
- The model uses this error to adjust its internal weights. This process is called backpropagation. The adjustments are made in such a way that the error would be reduced in the next iteration.

## 8. Iteration
- Steps 3-5 are repeated many times (epochs). With each iteration, the model should become better at predicting the category of a number.

## 9. Prediction
- After being trained, the model utilizes its acquired weights to determine whether the 'Number to Predict' is equal to or greater than the reference number.

- The whole training on a 1000 random numbers is just for one single prediction of the classification of the "Number to predict".

- The goal of the model is to learn the boundary (the reference number) that separates the numbers into two categories. The better it can learn this boundary, the better it can predict the category of a new number.
