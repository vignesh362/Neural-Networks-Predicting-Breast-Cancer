This repository implements a 3-layer neural network designed to differentiate benign and malignant breast cancer diagnoses. The network architecture comprises:

-> Input layer: 30 nodes (modify this value based on your input data)
-> Hidden layers: 2 layers with 60 nodes each (experiment with different configurations)
-> Output layer: 1 node using sigmoid activation for binary classification (benign vs. malignant)

Key features:
Sigmoid activation for hidden layers, providing smooth gradients during training.
ReLU (Rectified Linear Unit) activation for the output layer, promoting faster convergence and addressing vanishing gradient issues.
Clear focus on breast cancer classification, making it adaptable to similar binary classification problems.
