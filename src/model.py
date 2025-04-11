from typing import List, Tuple
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import json

from src.tensor import Tensor
from src.layer import Layer, Dense
from src.weight_initializer import ZeroInitializer, RandomUniformInitializer, RandomNormalInitializer, GlorotUniformInitializer, HeNormalInitializer
from src.loss_function import LossFunction, MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy
from src.activation_function import Linear, ReLU, Sigmoid, HyperbolicTangent, Softmax, GELU, SILU
from src.optimizer import Optimizer, StochasticGradientDescent
from src.regularized_loss import RegularizedLoss

class FFNN:
    optimizer: Optimizer
    loss_function: LossFunction | RegularizedLoss
    layers: List[Layer]
    output: Tensor
    __train_history: List[Tuple[float, float]]

    def __init__(self, layers: List[Layer]) -> None:
        if not layers:
            raise ValueError("Layers cannot be empty!")

        if not layers[0].weights:
            raise ValueError(
                "Input size of the first layer must be specified.\n"
                "Example: FFNN([\n"
                "   Dense(16, activation=\"relu\", kernel_initializer=\"he_normal\", input_size=4),\n"
                "   Dense(32, activation=\"sigmoid\", kernel_initializer=\"he_normal\")\n"
                "])\n"
            )

        self.layers = layers
        for i in range(1, len(self.layers)):
            self.layers[i].initialize_weights(self.layers[i - 1].get_neuron_size())
        
        self.loss_function = None
        self.output = None
        self.__train_history = None

    def get_parameters(self) -> List[Tensor]:
        """
        Returns all parameters in the neural network.
        """
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.get_parameters())

        return parameters
    
    def print_history(self) -> None:
        """
        Prints training history of the model.
        """
        if not self.__train_history:
            raise RuntimeError("No training history available! Train the model using fit().")

        train_loss, val_loss = zip(*self.__train_history)
        
        for i in range(len(train_loss)):
            print(f"Epoch {i+1} \nTraining Loss: {train_loss[i]:.4f}")
            if val_loss[i] is not None:
                print(f"Validation Loss: {val_loss[i]:.4f}")
            print("------------------------")

    def compile(self, optimizer: str = "sgd", loss: str = "mean_squared_error") -> None:
        """
        Initializes optimizer and loss function of the network.
        """
        match optimizer:
            case opt if isinstance(opt, Optimizer):
                self.optimizer = opt 
            case "sgd":
                self.optimizer = StochasticGradientDescent()
            case _:
                raise ValueError(
                    f"Optimizer '{optimizer}' is not supported. "
                    "Supported parameters: 'sgd'"
                )
        
        self.optimizer.set_parameters(self.get_parameters())
        
        match loss:
            case "mean_squared_error":
                self.loss_function = MeanSquaredError
            case "binary_crossentropy":
                self.loss_function = BinaryCrossEntropy
            case "categorical_crossentropy":
                self.loss_function = CategoricalCrossEntropy
            case _:
                raise ValueError(
                    f"Loss function '{optimizer}' is not supported. "
                    "Supported parameters: 'mean_squared_error', 'binary_crossentropy', 'categorical_crossentropy'"
                )

    def forward(self, input: Tensor) -> Tensor:
        """
        Initiates forwardpropagation through the network.
        """
        x = input
        for layer in self.layers:
            x = x.add_x0()
            x = layer.forward(x)
        
        self.output = x
        self.output.tensor_type = "output"
        return x.data
    
    def backward(self, y_true: np.ndarray) -> None:
        """
        Computes gradient through backpropagation using automated differentiation.
        """
        if not self.output:
            raise RuntimeError("Forwardpropagation has not been done yet! Initiate forwardpropagation using forward().")
        elif y_true.shape[0] != self.output.data.shape[0]:
            raise ValueError(f"y_true and y_pred have different dimensions: {y_true.shape[0]} and {self.output.data.shape[0]}")

        if self.layers[-1].activation_function is Softmax:            
            def __backward():
                self.output._Tensor__children[0].gradient += self.output.data - y_true

            self.output._Tensor__backward = __backward

        loss = self.output.compute_loss(y_true, self.loss_function)
        loss.backward()

        if isinstance(self.loss_function, RegularizedLoss):
            self.loss_function.backward(y_true, self.output.data)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 10, batch_size: int = 32, verbose: bool = 0, validation_data: tuple = None) -> None:
        """
        Trains the model with given training data.
        Input must be a 2D NumPy array, containing one or multiple data records for training.
        """
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"X_train and y_train must have the same number of entries!")
        
        if self.optimizer is None:
            raise RuntimeError(f"Model has not been compiled yet! Compile the model using compile().")
        
        batches = []
        for i in range(math.ceil(X_train.shape[0] / batch_size)):
            batch = []
            for j in range(i * batch_size, min(i * batch_size + batch_size, X_train.shape[0])):
                batch.append((X_train[j], y_train[j]))
            
            batches.append(batch)

        self.__train_history = []

        for i in range(epochs):
            train_loss = []
            for batch in batches:
                self.optimizer.zero_grad()
                for sample in batch:
                    self.forward(Tensor(sample[0])) # X
                    loss = self.output.compute_loss(sample[1], self.loss_function) # y
                    train_loss.append(loss.data[0])
                    self.backward(sample[1]) # y
                
                self.optimizer.step()
            
            train_loss = np.mean(np.array(train_loss))
            val_loss = None

            if validation_data is not None:
                self.output = Tensor(np.array(self.predict(validation_data[0]))) # X
                val_loss = self.output.compute_loss(validation_data[1], self.loss_function) # y

            self.__train_history.append((train_loss, val_loss.data[0]))

            if verbose:
                progress = (i + 1) / epochs
                progress_bar_length = 20
                block = int(progress_bar_length * progress)

                print(f"\rEpoch {i+1}/{epochs}" + " " * (progress_bar_length + 10))
                print(f"Training Loss: {train_loss:.4f}")
                
                if val_loss is not None:
                    print(f"Validation Loss: {val_loss.data[0]:.4f}")
                
                print(f"-" * 20)
                print(f"[{'#' * block}{'-' * (progress_bar_length - block)}] {progress * 100:.2f}%", end="")
        
    def predict(self, X_test: np.ndarray) -> List[np.ndarray]:
        """
        Predicts the class of given test data.
        Input must be a 2D NumPy array, containing multiple data records to be predicted.
        """
        y_pred = []
        for sample in X_test:
            y_pred.append(self.forward(Tensor(sample)))
        
        return y_pred
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
        """
        Evaluates the model performance on the test set.
        Automatically detects if it's classification or regression based on loss function.
        """
        y_pred = self.predict(X_test)
        loss = self.loss_function.forward(y_test, y_pred)
        
        if self.loss_function is MeanSquaredError: 
            metric = np.mean((y_pred - y_test) ** 2)
            metric_name = "MSE"
        elif self.loss_function is CategoricalCrossEntropy:
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_test_labels = np.argmax(y_test, axis=1)
            metric = np.mean(y_pred_labels == y_test_labels)
            metric_name = "Accuracy"
        elif self.loss_function is BinaryCrossEntropy:
            y_pred_labels = (y_pred > 0.5).astype(int)
            metric = np.mean(y_pred_labels == y_test)
            metric_name = "Accuracy"
        else:
            raise ValueError("Unknown loss function type.")

        print(f"Loss: {loss:.4f}, {metric_name}: {metric:.4f}")
        return loss, metric
    
    def __get_node_of_layers(self) -> List[int]:
        """
        Returns a list of the number of nodes in each layer, including the bias node.
        """
        n_layers = []
        n_layers.append(len(self.layers[0].weights[0].data))

        for i in range(len(self.layers)):
            n_layers.append(self.layers[i].get_neuron_size() + 1)

        return n_layers

    def __repr__(self) -> str:
        """
        Visualizes the neural network.
        """
        list_of_layer = self.__get_node_of_layers()
        width = len(list_of_layer)
        height = max(list_of_layer) * 1.5
        max_nodes = max(list_of_layer)

        radius = 0.5
        font_size = 16
        layer_spacing = 2.5 / (width ** 0.5)

        if (max_nodes <= 10 and width <= 15):
            _, ax = plt.subplots(figsize=(width,height))
            ax.axis('off')
            ax.set_aspect('equal')

            radius /= (max_nodes ** 0.2)
            font_size /= (max_nodes ** 0.2)

            for i, layer in enumerate(list_of_layer):
                for j in range(layer - (1 if i == len(list_of_layer) - 1 else 0)):
                    node = plt.Circle(
                        (i * layer_spacing, -(j - layer / 2)),
                        radius=radius,
                        color = plt.cm.tab10(i%10),
                        fill=True,
                        zorder = 2
                    )
                    ax.add_patch(node)

                    label = None
                    if j == layer - 1 and i != width - 1:
                        label = f'b{i}'
                    elif i == 0:
                        label = f'x{j+1}'
                    elif i == width - 1:
                        label = f'o{j+1}'
                    else:
                        label = f'h{i}-{j+1}'
                    
                    ax.text(
                        i * layer_spacing,
                        -(j - layer / 2),
                        label,
                        ha='center',
                        va='center',
                        color='white',
                        fontsize=font_size,
                        zorder = 3
                    )

            for i in range(width - 1):
                for j in range(list_of_layer[i]):
                    for k in range(list_of_layer[i+1] - 1):
                        if (i + 1 != width - 1) and (k == list_of_layer[i+1] - 1):
                            continue

                        x_start = i * layer_spacing
                        y_start = -(j - list_of_layer[i] / 2)
                        x_finish = (i+1) * layer_spacing
                        y_finish = -(k - list_of_layer[i+1] / 2)

                        ax.plot(
                            [x_start,x_finish],
                            [y_start,y_finish],
                            color='black',
                            zorder=1
                        )

            plt.xlim(-1, (width - 1) * layer_spacing + 1)
            plt.ylim(-max(list_of_layer)/2 - 0.5, max(list_of_layer) / 2 + 0.5)
            plt.tight_layout()
            plt.show()

        else:
            _, ax = plt.subplots(figsize=(10,10))
            ax.axis('off')
            ax.set_aspect('equal')

            for i, layer in enumerate(list_of_layer):
                node = plt.Circle(
                    (i * layer_spacing, 0),
                    radius=radius,
                    color = plt.cm.tab10(i%10),
                    fill=True,
                    zorder = 2
                )
                ax.add_patch(node)

                label = None
                if i == 0:
                    label = 'x'
                elif i == width - 1:
                    label = 'o'
                else:
                    label = f'h{i}'
                label += f'({(layer - 2) if (i == width - 1) else (layer - 1)})'

                ax.text(
                    i * layer_spacing,
                    0,
                    label,
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=font_size,
                    zorder = 3
                )

                if i < width - 1:
                    ax.plot(
                        [i * layer_spacing, (i+1) * layer_spacing],
                        [0, 0],
                        color='black',
                        zorder=1
                    )
            plt.xlim(-1, (width - 1) * layer_spacing + 1)
            plt.ylim(-0.5, 0.5)
            plt.tight_layout()
            plt.show()

        print("Weights (W[n][m] indicates weight value from node n to node m):\n")
        for i in range(len(self.layers)):
            l = self.layers[i]
            for j in range(len(l.weights)):
                w = l.weights[j].data
                for k in range(len(w)):
                    source = f'b{i}' if k == 0 else f'h{i}-{k}' if i > 0 else f'x{k}'
                    dest = f'o{j+1}' if (i == len(self.layers) - 1) else f'h{i+1}-{j+1}'
                    print(f'W[{source}][{dest}] = {w[k]}')
                print()
            print()

        print("\nGradients (D[n] indicates the gradient of node n):\n")
        for i in range(len(self.layers)):
            l = self.layers[i].gradients
            for j in range(len(l)):
                label = f'o{j+1}' if (i == len(self.layers) - 1) else f'h{i+1}-{j+1}'
                print(f"D[{label}] = {l[j].gradient}")
            print()
        return ""
    
    def plot_weights(self, layer: List[int]) -> None:
        """
        Plot weights distribution from multiple layers.
        """
        if min(layer) < 1:
            raise ValueError(
                "Any layer number must not be less than 1.\n"
                "Selecting layer n will plot the weights between layer n and layer n-1.\n"
                "Example for plotting weight between input layer (layer 0) and the first hidden layer (layer 1):\n"
                "plot_weights([1])\n"
            )
        if max(layer) > len(self.layers):
            raise ValueError(
                f"The amount of layers in the model is {len(self.layers)}.\n"
                "Any layer number must not exceed the amount of layers.\n"
            )
        
        for i in layer:
            self.layers[i-1].plot_dist(True, i, len(self.layers))

    def plot_gradients(self, layer: List[int]) -> None:
        """
        Plot gradients distribution from multiple layers.
        """
        if min(layer) < 1:
            raise ValueError(
                "Any layer number must not be less than 1\n"
                "Selecting layer n will plot the gradients of layer n\n"
                "Example for plotting gradients from the first hidden layer (layer 1):\n"
                "plot_gradients([1])\n"
            )
        if max(layer) > len(self.layers):
            raise ValueError(
                f"The amount of layers in the model is {len(self.layers)}\n"
                "Any layer number must not exceed the amount of layers\n"
            )
        
        for i in layer:
            self.layers[i-1].plot_dist(False, i, len(self.layers))
    
    def save(self, file_path: str) -> None:
        """
        Saves the model into a JSON file.
        """
        if not isinstance(file_path, str):
            raise TypeError("file_path must be a string")

        if not file_path.strip():
            raise ValueError("file_path cannot be empty or just spaces")

        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            raise FileNotFoundError(f"Directory does not exist: {dir_name}")

        print(f"Saving file to: {file_path}")

        model_data = {
            "layers": [],
            "output": {
                "data": self.output.data.tolist() if self.output else [0.0] * self.layers[-1].get_neuron_size(),
                "gradient": self.output.gradient.tolist() if self.output else [0.0] * self.layers[-1].get_neuron_size()
            },
            "optimizer": {
                "type": type(self.optimizer).__name__,
                "learning_rate": self.optimizer.learning_rate
            },
            "loss_function": self.loss_function.__name__,
            "train_history": self.__train_history if self.__train_history else []
        }

        for layer in self.layers:
            layer_dict = {
                "neuron_size": layer.get_neuron_size(),
                "weights": [],
                "gradients": [],
                "weight_initializer": {
                    "type": type(layer.weight_initializer).__name__,
                    "config": {}
                },
                "activation_function": layer.activation_function.__name__
            }

            for weight in layer.weights:
                layer_dict["weights"].append({
                    "data": weight.data.tolist(),
                    "gradient": weight.gradient.tolist()
                })

            for gradient in layer.gradients:
                layer_dict["gradients"].append({
                    "data": gradient.data.tolist(),
                    "gradient": gradient.gradient.tolist()
                })

            initializer = layer.weight_initializer
            if isinstance(initializer, RandomUniformInitializer):
                layer_dict["weight_initializer"]["config"] = {
                    "seed": initializer.seed,
                    "lower_bound": initializer.lower_bound,
                    "upper_bound": initializer.upper_bound
                }
            elif isinstance(initializer, RandomNormalInitializer):
                layer_dict["weight_initializer"]["config"] = {
                    "seed": initializer.seed,
                    "mean": initializer.mean,
                    "variance": initializer.variance
                }
            elif isinstance(initializer, (GlorotUniformInitializer, HeNormalInitializer)):
                layer_dict["weight_initializer"]["config"] = {
                    "seed": initializer.seed
                }

            model_data["layers"].append(layer_dict)

        with open(file_path, "w") as f:
            json.dump(model_data, f, indent=2)

        print(f"Model successfully saved to {file_path}!")

    def load(self, file_path: str) -> None:
        """
        Loads the JSON file created using save() to initialize model.
        """
        if not isinstance(file_path, str):
            raise TypeError("file_path must be a string!")
        if not file_path.strip():
            raise ValueError("file_path cannot be empty or just spaces!")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"File is not readable: {file_path}")

        print(f"Loading model from: {file_path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        activations = {
            "Linear": Linear,
            "ReLU": ReLU,
            "Sigmoid": Sigmoid,
            "HyperbolicTangent": HyperbolicTangent,
            "Softmax": Softmax,
            "GELU": GELU,
            "SILU": SILU
        }

        loss_functions = {
            "MeanSquaredError": MeanSquaredError,
            "BinaryCrossEntropy": BinaryCrossEntropy,
            "CategoricalCrossEntropy": CategoricalCrossEntropy
        }

        self.layers = []
        for layer_data in data["layers"]:
            neuron_size = layer_data["neuron_size"]
            layer = Dense(neuron_size)

            layer.weights = []
            for w in layer_data["weights"]:
                tensor = Tensor(np.array(w["data"]))
                tensor.gradient = np.array(w["gradient"])
                layer.weights.append(tensor)

            layer.gradients = []
            for g in layer_data["gradients"]:
                tensor = Tensor(np.array(g["data"]))
                tensor.gradient = np.array(g["gradient"])
                layer.gradients.append(tensor)

            init_type = layer_data["weight_initializer"]["type"]
            init_config = layer_data["weight_initializer"]["config"]

            if init_type == "ZeroInitializer":
                layer.weight_initializer = ZeroInitializer()
            elif init_type == "RandomUniformInitializer":
                layer.weight_initializer = RandomUniformInitializer(
                    lower_bound=init_config["lower_bound"],
                    upper_bound=init_config["upper_bound"],
                    seed=init_config["seed"]
                )
            elif init_type == "RandomNormalInitializer":
                layer.weight_initializer = RandomNormalInitializer(
                    mean=init_config["mean"],
                    variance=init_config["variance"],
                    seed=init_config["seed"]
                )
            elif init_type == "GlorotUniformInitializer":
                layer.weight_initializer = GlorotUniformInitializer(
                    seed=init_config["seed"]
                )
            elif init_type == "HeNormalInitializer":
                layer.weight_initializer = HeNormalInitializer(
                    seed=init_config["seed"]
                )
            else:
                raise ValueError(f"Unsupported weight initializer: {init_type}")

            activation_name = layer_data["activation_function"]
            layer.activation_function = activations[activation_name]

            self.layers.append(layer)

        output_data = np.array(data["output"]["data"])
        output_gradient = np.array(data["output"]["gradient"])
        self.output = Tensor(output_data)
        self.output.gradient = output_gradient

        opt_data = data["optimizer"]
        if opt_data["type"] == "StochasticGradientDescent":
            self.optimizer = StochasticGradientDescent(opt_data["learning_rate"])
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_data['type']}")

        self.loss_function = loss_functions[data["loss_function"]]

        self.__train_history = [tuple(entry) for entry in data.get("train_history", [])]

        print(f"Model successfully loaded from {file_path}!")
