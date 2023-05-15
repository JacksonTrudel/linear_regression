import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy as cp

# ---------------
# General Constants
# ---------------
INPUT_FILENAME = "../datasets/house_price_data.txt"
BLUE_CIRCLE_PLT = 'bo'
RED_CIRCLE_PLT = 'ro'
GRAPH_PADDING = 2.0

# ---------------
# Modeling Constants
# ---------------
NUM_EPOCHS = 50
LEARNING_RATE = 0.1


def load_input_file(filename, cols):
    """
    Loads input data from csv.

    Returns (features, labels)
    """
    num_features = len(cols) - 1
    df = pd.read_csv(filename, sep=",", index_col=False)
    df.columns = cols
    data = np.array(df, dtype=float)

    features, labels = data[:, :num_features], data[:, -1]

    num_examples, num_features = features.shape[0], features.shape[1]
    # Add const feature (1) for y-intercept
    features = np.hstack((np.ones((num_examples, 1)), features))
    # Reshape to (NUM_EXAMPLES, 1)
    labels = np.reshape(labels, (num_examples, 1))

    return features, labels


def plot_inputs(features, labels, feature_names):
    """
    Plots inputs for dataset.
    """
    feature_count = len(feature_names)
    figure, axes = plt.subplots(1, feature_count)
    for i in range(feature_count):
        feature_vals = features[:, i:i+1]
        axes[i].set_title(feature_names[i])
        color = RED_CIRCLE_PLT if i % 2 == 0 else BLUE_CIRCLE_PLT
        axes[i].plot(feature_vals, labels, color)

    figure.tight_layout(pad=GRAPH_PADDING)
    plt.show()


def normalize(x):
    """
    Normalizes features
    """
    data = cp.deepcopy(x)
    means, std_divs = [], []
    num_features = data.shape[1]

    for i in range(0, num_features):
        feature_mean = np.mean(data[:, i])
        feature_std_div = np.std(data[:, i])

        means.append(feature_mean)
        std_divs.append(feature_std_div)
        if feature_std_div == 0:
            continue
        data[:, i] = ((data[:, i] - feature_mean)/feature_std_div)
    return data, means, std_divs


def predict(features, theta):
    """
    Given features with shape (NUM_EX, NUM_FEATURES),
    and weights (NUM_FEATURES, 1), returns the predicted values.

    Return: predictions (NUM_EX, 1)
    """
    # x.shape is (NUM_EXAMPLES, NUM_FEATURES)
    # theta.shape is (NUM_FEATURES, 1)
    # therefore, output is (NUM_EXAMPLES, 1),
    #   containing a prediction for each example
    return np.matmul(features, theta)


def get_MSE_costs(x, y, theta):
    """
    Implements MSE cost function
    """
    # predictions and differences have shape (NUM_EXAMPLES, 1)
    # ex. [ [1], [2], [3] ]
    num_examples = y.shape[0]
    predictions = predict(x, theta)
    differences = predictions - y

    differences_squared = np.matmul(differences.T, differences)
    # differences_squared has shape (NUM_EX, 1). Output has same dim
    return (differences_squared)/(2*num_examples)


def gradient_descent(x, y, theta, learning_rate, num_epochs):
    """
    Performs gradient descent to optimize weights (theta)
    """
    NUM_EXAMPLES = x.shape[0]
    thetas = []
    J_all = []

    for _ in range(num_epochs):
        thetas.append(theta)

        # predictions.shape: (NUM_EXAMPLES, 1)
        predictions = predict(x, theta)

        # cost derivatives
        cost_derivatives = (1/NUM_EXAMPLES) * np.matmul(x.T, predictions - y)

        # update theta
        theta = theta - (learning_rate)*cost_derivatives
        cost_after_update = get_MSE_costs(x, y, theta)[0][0]
        J_all.append(cost_after_update)

    return theta, thetas, J_all


def plot_costs_over_epochs(J_all):
    """
    Plots epoch vs MSE
    """
    n_epochs = [i for i in range(len(J_all))]
    jplot = np.array(J_all)
    n_epochs = np.array(n_epochs)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.plot(n_epochs, J_all, 'm', linewidth="5")
    plt.show()


def inference(weights, features, means, std_divs):
    """
    Given model weights, returns prediction
    """
    num_weights = weights.shape[0]

    assert (len(features) == num_weights - 1), f"""
        Expected {num_weights - 1} features
    """
    weights = np.reshape(weights, (1, num_weights))[0]

    prediction = weights[0]
    for i in range(1, num_weights):
        mean, std_div = means[i], std_divs[i]
        feature_normalized = (features[i-1] - mean)/std_div
        prediction += feature_normalized * weights[i]
    return prediction


# ---------------
# Read input file
# ---------------
raw_features, labels = load_input_file(
    INPUT_FILENAME, ["housesize", "rooms", "price"]
)

NUM_EXAMPLES = raw_features.shape[0]
# This includes constant (1) feature
NUM_FEATURES = raw_features.shape[1]

# ---------------
# Plot input data
# ---------------
plot_inputs(raw_features[:, 1:], labels, ["Home Size", "Number of Rooms"])

# ---------------
# Normalize input data
# ---------------
features_norm, means, std_divs = normalize(raw_features)

# ---------------
# Initialize Params
# ---------------
theta = np.zeros((raw_features.shape[1], 1))

# --------------
# Perform GD
# --------------
theta, all_thetas, J_all = gradient_descent(
    features_norm, labels, theta, LEARNING_RATE, NUM_EPOCHS
)
print(f"Final weights: {theta}")

# -------------
# Plot Cost function over time
# -------------
plot_costs_over_epochs(J_all)

# -------------
# Use for inference
# -------------
home_area, num_rooms = 1850, 4
features = [home_area, num_rooms]
predicted_price = inference(theta, features, means, std_divs)
print(f"Prediction for {features}: {predicted_price}")
