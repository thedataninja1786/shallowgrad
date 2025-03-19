import numpy as np


def gradcheck(loss, x, y, layers=[], e=1e-10, threshold=1e-5):
    for i, layer in enumerate(layers):
        grad_backprop = layer.grad

        grad_numerical = np.zeros_like(layer.weights)
        original_weights = np.copy(layer.weights)

        for i in range(layer.weights.shape[0]):
            for j in range(layer.weights.shape[1]):
                layer.weights[i, j] = original_weights[i, j] + e
                loss_plus = loss(x, y)
                layer.weights[i, j] = original_weights[i, j] - e
                loss_minus = loss(x, y)
                layer.weights = original_weights

                grad_numerical[i, j] = (loss_plus - loss_minus) / (2.0 * e)

        relative_error = np.max(
            np.abs(grad_backprop - grad_numerical)
            / (np.abs(grad_backprop) + np.abs(grad_numerical))
        )

        print(f"Relative Error for layer {i + 1}: {relative_error}")
        assert relative_error < threshold, "The gradient is incorrect!"
