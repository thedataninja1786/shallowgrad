import numpy as np
def gradcheck(loss,x,y,layers=[],e=1e-10):
    for layer_index, layer in enumerate(layers):
        grad_backprop = layer.grad

        grad_numerical = np.zeros_like(layer.weights)
        original_weights = np.copy(layer.weights)
        
        layer.weights = original_weights + e
        loss_plus = loss(x,y)
        layer.weights = original_weights - e
        loss_minus = loss(x,y)

        layer.weights = original_weights

        grad_numerical = (loss_plus - loss_minus) / (2.0 * e)

        relative_error = np.linalg.norm(grad_backprop - grad_numerical) / (np.linalg.norm(grad_backprop) + np.linalg.norm(grad_numerical))

        print('Relative Error for Layer {}: '.format(layer_index + 1), relative_error)
        assert relative_error < 1e-5, 'The gradient is incorrect!'