import torch

from ch04.example_deep_neural_network import ExampleDeepNeuralNetwork, print_gradients
from ch04.feed_forward import FeedForward
from ch04.gpt_config import GPT_CONFIG_124M


ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)


layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1.0, 0.0, -1.0]])
torch.manual_seed(123)  # specify random seed for the initial for reproductivity
# vanishing gradient problem
model_without_shortcuct = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
print_gradients(model_without_shortcuct, sample_input)

print("====================")
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print_gradients(model_with_shortcut, sample_input)