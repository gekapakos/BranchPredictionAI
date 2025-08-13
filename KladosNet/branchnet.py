import torch
import torch.nn as nn
import numpy as np

# Helper functions and classes remain the same

def lists_have_equal_length(list_of_lists):
  """Helper function to check that the length of lists are equal."""
  set_of_lengths = set(map(len, list_of_lists))
  return len(set_of_lengths) <= 1


def extract_slice_history(x, config, global_shift, slice_id):
  """Extract a portion of history for a slice."""
  total_history_size = x.shape[1]
  slice_size = config['history_lengths'][slice_id]
  pooling_width = config['pooling_widths'][slice_id]
  assert slice_size <= total_history_size

  if config['shifting_pooling'][slice_id]:
    slice_shift = global_shift % pooling_width
    inputs = []
    for i in range(x.shape[0]):
      slice_end = total_history_size - slice_shift[i]
      slice_start = slice_end - slice_size
      inputs.append(x[i, slice_start:slice_end])
    return torch.stack(inputs)
  else:
    return x[:, -slice_size:]


class Slice(nn.Module):
  """A Pytorch neural network module class to define a BranchNet slice
    corresponding to some portion of the history.
  """

  def __init__(self, slice_id):
    """Creates all the layers and computes the expected output size."""
    super(Slice, self).__init__()

    # Config values directly assigned
    history_length = [42, 78, 150, 294, 582][slice_id]  # history_lengths
    conv_filters = [32, 32, 32, 32, 32][slice_id]  # conv_filters
    conv_width = 7  # conv_widths
    pooling_width = [3, 6, 12, 24, 48][slice_id]  # pooling_widths
    embedding_dims = 32  # embedding_dims

    # Declare all the neural network layers
    self.embedding_table = nn.Embedding(2 ** 12, embedding_dims)  # pc_hash_bits
    self.conv = nn.Conv1d(embedding_dims, conv_filters, conv_width)
    self.batchnorm = nn.BatchNorm1d(conv_filters)

    self.pooling = nn.AvgPool1d(pooling_width, padding=0)

    # compute the slice output size
    conv_output_size = (history_length - conv_width + 1)
    pooling_output_size = conv_output_size // pooling_width
    self.total_output_size = pooling_output_size * conv_filters

  def forward(self, x):
    """Forward pass for the slice."""
    history_length = 150  # history_lengths[3] for example
    conv_filters = 32  # conv_filters[3]
    conv_width = 7  # conv_widths[3]
    pooling_width = 12  # pooling_widths[3]

    x = self.embedding_table(x)
    x = torch.transpose(x, 1, 2)
    x = self.conv(x)
    x = self.batchnorm(x)

    # pooling
    if pooling_width == -1:
      x = torch.sum(x, 2, keepdim=True)
    elif pooling_width > 0:
      x = self.pooling(x) * pooling_width
    
    return x.view(-1, self.total_output_size)


class BranchNetMLP(nn.Module):
  """MLP module for the BranchNet."""

  def __init__(self, flattened_input_dim):
    super(BranchNetMLP, self).__init__()
    
    # Directly use config values in the neural net
    hidden_neurons = [128, 128]  # hidden_neurons
    self.hidden_layers = nn.ModuleList()

    next_input_dim = flattened_input_dim
    for hidden_output_dim in hidden_neurons:
      self.hidden_layers.append(FCLayer(next_input_dim, hidden_output_dim, activation='relu'))  # hidden_fc_activation
      next_input_dim = hidden_output_dim

    self.last_layer = FCLayer(next_input_dim, 1, activation=None)

  def forward(self, x):
    for layer in self.hidden_layers:
      x = layer(x)
    x = self.last_layer(x)
    return x.squeeze(dim=1)


class BranchNet(nn.Module):
  """
  A Pytorch neural network module class to define BranchNet architecture.
  """

  def __init__(self):
    super(BranchNet, self).__init__()

    # The architecture will be based on fixed config values
    history_lengths = [42, 78, 150, 294, 582]  # history_lengths
    conv_filters = [32, 32, 32, 32, 32]  # conv_filters
    conv_widths = [7, 7, 7, 7, 7]  # conv_widths
    pooling_widths = [3, 6, 12, 24, 48]  # pooling_widths

    num_slices = len(history_lengths)
    self.slices = nn.ModuleList()
    concatenated_slices_output_size = 0
    for slice_id in range(num_slices):
      if conv_filters[slice_id] > 0:
        self.slices.append(Slice(slice_id))
        concatenated_slices_output_size += self.slices[slice_id].total_output_size
      else:
        self.slices.append(nn.ReLU())  # Insert dummy module for slices with no filters    

    self.mlp = BranchNetMLP(concatenated_slices_output_size)

  def forward(self, x):
    slice_outs = []
    for slice_id in range(len([42, 78, 150, 294, 582])):
      if [32, 32, 32, 32, 32][slice_id] > 0:  # conv_filters
        x_ = extract_slice_history(x, None, None, slice_id)
        x_ = self.slices[slice_id](x_)
        slice_outs.append(x_)
    x = torch.cat(slice_outs, dim=1)
    x = self.mlp(x)
    return x


class FCLayer(nn.Module):
  def __init__(self, input_dim, output_dim, activation):
    super(FCLayer, self).__init__()
    self.activation = activation
    self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
    self.bias = nn.Parameter(torch.empty(output_dim))
    self.randomize_weights()

  def forward(self, x):
    x = nn.functional.linear(x, self.weight, self.bias)
    if self.activation == 'relu':
      x = nn.ReLU()(x)
    return x

  def randomize_weights(self):
    output_dim = self.weight.shape[0]
    input_dim = self.weight.shape[1]
    glorot_init_bound = np.sqrt(2. / (input_dim + output_dim))
    self.weight.data.uniform_(-glorot_init_bound, +glorot_init_bound)
    self.bias.data.uniform_(-glorot_init_bound, +glorot_init_bound)
