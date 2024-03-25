import torch

class Baseline_laptime_average(torch.nn.Module):
    """Baseline laptime model that predicts the average laptime of the inputs.
    """

    def __init__(self, input_size: int, input_num: int):
        super().__init__()

        self.input_size = input_size
        self.input_num = input_num

    def forward(self, _input: torch.Tensor):
        """Forward pass of the model.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        batch_size = _input.shape[0]
        laptimes = _input.view(batch_size, self.input_num, self.input_size)[:, :, 0]
        predictions = torch.mean(laptimes, dim=1)

        return predictions
    
class Baseline_position_last(torch.nn.Module):
    """Baseline position model that simply predicts the same position as the 
    last input position.
    """

    def __init__(self, input_size: int, input_num: int, position_idx_start: int = 57):
        super().__init__()

        self.input_size = input_size
        self.input_num = input_num
        self.position_idx_start = position_idx_start

    def forward(self, _input: torch.Tensor):
        """Forward pass of the model.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        num_positions = 20

        batch_size = _input.shape[0]
        resized_input = _input.view(batch_size, self.input_num, self.input_size)

        one_hot_positions =resized_input[:, :, self.position_idx_start : self.position_idx_start + num_positions]
        return one_hot_positions[:, -1, :]


# if __name__ == "__main__":
#     x, y, D = 2, 2, 77
#     model = Baseline_position_average(D, y)

#     _input = torch.zeros(x, y, D)

#     _input[0, 0, 57] = 1
#     _input[0, 1, 59] = 1

#     _input[1, 0, 58] = 1
#     _input[1, 1, 76] = 1

#     _input = _input.view(x, y * D)

#     output = model(_input)
#     print(output.shape)

#     print(model(_input))