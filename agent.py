import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, place_units, head_units, lstm_units, linear_units, num_features, dropout_prob=0.5):
        super(Network, self).__init__()

        self.lstm_units = lstm_units

        # LSTM_initialization
        self.Wcp = nn.Parameter(torch.rand((place_units, lstm_units)))
        self.Wcd = nn.Parameter(torch.rand((head_units, lstm_units)))
        self.Whp = nn.Parameter(torch.rand((place_units, lstm_units)))
        self.Whd = nn.Parameter(torch.rand((head_units, lstm_units)))

        # LSTM
        self.lstm = nn.LSTM(num_features, lstm_units, batch_first=True)

        # Linear Decoder
        self.linear = nn.Linear(lstm_units, linear_units)
        self.dropout = nn.Dropout(dropout_prob)

        # Linear for Place Cells
        self.linear_place_cell = nn.Linear(linear_units, place_units)

        # Linear for Head Cells
        self.linear_head_cell = nn.Linear(linear_units, head_units)

    def forward(self, input_tensor, place_cell_ground, head_cell_ground):
        # input_tensor should be [None, 100, self.num_features]

        # LSTM
        hidden_0 = torch.matmul(place_cell_ground, self.Wcp) + torch.matmul(head_cell_ground, self.Wcd)
        cell_0 = torch.matmul(place_cell_ground, self.Whp) + torch.matmul(head_cell_ground, self.Whd)

        outputs, (hidden_state, cell_state) = self.lstm(input_tensor, (hidden_0, cell_0))

        # Linear Decoder
        outputs = outputs.reshape(-1, self.lstm_units)
        outputs = self.linear(outputs)
        outputs = self.dropout(outputs)

        # Place Cells
        outputs_place = self.linear_place_cell(outputs)

        # Head Cells
        outputs_head = self.linear_head_cell(outputs)

        return outputs_place, outputs_head, self.linear_place_cell.weight, self.linear_head_cell.weight

    def linear_output(self, input_tensor, place_cell_ground, head_cell_ground):
        # input_tensor should be [None, 100, self.num_features]

        # LSTM
        hidden_0 = torch.matmul(place_cell_ground, self.Wcp) + torch.matmul(head_cell_ground, self.Wcd)
        cell_0 = torch.matmul(place_cell_ground, self.Whp) + torch.matmul(head_cell_ground, self.Whd)

        outputs, (hidden_state, cell_state) = self.lstm(input_tensor, (hidden_0, cell_0))

        # Linear Decoder
        outputs = outputs.reshape(-1, self.lstm_units)
        outputs = self.linear(outputs)
        return outputs