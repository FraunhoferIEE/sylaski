import torch
from torch import nn
from torch.autograd import Variable


def split_time_series(x: torch.Tensor, idx: int):
    """
    Splits a given batch of time series samples at the given index.
    Args:
        x: torch.Tensor of shape [n_samples, n_features, seq_len]. Input samples.
        idx: index where the time series shall be splitted.
    Returns:
        inout_seq: list of [sequence[0:idx-1], sequence[idx:]]
    """

    assert x.size(2) > idx, "Index out of range. Must be smaller than the length of the sequences."
    first_seq = x[:, :, :idx].transpose(1, 2)
    last_seq = x[:, :, idx:].transpose(1, 2)

    inout_seq = [first_seq, last_seq]

    return inout_seq


class LSTMPredictor(nn.Module):
    """
    LSTM model for predicting which takes parts of a sequence as input and learns to predict the subsequent steps
    of the sequence.
    """

    def __init__(self, input_features: int = 1, hidden_dim: int = 96, layer_dim: int = 2, output_features: int = 1,
                 pred_steps: int = 1):
        """
        Args:
            input_features: number of features of the input sequences.
            hidden_dim: hidden dimension of the lstm.
            layer_dim: number of (hidden) layers in the lstm.
            output_features: number of features in the output sequences.
            pred_steps: number of steps to "predict".
        """
        super().__init__()
        self.input_dim = input_features
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_features
        self.pred_steps = pred_steps

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.rnn = nn.LSTM(input_size=input_features, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_features)

    def forward(self, x: torch.Tensor):
        h0, c0 = self.init_hidden(x)
        out, (_, _) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -self.pred_steps:, :])

        return out

    def init_hidden(self, x: torch.Tensor):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        return [t.to(self.device) for t in (h0, c0)]


def predictive_score(train_x: torch.Tensor, val_x: torch.Tensor, pred_steps: int = 1, n_epochs: int = 30,
                     lr: float = 0.001, hidden_dim: int = 96, layer_dim: int = 2):
    """
    Calculates the "predictive score" according to the TSTR or TRTS method.
    Args:
        train_x: tensor of shape [n_samples, n_features, seq_len] to train the model on.
        val_x: tensor of shape [n_samples_val, n_features, seq_len] to validate the model on.
        pred_steps: number of time steps to predict.
        n_epochs: number of training epochs.
        lr: learning rate
        hidden_dim: hidden dimension of the lstm model.
        layer_dim: hidden layer dimension of the lstm model.

    """
    idx = train_x.size(2) - pred_steps
    n_features = train_x.size(1)

    train_x, train_y = split_time_series(train_x, idx=idx)
    val_x, val_y = split_time_series(val_x, idx=idx)
    assert train_x.size(1) == val_x.size(1), "train and validation sequences need to have the same number of features"
    assert train_x.size(2) == val_x.size(2), "train and validation data need to have the same sequence length"

    model = LSTMPredictor(input_features=n_features, hidden_dim=hidden_dim, layer_dim=layer_dim,
                          output_features=n_features, pred_steps=pred_steps)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # initialize tensors of train and val errors per epoch
    error_train = torch.zeros(n_epochs)
    error_val = torch.zeros(n_epochs)

    print("Start training the predictive model..")
    for epoch in range(1, n_epochs + 1):
        mean_error_train = 0
        for seq, label in zip(train_x, train_y):
            seq = torch.unsqueeze(seq, dim=0).to(device)
            label = torch.unsqueeze(label, dim=0).to(device)
            # train the model
            model.train()
            opt.zero_grad()
            label_pred = model(seq)
            loss = criterion(label, label_pred)
            # absolute error
            mean_error_train += abs(label - label_pred).mean()
            loss.backward()
            opt.step()
        mean_error_train = mean_error_train / train_x.size(0)
        error_train[epoch - 1] = mean_error_train

        # print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}')
        # print("Evaluating on validations set..")
        model.eval()
        mean_error_val = 0
        for seq, label in zip(val_x, val_y):
            seq = torch.unsqueeze(seq, dim=0)
            label = torch.unsqueeze(label, dim=0)
            with torch.no_grad():
                label_out = model(seq)
                error = abs(label_out - label).mean()
            mean_error_val += error
        mean_error_val = mean_error_val / val_x.size(0)
        print(f'MAE on validation set: {mean_error_val}')
        error_val[epoch - 1] = mean_error_val

    min_error_val = error_val.min()
    print(f'Lowest error on validation set {min_error_val} at epoch: {torch.argmax(error_val) + 1}')

    return error_train, error_val, min_error_val