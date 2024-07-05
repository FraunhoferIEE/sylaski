import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
from syndatagenerators.data_preparation.datasets import ClassificationDataset


def labeled_dataloaders(x_real: torch.Tensor, x_fake: torch.Tensor, split=[.8, .2], test: bool = True):
    """
    Returns labeled train-,validation- and test dataloaders for the training
    and evaluation of the discriminative model.
    -------
    Input:
        x_real: torch.Tensor of shape [n_samples, feature_dim, seq_len] containing real samples
        x_fake: torch.Tensor of shape [n_samples, feature_dim, seq_len] containing fake samples
        split: fractions used for training/ validation and testing
        test: whether validation set is splitted in validatin and separate testing
    Returns:
        loader_train: DataLoader for training
        loader_val: DataLoader for validation
        if test:
            loader_test
    """
    dataset = ClassificationDataset(x_real, x_fake)
    val_len = int(split[1] * len(dataset))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    test_len = int(0.5 * len(val_set))

    loader_train = DataLoader(train_set, shuffle=True, batch_size=64)

    if test:
        val_set, test_set = random_split(val_set, [len(val_set) - test_len, test_len])
        loader_test = DataLoader(test_set, shuffle=True, batch_size=1)

    loader_val = DataLoader(val_set, shuffle=False, batch_size=1)

    if test:
        return loader_train, loader_val, loader_test
    else:
        return loader_train, loader_val


class LSTMClassifier(nn.Module):
    """Very simple implementation of an LSTM-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.to(self.device) for t in (h0, c0)]


def discriminative_score(x_real, x_fake, n_epochs: int = 100, lr: float = 0.001,
                         model_cls=LSTMClassifier, hidden_dim: int = 48, layer_dim: int = 2, output_dim=2, patience=20,
                         enhanced_output: bool = False):
    """
    Parameters
    ----------
    x_real: torch.Tensor
        real data of shape [n_real, feature_dim, seq_len]
    x_fake: torch.Tensor
        generated data of shape [n_fake, feature_dim, seq_len]
    n_epochs : int
        number of training epochs.
    lr : float
        learning rate for training.
    model_cls : optional
        class of the model trained for classification. The default is LSTMClassifier.
    hidden_dim : int, optional
        hidden dimension of LSTM model. The default is 48.
    layer_dim : int, optional
        layer dimension of LSTM model. The default is 2.
    output_dim : int, optional
        dimension of output. The default is 2. (since there are two classes, real vs fake)
    patience : int, optional
        parameter that defines when to perform early stopping if accuracy does not increase over number of time steps.
        The default is 20.
    enhanced_output: boolean whether  accuracies per epoch/on validation set shall be returned, or only test score

    Returns
    -------
    if enhancced_output:
        acc_total : numpy.array
            accuracies on validation set per each training epoch.
        best_acc : the best accuracy on validation set
        score_test: discriminative score on test set
        acc_test: accuracy on test set
    else:
        score_test
    """
    train_loader, val_loader, test_loader = labeled_dataloaders(x_real, x_fake)
    feature_dim = x_real.shape[1]
    print(feature_dim)
    # initialise LSTM model, define criteria
    if model_cls == LSTMClassifier:
        model = model_cls(feature_dim, hidden_dim, layer_dim, output_dim)
    else:
        model = model_cls()
    model = model.to(model.device).float()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss().to(model.device)

    acc_total = np.zeros(n_epochs)
    best_acc = 0
    trials = 0
    # train discriminative model on train set
    for epoch in range(1, n_epochs + 1):
        model.train()
        for i, (x_batch, label) in enumerate(train_loader):
            x_batch, label = x_batch.to(model.device), label.to(model.device)
            opt.zero_grad()
            torch.set_grad_enabled(True)
            out = model(x_batch.float())
            loss = criterion(out, label)
            loss.backward()
            opt.step()

        model.eval()
        # validate on validation loader
        correct, total = 0, 0
        for i, (x_val, y_val) in enumerate(val_loader):
            x_val, y_val = [t.to(model.device) for t in (x_val, y_val)]
            out = model(x_val.float())
            preds = F.log_softmax(out, dim=1).argmax(dim=1)
            total += y_val.size(0)
            correct += (preds == y_val).sum().item()

        acc = correct / total
        acc_total[epoch - 1] = acc

        if enhanced_output:
            print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Acc.: {acc:2.2%}')
        if acc > best_acc:
            trials = 0
            best_acc = acc
            torch.save(model.state_dict(), 'best.pth')  # saving model parameters of best accuracy
            print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
        else:
            trials += 1
            if trials >= patience:
                acc_total = acc_total[:epoch + 1]
                print(f'Early stopping on epoch {epoch}')
                break

    model.load_state_dict(torch.load('best.pth'))  # load best model
    model.eval()
    for i, (x_test, y_test) in enumerate(test_loader):
        x_test, y_test = [t.to(model.device) for t in (x_test, y_test)]
        out = model(x_test.float())
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        total += y_test.size(0)
        correct += (preds == y_test).sum().item()

    acc_test = correct / total
    print(f'Accuracy on test set {acc_test}')
    # discriminative score: 0.5-(1-accuracy)
    score_test = 0.5 - (1 - acc_test)

    if enhanced_output:
        return acc_total, best_acc, score_test, acc_test
    else:
        return score_test
