import sys
import torch
from torch.utils.data import DataLoader
from embedding_data import get_kinase_dataloaders, get_davis_dataloaders, get_kinase_test_dataloader
from plot_training_losses import plot_training_losses
from save_model import save_model
import utils
from utils import create_directory_if_missing, get_from_args
from evaluation import show_metrics
from stopwatch import Stopwatch
from decision_head import DecisionHead
from datetime import datetime
import numpy as np

watch = Stopwatch()

def get_predictions(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            z = model(x.to(device))
            for i in range(y.shape[0]):
                y_true.append(y[i].item())
                y_pred.append(z[i].item())
    return y_true, y_pred

def calc_eval_loss(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion = torch.nn.functional.mse_loss
):
    y_true, y_pred = get_predictions(model, data_loader, device)
    loss = criterion(torch.tensor(y_pred), torch.tensor(y_true))
    return loss.item()

def get_latents(model: DecisionHead, keys: list[int], data_loader: torch.utils.data.DataLoader, device: torch.device):
    latents = torch.zeros((np.max(keys) + 1, 128))
    model.eval()
    index = 0
    with torch.no_grad():
        for x, _ in data_loader:
            latent = model.get_latent(x.to(device))
            for i in range(latent.shape[0]):
                latents[keys[index]] = latent[i]
                index += 1
    return latents

def train(
        model: torch.nn.Module,
        epochs: int,
        lr: float,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        device: torch.device,
        criterion = torch.nn.functional.mse_loss,
        model_checkpoint_file: str | None = None,
        weight_decay = 0.0
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses: list[float] = []
    validation_loss: float = calc_eval_loss(model, validation_loader, device, criterion)
    validation_losses: list[float] = [validation_loss]
    print(f'validation loss: {validation_loss}')

    best_loss = validation_loss
    train_loss_index = 0

    for epoch in range(epochs):
        print(f'epoch {epoch + 1}')
        train_loss_index = len(train_losses)
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model.forward(x.to(device))
            loss = criterion(z, y.to(device))
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            train_losses.append(loss_value)
            sys.stdout.write(f'\r{loss_value:.8f}  ')
        sys.stdout.write('\n')
        print(f'average train loss: {sum(train_losses[train_loss_index:]) / (len(train_losses) - train_loss_index)}')
        validation_loss = calc_eval_loss(model, validation_loader, device, criterion)
        print(f'validation loss: {validation_loss}')
        validation_losses.append(validation_loss)

        if best_loss > validation_loss and model_checkpoint_file is not None:
            best_loss = validation_loss
            torch.save(model.state_dict(), model_checkpoint_file)
            print('saved a new checkpoint')
        
        sys.stdout.flush()
    
    return train_losses, validation_losses

def main(args: list[str]):
    print('lm mlp baseline training script')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    run_name = get_from_args(args, '--run-name', datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    print('run name:', run_name)

    split = utils.get_from_args(args, '--split', None)
    if split is not None:
        split = int(split)

    new_drug = '--new-drug' in args
    if new_drug:
        print('using "new drug" split')

    watch.start('loading data')
    if '--kinase' in args:
        print('using kinase data')
        train_loader, validation_loader = get_kinase_dataloaders(False, split, new_drug)
    else:
        print('using davis data')
        train_loader, validation_loader = get_davis_dataloaders(False)
    watch.stop()
    
    create_directory_if_missing('./../saved_models/embeddings/')
    model_save_file = get_from_args(
        args, '--save-file', f'./../saved_models/embeddings/model_{run_name}.pth'
    )
    model_checkpoint_file = get_from_args(
        args, '--checkpoint-file', f'./../saved_models/embeddings/checkpoint_{run_name}.pth'
    )
    model_final_file = get_from_args(
        args, '--final-file', f'./../saved_models/embeddings/final_{run_name}.pth'
    )
    print('save file:', model_save_file)
    print('checkpoint file:', model_checkpoint_file)
    print('final file:', model_final_file)
    
    model = DecisionHead(384 + 1024)
    if '--continue' in args:
        model.load_state_dict(torch.load(model_save_file))
        print('continuing with the saved model')
    model.to(device)
    
    lr = float(get_from_args(args, '--lr', '0.00001'))
    print('learning rate:', lr)

    weigth_decay = float(get_from_args(args, '--weight-decay', '0.001'))
    print('weight decay:', weigth_decay)
    
    epochs = int(get_from_args(args, '--epochs', '2000'))
    print('epochs:', epochs)

    loss_plot_file = get_from_args(args, '--loss-plot-file', f'losses_{run_name}.png')
    print('loss plot save file:', loss_plot_file)
    loss_json_file = f'losses_{run_name}.json'

    sys.stdout.flush()
    
    criterion = torch.nn.functional.mse_loss
    #criterion = lambda a, b: torch.mean(torch.log(torch.cosh(a - b)))
    
    watch.start('training')
    train_losses, validation_losses = train(
        model, epochs, lr, train_loader, validation_loader, device,
        criterion, model_checkpoint_file, weigth_decay
    )
    watch.stop()

    save_model(model, model_final_file, True)
    y_true, y_pred = get_predictions(model, validation_loader, device)
    print('final model')
    show_metrics(y_true, y_pred)

    model.load_state_dict(torch.load(model_checkpoint_file))

    plot_training_losses(
        epochs, train_losses, validation_losses, loss_plot_file,
        logarithmic_y=True
    )
    utils.save_json(loss_json_file, {'training': train_losses, 'validation': validation_losses})

    print('best model')
    y_true, y_pred = get_predictions(model, validation_loader, device)
    show_metrics(y_true, y_pred)

    save_model(model, model_save_file, '--force-save' in args)

    if '--show-test' in args:
        print('test data')
        test_loader = get_kinase_test_dataloader(False, split, new_drug)
        y_true, y_pred = get_predictions(model, test_loader, device)
        show_metrics(y_true, y_pred)

if __name__ == '__main__':
    main(sys.argv)
