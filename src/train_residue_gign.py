import sys
import torch
from torch_geometric.loader import DataLoader
import residue_gign_data
from residue_gign import ResidueGIGN
import utils
from save_model import save_model
from evaluation import show_metrics
from plot_training_losses import plot_training_losses
from stopwatch import Stopwatch
from datetime import datetime
import numpy as np

watch = Stopwatch()

def get_predictions(
    model: ResidueGIGN,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch.to(device)
            z = model(batch)
            y = batch.y
            for i in range(y.shape[0]):
                y_true.append(y[i].item())
                y_pred.append(z[i].item())
    return y_true, y_pred

def calc_eval_loss(
    model: ResidueGIGN,
    data_loader: DataLoader,
    device: torch.device,
    criterion = torch.nn.functional.mse_loss
):
    y_true, y_pred = get_predictions(model, data_loader, device)
    loss = criterion(torch.tensor(y_pred), torch.tensor(y_true))
    return loss.item()

def train(
        model: ResidueGIGN,
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
        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()
            z = model(batch)
            loss = criterion(z, batch.y)
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
    print('gign training script')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    run_name = utils.get_from_args(args, '--run-name', datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
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
        train_loader, validation_loader = residue_gign_data.get_kinase_dataloaders(split, new_drug)
    else:
        print('using davis data')
        train_loader, validation_loader = residue_gign_data.get_davis_dataloaders()
    watch.stop()
    
    utils.create_directory_if_missing('./../saved_models/residue_gign/')
    model_save_file = utils.get_from_args(
        args, '--save-file', f'./../saved_models/residue_gign/model_{run_name}.pth'
    )
    model_checkpoint_file = utils.get_from_args(
        args, '--checkpoint-file', f'./../saved_models/residue_gign/checkpoint_{run_name}.pth'
    )
    model_final_file = utils.get_from_args(
        args, '--final-file', f'./../saved_models/residue_gign/final_{run_name}.pth'
    )
    print('save file:', model_save_file)
    print('checkpoint file:', model_checkpoint_file)
    print('final file:', model_final_file)

    model = ResidueGIGN(35, 1024, 256)
    if '--continue' in args:
        model.load_state_dict(torch.load(model_save_file))
        print('continuing with the saved model')
    model.to(device)

    lr = float(utils.get_from_args(args, '--lr', '0.0005'))
    print('learning rate:', lr)

    weight_decay = float(utils.get_from_args(args, '--weight-decay', '0.000001'))
    print('weight decay:', weight_decay)
    
    epochs = int(utils.get_from_args(args, '--epochs', '800'))
    print('epochs:', epochs)

    loss_plot_file = utils.get_from_args(args, '--loss-plot-file', f'losses_{run_name}.png')
    print('loss plot save file:', loss_plot_file)
    loss_json_file = f'losses_{run_name}.json'

    sys.stdout.flush()
    
    criterion = torch.nn.functional.mse_loss
    
    watch.start('training')
    train_losses, validation_losses = train(
        model, epochs, lr, train_loader, validation_loader, device,
        criterion, model_checkpoint_file, weight_decay
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
        test_loader = residue_gign_data.get_kinase_test_dataloader(split, new_drug)
        y_true, y_pred = get_predictions(model, test_loader, device)
        show_metrics(y_true, y_pred)

if __name__ == '__main__':
    main(sys.argv)
