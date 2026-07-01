import sys
import torch
from torch_geometric.loader import DataLoader
import gign_data
from GIGN import GIGN
import utils
from save_model import save_model
from evaluation import show_metrics
from plot_training_losses import plot_training_losses
from stopwatch import Stopwatch
from datetime import datetime
import numpy as np

watch = Stopwatch()

def get_predictions(
    model: GIGN,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_lm_embeddings = False,
    lm_embedding_weight = 1.0
):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch.to(device)
            z = model.embed(batch)
            if use_lm_embeddings:
                z = torch.concat((z, batch.embedding * lm_embedding_weight), dim=1)
            z = model.decide(z)
            y = batch.y
            for i in range(y.shape[0]):
                y_true.append(y[i].item())
                y_pred.append(z[i].item())
    return y_true, y_pred

def calc_eval_loss(
    model: GIGN,
    data_loader: DataLoader,
    device: torch.device,
    criterion = torch.nn.functional.mse_loss,
    use_lm_embeddings = True,
    lm_embedding_weight = 1.0
):
    y_true, y_pred = get_predictions(model, data_loader, device, use_lm_embeddings, lm_embedding_weight)
    loss = criterion(torch.tensor(y_pred), torch.tensor(y_true))
    return loss.item()

def train(
        model: GIGN,
        epochs: int,
        gnn_lr: float,
        mlp_lr: float,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        device: torch.device,
        criterion = torch.nn.functional.mse_loss,
        model_checkpoint_file: str | None = None,
        gnn_weight_decay = 0.0,
        mlp_weight_decay = 0.0,
        use_lm_embeddings = True,
        min_lm_embedding_weight = 1.0,
        max_lm_embedding_weight = 1.0
):
    gnn_optimizer = torch.optim.Adam(model.gnn.parameters(), lr=gnn_lr, weight_decay=gnn_weight_decay)
    mlp_optimizer = torch.optim.Adam(model.head.parameters(), lr=mlp_lr, weight_decay=mlp_weight_decay)

    train_losses: list[float] = []
    validation_loss: float = calc_eval_loss(model, validation_loader, device, criterion, use_lm_embeddings, min_lm_embedding_weight)
    validation_losses: list[float] = [validation_loss]
    print(f'validation loss: {validation_loss}')

    best_loss = validation_loss
    train_loss_index = 0

    lm_embedding_weights = np.linspace(min_lm_embedding_weight, max_lm_embedding_weight, epochs)

    for epoch in range(epochs):
        print(f'epoch {epoch + 1}')
        train_loss_index = len(train_losses)
        model.train()
        for batch in train_loader:
            batch.to(device)
            gnn_optimizer.zero_grad()
            mlp_optimizer.zero_grad()
            z = model.embed(batch)
            if use_lm_embeddings:
                z = torch.concat((z, batch.embedding * lm_embedding_weights[epoch]), dim=1)
            z = model.decide(z)
            loss = criterion(z, batch.y)
            loss.backward()
            gnn_optimizer.step()
            mlp_optimizer.step()
            loss_value = loss.item()
            train_losses.append(loss_value)
            sys.stdout.write(f'\r{loss_value:.8f}  ')
        sys.stdout.write('\n')
        print(f'average train loss: {sum(train_losses[train_loss_index:]) / (len(train_losses) - train_loss_index)}')
        validation_loss = calc_eval_loss(model, validation_loader, device, criterion, use_lm_embeddings, lm_embedding_weights[epoch])
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

    use_lm_embeddings = '--use-lm-embeddings' in args
    if use_lm_embeddings:
        print('using LM embeddings')
    else:
        print('not using LM embeddings')

    split = utils.get_from_args(args, '--split', None)
    if split is not None:
        split = int(split)

    new_drug = '--new-drug' in args
    if new_drug:
        print('using "new drug" split')

    watch.start('loading data')
    if '--kinase' in args:
        print('using kinase data')
        train_loader, validation_loader = gign_data.get_kinase_dataloaders(use_lm_embeddings, split, new_drug)
    else:
        print('using davis data')
        train_loader, validation_loader = gign_data.get_davis_dataloaders(use_lm_embeddings)
    watch.stop()
    
    utils.create_directory_if_missing('./../saved_models/gnn/')
    model_save_file = utils.get_from_args(
        args, '--save-file', f'./../saved_models/gign/model_{run_name}.pth'
    )
    model_checkpoint_file = utils.get_from_args(
        args, '--checkpoint-file', f'./../saved_models/gign/checkpoint_{run_name}.pth'
    )
    model_final_file = utils.get_from_args(
        args, '--final-file', f'./../saved_models/gign/final_{run_name}.pth'
    )
    print('save file:', model_save_file)
    print('checkpoint file:', model_checkpoint_file)
    print('final file:', model_final_file)

    model = GIGN(35, 256, 1664) if use_lm_embeddings else GIGN(35, 256)
    if '--continue' in args:
        model.load_state_dict(torch.load(model_save_file))
        print('continuing with the saved model')
    model.to(device)

    lr_str = utils.get_from_args(args, '--lr', '0.0005')
    gnn_lr = float(utils.get_from_args(args, '--gnn-lr', lr_str))
    mlp_lr = float(utils.get_from_args(args, '--mlp-lr', lr_str))
    print('gnn learning rate:', gnn_lr)
    print('mlp learning rate:', mlp_lr)

    wd_str = utils.get_from_args(args, '--weight-decay', '0.000001')
    gnn_weight_decay = float(utils.get_from_args(args, '--gnn-weight-decay', wd_str))
    mlp_weight_decay = float(utils.get_from_args(args, '--mlp-weight-decay', wd_str))
    print('gnn weight decay:', gnn_weight_decay)
    print('mlp weight decay:', mlp_weight_decay)
    
    epochs = int(utils.get_from_args(args, '--epochs', '800'))
    print('epochs:', epochs)

    loss_plot_file = utils.get_from_args(args, '--loss-plot-file', f'losses_{run_name}.png')
    print('loss plot save file:', loss_plot_file)
    loss_json_file = f'losses_{run_name}.json'

    min_lm_embedding_weight = float(utils.get_from_args(
        args, '--min-lm-embedding-weight', '1.0'
    ))
    max_lm_embedding_weight = float(utils.get_from_args(
        args, '--max-lm-embedding-weight', '1.0'
    ))
    print('min lm embedding weight:', min_lm_embedding_weight)
    print('max lm embedding weight:', max_lm_embedding_weight)

    sys.stdout.flush()
    
    criterion = torch.nn.functional.mse_loss
    
    watch.start('training')
    train_losses, validation_losses = train(
        model, epochs, gnn_lr, mlp_lr, train_loader, validation_loader, device,
        criterion, model_checkpoint_file, gnn_weight_decay, mlp_weight_decay,
        use_lm_embeddings, min_lm_embedding_weight, max_lm_embedding_weight
    )
    watch.stop()

    save_model(model, model_final_file, True)
    y_true, y_pred = get_predictions(model, validation_loader, device, use_lm_embeddings, max_lm_embedding_weight)
    print('final model')
    show_metrics(y_true, y_pred)

    model.load_state_dict(torch.load(model_checkpoint_file))

    plot_training_losses(
        epochs, train_losses, validation_losses, loss_plot_file,
        logarithmic_y=True
    )
    utils.save_json(loss_json_file, {'training': train_losses, 'validation': validation_losses})

    print('best model')
    y_true, y_pred = get_predictions(model, validation_loader, device, use_lm_embeddings, max_lm_embedding_weight)
    show_metrics(y_true, y_pred)

    save_model(model, model_save_file, '--force-save' in args)

    if '--show-test' in args:
        print('test data')
        test_loader = gign_data.get_kinase_test_dataloader(use_lm_embeddings, split, new_drug)
        y_true, y_pred = get_predictions(model, test_loader, device, use_lm_embeddings, max_lm_embedding_weight)
        show_metrics(y_true, y_pred)

if __name__ == '__main__':
    main(sys.argv)
