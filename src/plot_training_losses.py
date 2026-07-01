import numpy as np
import matplotlib.pyplot as plt
import sys
import json

def plot_training_losses(
    epochs: int,
    train_losses: list[float],
    validation_losses: list[float],
    save_file: str | None = None,
    logarithmic_y = False,
    figsize: tuple[int, int] | None = None,
    title: str | None = 'Loss during training'
):
    train_epochs = np.linspace(0, epochs, len(train_losses))
    test_epochs = np.linspace(0, epochs, epochs + 1)
    epoch_size = int(len(train_losses) / epochs)
    averaged_train_losses = [train_losses[0]]
    for i in range(0, len(train_losses), epoch_size):
        averaged_train_losses.append(sum(train_losses[i : i + epoch_size]) / epoch_size)
    plt.figure(figsize=figsize)
    if logarithmic_y:
        plt.yscale('log')
    plt.plot(train_epochs, train_losses, '#1f77b4', label='train')
    plt.plot(test_epochs, averaged_train_losses, '#e7298a', label='train avg')
    plt.plot(test_epochs, validation_losses, '#ff7f0e', label='validation')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if title is not None:
        plt.title(title)
    plt.legend()
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight')
    else:
        plt.show()

def load_losses(model: str, split: int):
    with open(f'losses_{model}_split_{split}.json', 'r') as f:
        return json.load(f)

def main(args: list[str]):
    plt.rcParams.update({"font.size": 14})

    models = [
        '27_9_lm_mlp_baseline',
        '27_9_gign',
        '27_9_residue_gign'
    ]

    for model in models:
        split = 1
        data = load_losses(model, split)
        epochs = len(data['validation']) - 1
        plot_training_losses(
            epochs,
            data['training'],
            data['validation'],
            save_file=f'plotted_losses_{model}_split_{split}.png',
            logarithmic_y=True,
            figsize=(4, 4),
            title=None
        )

if __name__ == '__main__':
    main(sys.argv)

