import torch

def save_model(model: torch.nn.Module, file: str, force = False):
    while not force:
        r = input(f'save model to {file} (yes/no)? ')
        if r == 'yes':
            break
        elif r == 'no':
            return
    torch.save(model.state_dict(), file)
