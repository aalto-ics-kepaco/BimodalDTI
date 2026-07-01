import torch
import residue_gign_data
import embedding_data
import gign_data
from residue_gign import ResidueGIGN
from decision_head import DecisionHead
from GIGN import GIGN
import json
from stopwatch import Stopwatch
from train_residue_gign import get_predictions as get_residue_gign_predictions
from train_lm_mlp import get_predictions as get_lm_mlp_predictions
from train_gign import get_predictions as get_gign_predictions
import numpy as np
from evaluation import show_metrics

new_drug = False

device = torch.device('cuda')

watch = Stopwatch()

for i in range(1, 6):
    print(i)



    watch.start('loading ResidueGIGN data')
    dataloader = residue_gign_data.get_kinase_test_dataloader(i, new_drug=new_drug)
    watch.stop()

    model = ResidueGIGN(35, 1024, 256)
    model.load_state_dict(torch.load(f'./../saved_models/residue_gign/model_residue_gign_{'new_drug_' if new_drug else ''}split_{i}.pth'))
    model.to(device)

    watch.start('ResidueGIGN predicting')
    y_true, y_pred = get_residue_gign_predictions(model, dataloader, device)
    watch.stop()
    bimodal_dti = np.array(y_pred)



    dataloader = None
    watch.start('loading LM-MLP data')
    dataloader = embedding_data.get_kinase_test_dataloader(False, i, new_drug=new_drug)
    watch.stop()

    model = DecisionHead(384 + 1024)
    model.load_state_dict(torch.load(f'./../saved_models/embeddings/model_lm_mlp_{'new_drug_' if new_drug else ''}split_{i}.pth'))
    model.to(device)

    watch.start('LM-MLP predicting')
    y_true, y_pred = get_lm_mlp_predictions(model, dataloader, device)
    watch.stop()
    bimodal_dti += np.array(y_pred)



    dataloader = None
    watch.start('loading GIGN data')
    dataloader = gign_data.get_kinase_test_dataloader(False, i, new_drug=new_drug)
    watch.stop()

    model = GIGN(35, 256)
    model.load_state_dict(torch.load(f'./../saved_models/gign/model_gign_{'new_drug_' if new_drug else ''}split_{i}.pth'))
    model.to(device)

    watch.start('GIGN predicting')
    y_true, y_pred = get_gign_predictions(model, dataloader, device)
    watch.stop()
    bimodal_dti += np.array(y_pred)
    bimodal_dti *= (1 / 3)

    

    show_metrics(y_true, bimodal_dti.tolist())

print('done')

