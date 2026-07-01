from transformers import AutoTokenizer, AutoModel
import torch

batch_size = 512
cache_dir = "./../.model_cache"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM", cache_dir=cache_dir)
model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM", cache_dir=cache_dir).to(device)

with open('./../data/drugs/schulman_smiles_processed.txt', 'r') as f:
    smiles_strings = list(map(lambda line: line.strip(), f.readlines()))
count = len(smiles_strings)

representations = torch.zeros((count, 384))
index = 0
while index < count:
    stop = min(index + batch_size, count)
    print(f'{index}-{stop}/{count}')
    seq = smiles_strings[index:stop]
    inputs = tokenizer(seq, return_tensors="pt", padding='longest')
    
    # generate embeddings
    with torch.no_grad():
        outputs = model(**(inputs.to(device)))

    for i in range(index, stop):
        internal_index = i - index #the index inside the batch
        size = inputs['attention_mask'][internal_index].sum().item()
        emb = outputs.last_hidden_state[internal_index,:size]
        representations[i] = emb.mean(dim=0)

    index += batch_size

torch.save(representations, './../data/drugs/schulman_representations.pt')
