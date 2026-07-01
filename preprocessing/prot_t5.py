from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import pickle

batch_size = 1
cache_dir = "./../.model_cache"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')

tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False, cache_dir=cache_dir)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", cache_dir=cache_dir).to(device)

with open('./../data/proteins/sequences.txt', 'r') as f:
    sequences = list(map(lambda line: line.strip(), f.readlines()))
count = len(sequences)

representations = torch.zeros((count, 1024))
residue_representations = list()
index = 0
while index < count:
    stop = min(index + batch_size, count)
    print(f'{index}-{stop}/{count}')
    seq = sequences[index:stop]
    seq = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seq]
    ids = tokenizer.batch_encode_plus(seq, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    
    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)

    for i in range(index, stop):
        size = len(sequences[i])
        emb = embedding_repr.last_hidden_state[i-index,:size].to(cpu)
        representations[i] = emb.mean(dim=0)
        residue_representations.append(emb)

    index += batch_size

torch.save(representations, './../data/proteins/representations.pt')

with open('./../data/proteins/residue_representations', 'wb') as f:
    pickle.dump(residue_representations, f)
