def get_key_files(split: int | None = None, new_drug = False):
    if split is None:
        print('using the original split')
        return './../data/keys/train_keys_clean.txt', \
               './../data/keys/validation_keys_clean.txt', \
               './../data/keys/test_keys_clean.txt'
    elif isinstance(split, int) and split >= 1 and split <= 5:
        s = 'new_drug_' if new_drug else ''
        print(f'using split {split}, new drug: {new_drug}')
        return f'./../data/keys/{s}split_{split}/train.txt', \
               f'./../data/keys/{s}split_{split}/validation.txt', \
               f'./../data/keys/{s}split_{split}/test.txt'
    else:
        raise Exception('split not recognized')
