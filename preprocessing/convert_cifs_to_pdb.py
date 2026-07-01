from openbabel import pybel
import os
import sys
import utils

def convert_entry(folder: str, entry: str):
    molecule = next(pybel.readfile('cif', f'{folder}{entry}/{entry}_model_0.cif'))
    molecule.write('pdb', f'{folder}{entry}/{entry}.pdb', overwrite=True)

def main(args: list[str]):
    folder = utils.get_from_args(args, '--path', './')
    entries = os.listdir(folder)
    start, end = utils.get_start_and_end(args, len(entries))
    i = start
    while i < end:
        entry = entries[i]
        convert_entry(folder, entry)
        sys.stdout.write(f'{i + 1 - start}/{end - start} ({entry})\n')
        sys.stdout.flush()
        i += 1

if __name__ == '__main__':
    main(sys.argv)
