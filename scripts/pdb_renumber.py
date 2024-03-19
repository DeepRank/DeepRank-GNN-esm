import os
import sys



def renumber_res(pdb_input, output_dir, start_res=1):

    ##get all chainIDs from the pdb file
    chainIDs = []
    with open(pdb_input, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                chainID = line[21]
                if chainID not in chainIDs:
                    chainIDs.append(chainID)
            
    ## split the chain in the pdb file with pdb_splitchain
    os.system(f'pdb_splitchain {pdb_input}')

    #get the current working directory
    cwd = os.getcwd()

    #renumber all the chains:
    all_outputs = []
    chains = []
    for chainID in chainIDs:
        chain = os.path.join(cwd, os.path.basename(pdb_input).replace('.pdb', f'_{chainID}.pdb'))
        output = chain.replace('.pdb', '_renumbered.pdb')
        all_outputs.append(output)
        os.system(f'pdb_reres -{start_res} {chain} > {output}')
        chains.append(chain)

    #merge all chains into one file 
    out_name = os.path.basename(pdb_input).replace('.pdb', '_merged.pdb')
    output = os.path.join(output_dir, out_name)

    #sort the outputs
    sorted_outputs = sorted(all_outputs)
    #combine items in the list to a string
    command = 'pdb_merge ' + ' '.join(sorted_outputs) + ' > ' + output
    os.system(command)

    #remove all the middle files
    middle_file = all_outputs + chains
    os.system(f'rm {" ".join(middle_file)}')


def show_help():
    print("""
    Usage: python script_name.py PDB_INPUT_FILE START_RESIDUE_NUMBER

    Description:
    This script renumbers the residue IDs in a PDB file starting from a specified residue number.

    Example:
    python script_name.py input.pdb 100 . 

    This will renumber the residues in 'input.pdb' starting from residue number 100 in the current dir
    """)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        show_help()
    else:
        pdb_input = sys.argv[1]
        start_res = int(sys.argv[2])
        output_dir = sys.argv[3]
        renumber_res(pdb_input, output_dir, start_res)
