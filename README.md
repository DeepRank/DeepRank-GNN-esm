# DeepRank-GNN-esm
Graph Network for protein-protein interface including language model features

## Installation
Before installing DeepRank-GNN-esm you need to install pytorch, pytorch_geometric and esm_2 according to your needs. You can find detailed instructions here:
  * pytorch: https://pytorch.org/
  * pytorch_geometric: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
  * External library for pytorch_geometric can be installed by 
   ```
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f 
   https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
   ```
   Replace ${TORCH} and ${CUDA} with specific PyTorch and CUDA versions of your system. 

  * esm_2: https://github.com/facebookresearch/esm

  * To install DeepRank-GNN-esm,
    ```
    git clone https://github.com/DeepRank/DeepRank-GNN-esm.git
    cd Deeprank-GNN-esm/
    pip install .
    ```
 * To test installation,
   ```
   cd tests/
   pytest
   ```
## Generate ems-2 embeddings for your protein
* Step1. Generate fasta sequence in bulk, use script 'get_fasta.py'
```
usage: get_fasta.py [-h] pdb_dir output_fasta_name

positional arguments:
  pdb_dir            Path to the directory containing PDB files
  output_fasta_name  Name of the combined output FASTA file

options:
  -h, --help         show this help message and exit
```
* Step2. Generate embeddings in bulk from combined fasta files, use the script provided inside esm-2 package,
```
python esm_2_installation_location/scripts/extract.py esm2_t33_650M_UR50D all.fasta 
  tests/data/embedding/1ATN/ --repr_layers 0 32 33 --include mean per_tok
```
Replace 'esm_2_installation_location' with your installation location, 'all.fasta' with fasta sequence generated above, 'tests/data/embedding/1ATN/' with the output folder name for esm embeddings 

## Generate graph
  * Example code to generate residue graphs in hdf5 format:
    ```
    from deeprank_gnn.GraphGenMP import GraphHDF5
    
    pdb_path = "tests/data/pdb/1ATN/"
    pssm_path = "tests/data/pssm/1ATN/"
    embedding_path = "tests/data/embedding/1ATN/"
    nproc = 20
    outfile = "1ATN_residue.hdf5"

    GraphHDF5(
        pdb_path = pdb_path,
        pssm_path = pssm_path,
        embedding_path = embedding_path,
        graph_type = "residue",
        outfile = outfile,
        nproc = nproc,    #number of cores to use 
        tmpdir="./tmpdir")
    ```
  * Example code to add continuous or binary targets to the hdf5 file
    ```
    import h5py
    import random
   
    hdf5_file = h5py.File('1ATN_residue.hdf5', "r+")
    for mol in hdf5_file.keys():
        fnat = random.random()
        bin_class = [1 if fnat > 0.3 else 0]
        hdf5_file.create_dataset(f"/{mol}/score/binclass", data=bin_class)
        hdf5_file.create_dataset(f"/{mol}/score/fnat", data=fnat)
    hdf5_file.close()
    ```

## Use pre-trained models to predict
  * Example code to use pre-trained DeepRank-GNN-esm model
  ```
  from deeprank_gnn.ginet import GINet
  from deeprank_gnn.NeuralNet import NeuralNet

  database_test = "1ATN_residue.hdf5" 
  gnn = GINet 
  target = "fnat"
  edge_attr = ["dist"]
  threshold = 0.3
  pretrained_model = deeprank-GNN-esm/paper_pretrained_models/scoring_of_docking_models/gnn_esm/treg_yfnat_b64_e20_lr0.001_foldall_esm.pth.tar
  node_feature = ["type", "polarity", "bsa", "charge", "embedding"]
  device_name = "cuda:0"
  num_workers = 10
  
  model = NeuralNet(
      database_test,
      gnn,
      device_name = device_name,
      edge_feature = edge_attr,
      node_feature = node_feature,
      target = target,
      num_workers = num_workers,
      pretrained_model = pretrained_model,
      threshold = threshold)

  model.test(hdf5 = "tmpdir/GNN_esm_prediction.hdf5")
  ```

  

