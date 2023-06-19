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
   Replace ${TORCH} and ${CUDA} with the specific PyTorch and CUDA versions of your system. 

  * esm_2: https://github.com/facebookresearch/esm

  * To install DeepRank-GNN-esm,
    ```
    git clone https://github.com/DeepRank/DeepRank-GNN-esm.git
    cd Deeprank-GNN-esm/
    pip install .
    ```

## Generate graph
  * Example code to generate residue graphs in hdf5 format:
    ```
    pdb_path = "./data/pdb/1ATN/"
    pssm_path = "./data/pssm/1ATN/"
    embedding_path = "./data/embedding/1ATN/"
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
  database_test = "1ATN_residue.hdf5"
  gnn = GINet
  target = "fnat"
  edge_attr = ["dist"]
  threshold = 0.3
  pretrained_model = deeprank-GNN-esm/paper_pretrained_models/scoring_of_docking_models/gnn_esm/treg_yfnat_b64_e20_lr0.001_foldall_esm.pth.tar
  node_feature = ["type", "polarity", "bsa", "charge", "embedding"]
  device_name = "cuda:0"
  model = NeuralNet(
      database_test,
      gnn,
      device_name = device_name,
      edge_feature = edge_attr,
      node_feature = node_feature,
      target = target,
      pretrained_model = pretrained_model,
      threshold = threshold)
  model.test(hdf5 = "tmpdir/GNN_esm_prediction.hdf5")
  ```

  

