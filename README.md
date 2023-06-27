[![build](https://github.com/haddocking/DeepRank-GNN-esm/actions/workflows/build.yml/badge.svg)](https://github.com/haddocking/DeepRank-GNN-esm/actions/workflows/build.yml)

# DeepRank-GNN-esm
Graph Network for protein-protein interface including language model features

## Installation

With Anaconda

1. Clone the repository
```bash
$ git clone https://github.com/DeepRank/DeepRank-GNN-esm.git
$ cd DeepRank-GNN-esm
```

2. Install either the CPU or GPU version of DeepRank-GNN-esm
```bash
$ conda env create -f environment-cpu.yml && conda activate deeprank-gnn-esm-cpu-env
```
OR
```bash
$ conda env create -f environment-gpu.yml && conda activate deeprank-gnn-esm-gpu-env
```

3. Install the command line tool
```bash
$ pip install .
```

4. Run the tests to make sure everything is working
```bash
$ pytest tests/
```

## Usage

### As a scoring function

We provide a command-line interface for DeepRank-GNN-ESM that can be used to score protein-protein complexes. The command-line interface can be used as follows:

```bash
$ deeprank-gnn-esm-predict -h
usage: deeprank-gnn-esm-predict [-h] pdb_file

positional arguments:
  pdb_file    Path to the PDB file.

optional arguments:
  -h, --help  show this help message and exit
```

Example, score the `2oob` complex

```bash
# download it
$ wget https://files.rcsb.org/view/2OOB.pdb -q

# make sure the environment is activated
$ conda activate deeprank-gnn-esm-cpu-env
(deeprank-gnn-esm-cpu-env) $ deeprank-gnn-esm-predict 2OOB.pdb
 2023-06-27 16:39:06,864 predict:51 INFO - Setting up workspace - /home/rodrigo/repos/DeepRank-GNN-esm/2OOB-gnn_esm_pred
 2023-06-27 16:39:06,865 predict:60 INFO - Reading sequence of PDB 2OOB.pdb
 2023-06-27 16:39:06,873 predict:86 INFO - Generating embedding for protein sequence.
 2023-06-27 16:39:06,873 predict:87 INFO - ################################################################################
 2023-06-27 16:39:10,740 predict:102 INFO - Read all.fasta with 2 sequences
 2023-06-27 16:39:10,742 predict:112 INFO - Processing 1 of 1 batches (2 sequences)
 2023-06-27 16:39:11,955 predict:155 INFO - ################################################################################
 2023-06-27 16:39:11,974 predict:160 INFO - Generating graph, using 23 processors
# ...
 2023-06-27 16:39:12,831 predict:233 INFO - Predicted fnat for 2OOB: 0.899
 2023-06-27 16:39:12,831 predict:243 INFO - Output written to /home/rodrigo/repos/DeepRank-GNN-esm/2OOB-gnn_esm_pred/output.csv
```

From the output above you can see that the predicted fnat for the 2oob complex is **0.899**, this information is also written to the `output.csv` file.

The command above will generate a folder in the current working directory, containing the following:

```
2OOB-gnn_esm_pred
├── 2OOB.A.pt
├── 2OOB.B.pt
├── 2OOB.pdb
├── GNN_esm_prediction.csv
├── GNN_esm_prediction.hdf5
├── GNN_esm_prediction_001.hdf5
├── graph.hdf5
└── output.csv
```

* * *
### As a framework


#### Generate ems-2 embeddings for your protein
1. Generate fasta sequence in bulk, use script 'get_fasta.py'
    ```bash
    usage: get_fasta.py [-h] pdb_dir output_fasta_name

    positional arguments:
      pdb_dir            Path to the directory containing PDB files
      output_fasta_name  Name of the combined output FASTA file

    options:
      -h, --help         show this help message and exit
    ```
2. Generate embeddings in bulk from combined fasta files, use the script provided inside esm-2 package,

    ```bash
    $ python esm_2_installation_location/scripts/extract.py \
        esm2_t33_650M_UR50D \
        all.fasta \
        tests/data/embedding/1ATN/ \
        --repr_layers 0 32 33 \
        --include mean per_tok
    ```
    Replace 'esm_2_installation_location' with your installation location, 'all.fasta' with fasta sequence generated above, 'tests/data/embedding/1ATN/' with the output folder name for esm embeddings

#### Generate graph
  * Example code to generate residue graphs in hdf5 format:
    ```python
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
    ```python
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

#### Use pre-trained models to predict
  * Example code to use pre-trained DeepRank-GNN-esm model
    ```python
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
