import h5py
import random

from deeprank_gnn.GraphGenMP import GraphHDF5
from deeprank_gnn.ginet import GINet
from deeprank_gnn.NeuralNet import NeuralNet


# example of creating hdf5 file
def create_graph():
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
        nproc = nproc,
        tmpdir = "./tmpdir",
    )


# example of adding target value to hdf5 file
def add_target(hdf5):
    hdf5_file = h5py.File(hdf5, "r+")
    for mol in hdf5_file.keys():
        fnat = random.random()
        bin_class = [1 if fnat > 0.3 else 0]
        hdf5_file.create_dataset(f"/{mol}/score/binclass", data=bin_class)
        hdf5_file.create_dataset(f"/{mol}/score/fnat", data=fnat)
    hdf5_file.close()
    print(f"{hdf5} target added!")


# exmaple of using pretrained model to predict
def predict(model):
    database_test = "1ATN_residue.hdf5"
    gnn = GINet
    target = "fnat"
    edge_attr = ["dist"]
    threshold = 0.3
    if model == "gnn_esm":
        pretrained_model = "../paper_pretrained_models/scoring_of_docking_models/gnn_esm/treg_yfnat_b64_e20_lr0.001_foldall_esm.pth.tar"
        node_feature = ["type", "polarity", "bsa", "charge", "embedding"]
        device_name = "cpu"
        model = NeuralNet(
            database_test,
            gnn,
            device_name = device_name,
            edge_feature = edge_attr,
            node_feature = node_feature,
            target = target,
            pretrained_model = pretrained_model,
            threshold = threshold,
        )
        model.test(hdf5 = "tmpdir/GNN_esm_prediction.hdf5")
    if model == "gnn_esm_pssm":
        pretrained_model = "../paper_pretrained_models/scoring_of_docking_models/gnn_esm_pssm/treg_yfnat_b50_e20_lr0.001_foldall_esm_pssm.pth.tar"
        node_feature = [
            "type",
            "polarity",
            "bsa",
            "charge",
            "embedding",
            "pssm",
            "cons",
            "ic",
        ]
        device_name = "cuda:0"
        model = NeuralNet(
            database_test,
            gnn,
            device_name = device_name,
            edge_feature = edge_attr,
            node_feature = node_feature,
            target = target,
            pretrained_model = pretrained_model,
            threshold = threshold,
        )
        model.test(hdf5 = "tmpdir/GNN_esm_pssm_prediction.hdf5")


def main():
    create_graph()
    add_target("1ATN_residue.hdf5")
    predict("gnn_esm")
    predict("gnn_esm_pssm")


if __name__ == "__main__":
    main()
