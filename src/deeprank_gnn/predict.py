# Command line interface for predicting fnat
import logging
import multiprocessing as mp
import os
import shutil
import re
import tempfile
import uuid
import warnings
from io import TextIOWrapper
from pathlib import Path
import argparse
import torch
from Bio.PDB import PDBParser
from esm import FastaBatchedDataset, pretrained

from deeprank_gnn.ginet import GINet
from deeprank_gnn.GraphGenMP import GraphHDF5
from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.tools.hdf5_to_csv import hdf5_to_csv

# Ignore some BioPython warnings
warnings.filterwarnings("ignore")

# Configure logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    " %(asctime)s %(module)s:%(lineno)d %(levelname)s - %(message)s"
)
ch.setFormatter(formatter)
log.addHandler(ch)

# Constants
# TODO: Make these configurable
ESM_MODEL = "esm2_t33_650M_UR50D"
GNN_ESM_MODEL = "/home/rodrigo/repos/DeepRank-GNN-esm/paper_pretrained_models/scoring_of_docking_models/gnn_esm/treg_yfnat_b64_e20_lr0.001_foldall_esm.pth.tar"
TOKS_PER_BATCH = 4096
REPR_LAYERS = [0, 32, 33]
TRUNCATION_SEQ_LENGTH = 1022
INCLUDE = ["mean"]
NPROC = mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1
###########################################################


def setup_workspace(identificator: str) -> Path:
    """Create a temporary directory for storing intermediate files."""
    cwd = Path.cwd()
    workspace = cwd / "DEBUG" / identificator

    log.info(f"Setting up workspace - {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    # else:
    #     log.info(f"WARNING: {workspace} already exists!")
    return workspace


def pdb_to_fasta(pdb_file_path: Path, main_fasta_fh: TextIOWrapper) -> None:
    """Convert a PDB file to a FASTA file."""
    log.info(f"Reading sequence of PDB {pdb_file_path.name}")
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_file_path)

    for chain_id in ("A", "B"):
        chain = structure[0][chain_id]
        sequence = ""

        # Get the sequence of the chain
        for residue in chain:
            if residue.get_resname() == "HOH":
                continue
            sequence += residue.get_resname()

        # Write the sequence to a FASTA file
        root = re.findall(r"(.*).pdb", pdb_file_path.name)[0]

        main_fasta_fh.write(f">{root}.{chain.id}\n{sequence}\n")


def get_embedding(fasta_file: str, output_dir: Path) -> None:
    """
    Get the embedding of a protein sequence.

    Adapted from: <https://github.com/facebookresearch/esm/blob/d7b3331f41442ed4ffde70cb95bdd48cabcec2e9/scripts/extract.py#L63>
    """
    log.info("Generating embedding for protein sequence.")
    log.info("#" * 80)
    model, alphabet = pretrained.load_model_and_alphabet(ESM_MODEL)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        log.info("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(TOKS_PER_BATCH, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(  # type: ignore
        dataset,
        collate_fn=alphabet.get_batch_converter(TRUNCATION_SEQ_LENGTH),
        batch_sampler=batches,
    )
    log.info(f"Read {fasta_file} with {len(dataset)} sequences")

    output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in INCLUDE

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in REPR_LAYERS)  # type: ignore
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in REPR_LAYERS]  # type: ignore

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            log.info(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):
                output_file = output_dir / f"{label}.pt"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                truncate_len = min(TRUNCATION_SEQ_LENGTH, len(strs[i]))
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                # if "per_tok" in args.include:
                #     result["representations"] = {
                #         layer: t[i, 1 : truncate_len + 1].clone()
                #         for layer, t in representations.items()
                #     }
                if "mean" in INCLUDE:
                    result["mean_representations"] = {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                # if "bos" in args.include:
                #     result["bos_representations"] = {
                #         layer: t[i, 0].clone() for layer, t in representations.items()
                #     }
                if return_contacts:
                    result["contacts"] = contacts[i, :truncate_len, :truncate_len].clone()  # type: ignore

                torch.save(
                    result,
                    output_file,
                )
    log.info("#" * 80)


def create_graph(pdb_path: Path, workspace_path: Path) -> str:
    """Generate a graph ...?"""
    log.info(f"Generating graph, using {NPROC} processors")

    outfile = str(workspace_path / "graph.hdf5")

    with tempfile.TemporaryDirectory() as tmpdir:
        GraphHDF5(
            pdb_path=pdb_path,
            embedding_path=workspace_path,
            graph_type="residue",
            outfile=outfile,
            nproc=NPROC,
            tmpdir=tmpdir,
        )

    assert os.path.exists(outfile), f"Graph file {outfile} not found."
    log.info(f"Graph file generated: {outfile}")
    return outfile


def predict(input: str, workspace_path: Path) -> str:
    """Predict the fnat of a protein complex."""
    log.info("Predicting fnat of protein complex.")
    gnn = GINet
    target = "fnat"
    edge_attr = ["dist"]
    #
    threshold = 0.3

    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device_name}")

    node_feature = ["type", "polarity", "bsa", "charge", "embedding"]
    output = str(workspace_path / "GNN_esm_prediction.hdf5")
    # with nostdout():
    model = NeuralNet(
        input,
        gnn,
        device_name=device_name,
        edge_feature=edge_attr,
        node_feature=node_feature,
        target=target,
        pretrained_model=GNN_ESM_MODEL,
        threshold=threshold,
    )
    model.test(hdf5=output)

    output_csv = convert_to_csv(output)

    return output_csv


def convert_to_csv(hdf5_path: str) -> str:
    """Convert the hdf5 file to csv."""
    hdf5_to_csv(hdf5_path)

    csv_path = str(hdf5_path).replace(".hdf5", ".csv")

    assert os.path.exists(csv_path), f"CSV file {csv_path} not found."

    return csv_path


def parse_output(csv_output: str, workspace_path: Path) -> None:
    """Parse the csv output and return the predicted fnat."""
    _data = []
    with open(csv_output, "r") as f:
        for line in f.readlines():
            if line.startswith(","):
                # this is a header
                continue
            data = line.split(",")
            pdb_id = re.findall(r"b'(.*)'", str(data[3]))[0]
            predicted_fnat = float(data[5])
            log.info(f"Predicted fnat for {pdb_id}: {predicted_fnat:.3f}")
            _data.append([pdb_id, predicted_fnat])

    output_fname = Path(workspace_path, "output.csv")
    with open(output_fname, "w") as f:
        f.write("pdb_id,predicted_fnat\n")
        for entry in _data:
            pdb, fnat = entry
            f.write(f"{pdb},{fnat:.3f}\n")

    log.info(f"Output written to {output_fname}")


def main():
    """Main function."""

    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_file", help="Path to the PDB file.")
    args = parser.parse_args()

    pdb_file = args.pdb_file

    identificator = uuid.uuid4().hex[:6]
    workspace_path = setup_workspace(identificator)

    # Copy PDB file to workspace
    src = Path(pdb_file)
    dst = Path(workspace_path) / Path(pdb_file).name
    shutil.copy(src, dst)
    pdb_file = dst

    ## PDB to FASTA
    with open("all.fasta", "w") as f:
        pdb_to_fasta(pdb_file_path=Path(pdb_file), main_fasta_fh=f)

    ## Generate embeddings
    get_embedding(fasta_file="all.fasta", output_dir=workspace_path)

    ## Generate graphs
    graph = create_graph(pdb_path=pdb_file.parent, workspace_path=workspace_path)

    ## Predict fnat
    csv_output = predict(input=graph, workspace_path=workspace_path)

    ## Present the results
    parse_output(csv_output=csv_output, workspace_path=workspace_path)

    ## Clean workspace
    shutil.rmtree(workspace_path)


if __name__ == "__main__":
    main()
