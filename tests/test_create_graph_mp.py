import unittest
from deeprank_gnn.GraphGenMP import GraphHDF5
import os
import multiprocessing as mp

CWD = os.path.dirname(os.path.realpath(__file__))
NPROC = mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1

class TestCreateGraph(unittest.TestCase):
    def setUp(self):
        self.pdb_path = f"{CWD}/data/pdb/1ATN/"
        self.pssm_path = f"{CWD}/data/pssm/1ATN/"
        self.embeddings_path = f"{CWD}/data/embeddings/1ATN/"

    def test_create_serial_with_bio(self):
        outfile = "1ATN_residue_serial.hdf5"
        GraphHDF5(
            pdb_path=self.pdb_path,
            pssm_path=self.pssm_path,
            embedding_path=self.embeddings_path,
            graph_type="residue",
            outfile=outfile,
            nproc=1,
            biopython=False,
        )

        os.remove(outfile)

    def test_create_mp(self):
        outfile = "1ATN_residue_mp.hdf5"
        GraphHDF5(
            pdb_path=self.pdb_path,
            pssm_path=self.pssm_path,
            embedding_path=self.embeddings_path,
            graph_type="residue",
            outfile=outfile,
            nproc=NPROC,
            biopython=False,
        )

        os.remove(outfile)


if __name__ == "__main__":
    unittest.main()
