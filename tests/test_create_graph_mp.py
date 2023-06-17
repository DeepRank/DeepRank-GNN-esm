import unittest
from deeprank_gnn.GraphGenMP import GraphHDF5


class TestCreateGraph(unittest.TestCase):
    def setUp(self):
        self.pdb_path = "data/pdb/1ATN/"
        self.pssm_path = "data/pssm/1ATN/"
        self.ref = "data/ref/1ATN/"
        self.embeddings_path = "data/embeddings/1ATN/"

    def test_create_serial_with_bio(self):
        GraphHDF5(
            pdb_path=self.pdb_path,
            ref_path=self.ref,
            pssm_path=self.pssm_path,
            embedding_path=self.embeddings_path,
            graph_type="residue",
            outfile="1ATN_residue.hdf5",
            nproc=1,
            tmpdir="./tmpdir",
            biopython=True,
        )

    def test_create_serial(self):
        GraphHDF5(
            pdb_path=self.pdb_path,
            ref_path=self.ref,
            pssm_path=self.pssm_path,
            embedding_path=self.embeddings_path,
            graph_type="residue",
            outfile="1ATN_residue.hdf5",
            nproc=1,
            tmpdir="./tmpdir2",
            biopython=False,
        )

    def test_create_mp(self):
        GraphHDF5(
            pdb_path=self.pdb_path,
            ref_path=self.ref,
            pssm_path=self.pssm_path,
            embedding_path=self.embeddings_path,
            graph_type="residue",
            outfile="1ATN_residue.hdf5",
            nproc=2,
            tmpdir="./tmpdir3",
            biopython=False,
        )


if __name__ == "__main__":
    unittest.main()
