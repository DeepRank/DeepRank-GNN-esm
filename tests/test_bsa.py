import unittest
from deeprank_gnn.DataSet import HDF5DataSet, DivideDataSet, PreCluster
from deeprank_gnn.tools.BSA import BSA
import os

CWD = os.path.dirname(os.path.realpath(__file__))

class TestBSA(unittest.TestCase):

    def setUp(self):
        self.bsa = BSA(f'{CWD}/data/pdb/1ATN/1ATN_1w.pdb')

    def test_structure(self):
        self.bsa.get_structure()
        self.bsa.get_contact_residue_sasa()


if __name__ == "__main__":
    unittest.main()
