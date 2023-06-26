import unittest
from deeprank_gnn.DataSet import HDF5DataSet, DivideDataSet, PreCluster
import os

CWD = os.path.dirname(os.path.realpath(__file__))

class TestDataSet(unittest.TestCase):
    def setUp(self):
        self.database = f'{CWD}/data/hdf5/1ATN_residue.hdf5'

    def test_dataset(self):
        _ = HDF5DataSet(
            database=self.database,
            node_feature=["type", "polarity", "bsa", "embedding"], # type: ignore
            edge_feature=["dist"],
            target="fnat",
            index=None,
        )

    def test_dataset_filter(self):
        _ = HDF5DataSet(
            database=self.database,
            node_feature=["type", "polarity", "bsa", "embedding"], # type: ignore
            edge_feature=["dist"],
            target="fnat",
            index=None,
            dict_filter={"fnat": "<10"},
        )


if __name__ == "__main__":
    unittest.main()
