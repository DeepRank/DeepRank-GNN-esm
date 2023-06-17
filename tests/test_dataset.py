import unittest
from deeprank_gnn.DataSet import HDF5DataSet, DivideDataSet, PreCluster


class TestDataSet(unittest.TestCase):
    def setUp(self):
        self.database = "hdf5/1ATN_residue.hdf5"

    def test_dataset(self):
        dataset = HDF5DataSet(
            database=self.database,
            node_feature=["type", "polarity", "bsa", "embedding"],
            edge_feature=["dist"],
            target="fnat",
            index=None,
        )

    def test_dataset_filter(self):
        dataset = HDF5DataSet(
            database=self.database,
            node_feature=["type", "polarity", "bsa", "embedding"],
            edge_feature=["dist"],
            target="fnat",
            index=None,
            dict_filter={"fnat": "<10"},
        )


if __name__ == "__main__":
    unittest.main()
