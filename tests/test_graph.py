
import unittest
import h5py
import os

from deeprank_gnn.Graph import Graph

CWD = os.path.dirname(os.path.realpath(__file__))

class TestGraph(unittest.TestCase):

    def setUp(self):
        self.graph = Graph()
        self.graph.h52nx(f'{CWD}/data/hdf5/1ATN_residue.hdf5', '1ATN_1w')
        self.graph.pdb = f'{CWD}/data/pdb/1ATN/1ATN_1w.pdb'
        self.ref = f'{CWD}/data/pdb/1ATN/1ATN_2w.pdb'

    def test_score(self):
        self.graph.get_score(self.ref)

    def test_nx2h5(self):
        f5 = h5py.File('test_graph.hdf5', 'w')
        self.graph.nx2h5(f5)

        os.remove('test_graph.hdf5')

    def test_plot_2d(self):
        self.graph.plotly_2d('1ATN', disable_plot=True)

    def test_plot_3d(self):
        self.graph.plotly_3d('1ATN', disable_plot=True)


if __name__ == "__main__":
    unittest.main()
