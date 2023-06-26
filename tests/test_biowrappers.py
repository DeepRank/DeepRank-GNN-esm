
import unittest
import h5py
import os

from deeprank_gnn.tools.BioWrappers import get_bio_model, get_depth_res, get_depth_contact_res, get_hse

CWD = os.path.dirname(os.path.realpath(__file__))


class TestBioWrappers(unittest.TestCase):

    def setUp(self):
        self.pdb = f'{CWD}/data/pdb/1ATN/1ATN_1w.pdb'

    def test_hse(self):
        model = get_bio_model(self.pdb)
        _ = get_hse(model)

    @unittest.expectedFailure
    def test_depth_res(self):
        model = get_bio_model(self.pdb)
        _ = get_depth_res(model)

    @unittest.expectedFailure
    def test_depth_contact_res(self):
        model = get_bio_model(self.pdb)
        _ = get_depth_contact_res(model, [('A', '1')])


if __name__ == "__main__":
    unittest.main()
