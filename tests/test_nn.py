import unittest

from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.ginet import GINet
import torch
import os

CWD = os.path.dirname(os.path.realpath(__file__))

def _model_base_test(database, model, task='reg', target='test_target', plot=False):

    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    NN = NeuralNet(database, model,
                   node_feature=['type', 'polarity', 'bsa', 'charge',
                                 'embedding'],
                   edge_feature=['dist'],
                   target=target,
                   index=None,
                   task=task,
                   batch_size=64,
                   num_workers=4,
                   percent=[0.8, 0.2],
                   device_name=device_name,
                   )

    NN.train(nepoch=2, validate=True)

    NN.save_model('test.pth.tar')

    _ = NeuralNet(database, model,
                       pretrained_model='test.pth.tar',
                       device_name=device_name,
                       )

    if plot:
        NN.plot_scatter()
        NN.plot_loss()
        NN.plot_acc()
        NN.plot_hit_rate()


class TestNeuralNet(unittest.TestCase):

    def setUp(self):
        self.database = f'{CWD}/data/hdf5/1ATN_residue.hdf5'

    def test_ginet(self):
        _model_base_test(self.database,
                         GINet)

        generated_files = ['test.pth.tar', "train_data.hdf5", "treg_ytest_target_b64_e2_lr0.01.pth.tar"]
        for file in generated_files:
            os.remove(file)

    def test_ginet_class(self):
        _model_base_test(self.database,
                         GINet,
                         target='test_target_class',
                         task='class')

        generated_files = ['test.pth.tar', "train_data.hdf5", "tclass_ytest_target_class_b64_e2_lr0.01.pth.tar"]
        for file in generated_files:
            os.remove(file)



if __name__ == "__main__":
    unittest.main()
