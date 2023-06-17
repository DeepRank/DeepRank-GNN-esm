import unittest

from deeprank_gnn.NeuralNet import NeuralNet
from deeprank_gnn.ginet import GINet



def _model_base_test(database, model, task='reg', target='fnat', plot=False):

    NN = NeuralNet(database, model,
                   node_feature=['type', 'polarity', 'bsa',
                                 'depth', 'hse', 'ic', 'pssm',
                                 'embedding'],
                   edge_feature=['dist'],
                   target=target,
                   index=None,
                   task=task,
                   batch_size=64,
                   device_name='cuda:0',
                   num_workers=4,
                   percent=[0.8, 0.2])

    NN.train(nepoch=5, validate=True)

    NN.save_model('test.pth.tar')

    NN_cpy = NeuralNet(database, model,
                       pretrained_model='test.pth.tar')

    if plot:
        NN.plot_scatter()
        NN.plot_loss()
        NN.plot_acc()
        NN.plot_hit_rate()


class TestNeuralNet(unittest.TestCase):

    def setUp(self):
        self.database = '../example/1AK4_residue.hdf5'

    def test_ginet(self):
        _model_base_test(self.database, GINet, plot=True)

    def test_ginet_class(self):
        _model_base_test(self.database, GINet,
                         task='class', target='binclass')





if __name__ == "__main__":
    unittest.main()
