from deeprank_gnn.GraphGenMP import GraphHDF5

pdb_path = './data/pdb/1ATN/'
pssm_path = './data/pssm/1ATN/'
embedding_path = './data/embedding/1ATN/'
nproc=20 


GraphHDF5(pdb_path=pdb_path, pssm_path=None, embedding_path=embedding_path,
          graph_type='residue', outfile='1ATN_residue.hdf5',
          nproc=nproc, tmpdir='./tmpdir')
