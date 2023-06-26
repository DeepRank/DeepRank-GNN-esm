import os
import numpy as np
from pdb2sql.interface import interface

try:
    import freesasa

except ImportError:
    print('Freesasa not found')


class BSA(object):

    def __init__(self, pdb_data, sqldb=None, chainA='A', chainB='B'):
        '''Compute the burried surface area feature

        Freesasa is required for this feature.

        https://freesasa.github.io

        >>> wget http://freesasa.github.io/freesasa-2.0.2.tar.gz
        >>> tar -xvf freesasa-2.0.3.tar.gz
        >>> cd freesasa
        >>> ./configure CFLAGS=-fPIC (--prefix /home/<user>/)
        >>> make
        >>> make install

        Since release 2.0.3 the python bindings are separate module
        >>> pip install freesasa

        Args :
            pdb_data (list(byte) or str): pdb data or filename of the pdb
            sqldb (pdb2sql.interface instance or None, optional) if the sqldb is None the sqldb will be created
            chainA (str, optional): name of the first chain
            chainB (str, optional): name of the second chain

        Example :

        >>> bsa = BSA('1AK4.pdb')
        >>> bsa.get_structure()
        >>> bsa.get_contact_residue_sasa()
        >>> bsa.sql.close()

        '''

        self.pdb_data = pdb_data
        if sqldb is None:
            self.sql = interface(pdb_data)
        else:
            self.sql = sqldb
        self.chains_label = [chainA, chainB]

        freesasa.setVerbosity(freesasa.nowarnings)

    def get_structure(self):
        """Get the pdb structure of the molecule."""

        # we can have a str or a list of bytes as input
        if isinstance(self.pdb_data, str):
            self.complex = freesasa.Structure(self.pdb_data)
        else:
            self.complex = freesasa.Structure()
            atomdata = self.sql.get(
                'name,resName,resSeq,chainID,x,y,z')
            for atomName, residueName, residueNumber, chainLabel, x, y, z in atomdata:
                atomName = '{:>2}'.format(atomName[0])
                self.complex.addAtom(
                    atomName, residueName, residueNumber, chainLabel, x, y, z)
        self.result_complex = freesasa.calc(self.complex)

        self.chains = {}
        self.result_chains = {}
        for label in self.chains_label:
            self.chains[label] = freesasa.Structure()
            atomdata = self.sql.get(
                'name,resName,resSeq,chainID,x,y,z', chainID=label)
            for atomName, residueName, residueNumber, chainLabel, x, y, z in atomdata:
                atomName = '{:>2}'.format(atomName[0])
                self.chains[label].addAtom(
                    atomName, residueName, residueNumber, chainLabel, x, y, z)
            self.result_chains[label] = freesasa.calc(
                self.chains[label])

    def get_contact_residue_sasa(self, cutoff=8.5):
        """Compute the feature value."""

        self.bsa_data = {}
        self.bsa_data_xyz = {}

        res = self.sql.get_contact_residues(cutoff=cutoff)
        keys = list(res.keys())
        res = res[keys[0]]+res[keys[1]]

        for r in res:

            # define the selection string and the bsa for the complex
            select_str = ('res, (resi %d) and (chain %s)' %
                          (r[1], r[0]),)
            asa_complex = freesasa.selectArea(
                select_str, self.complex, self.result_complex)['res']

            # define the selection string and the bsa for the isolated
            select_str = ('res, resi %d' % r[1],)
            asa_unbound = freesasa.selectArea(
                select_str, self.chains[r[0]], self.result_chains[r[0]])['res']

            # define the bsa
            bsa = asa_unbound-asa_complex

            # define the xyz key : (chain,x,y,z)
            chain = {'A': 0, 'B': 1}[r[0]]
            xyz = np.mean(self.sql.get(
                'x,y,z', resSeq=r[1], chainID=r[0]), 0)
            xyzkey = tuple([chain]+xyz.tolist())

            # put the data in dict
            self.bsa_data[r] = [bsa]
