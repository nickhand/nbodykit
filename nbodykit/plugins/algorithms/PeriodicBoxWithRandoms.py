from nbodykit.extensionpoints import Algorithm, DataSource
from nbodykit.periodic_fkp import FKPCatalog
import numpy
import logging

class FFTPowerWithRandomsAlgorithm(Algorithm):
    """
    Algorithm similar to `FFTPower`, but using randoms
    to define the mean density
    """
    plugin_name = "FFTPowerWithRandoms"
    logger = logging.getLogger(plugin_name)

    def __init__(self, data, randoms, BoxSize, Nmesh,
                    paintbrush='cic',
                    los='z', 
                    Nmu=5, 
                    dk=None, 
                    kmin=0.):
                           
        # initialize the FKP catalog (unopened)
        self.catalog = FKPCatalog(self.data, self.randoms, self.BoxSize)
        
    @classmethod
    def register(cls):
        
        s = cls.schema
        s.description = "compute the power spectrum, using randoms"

        # the required arguments
        s.add_argument("data", type=DataSource.from_config,
            help="DataSource representing the `data` catalog")
        s.add_argument("randoms", type=DataSource.from_config,
            help="DataSource representing the `randoms` catalog")
        s.add_argument("BoxSize", type=DataSource.BoxSizeParser,
            help="the size of the box")
        s.add_argument("Nmesh", type=int,
            help='the number of cells in the gridded mesh (per axis)')

        # the optional arguments
        s.add_argument("los", type=str, choices="xyz",
            help="the line-of-sight direction -- the angle `mu` is defined with respect to")
        s.add_argument("Nmu", type=int,
            help='the number of mu bins to use from mu=[0,1]; if `mode = 1d`, then `Nmu` is set to 1' )
        
        s.add_argument('paintbrush', type=lambda x: x.lower(), choices=['cic', 'tsc'],
            help='the density assignment kernel to use when painting; '
                 'CIC (2nd order) or TSC (3rd order)')
        s.add_argument("dk", type=float,
            help='the spacing of k bins to use; if not provided, '
                 'the fundamental mode of the box is used')
        s.add_argument("kmin", type=float,
            help='the edge of the first `k` bin to use; default is 0')

                                
    def run(self):
        """
        Run the algorithm, which computes and returns the power spectrum
        """
        from nbodykit import measurestats
        from nbodykit.extensionpoints import Painter, Transfer
        from pmesh.particlemesh import ParticleMesh
            
        if self.comm.rank == 0: self.logger.info('importing done')

        # the painting kernel transfer
        if self.paintbrush == 'cic':
            transfer = [Transfer.create('AnisotropicCIC')]
        elif self.paintbrush == 'tsc':
            transfer = [Transfer.create('AnisotropicTSC')]
        else:
            raise ValueError("valid `paintbrush` values are: ['cic', 'tsc']")

        #transfer = [Transfer.create(x) for x in ['NormalizeDC', 'RemoveDC']] + transfer

        # load the data/randoms and setup boxsize, etc
        with self.catalog:
                
            # initialize the particle mesh
            pm = ParticleMesh(self.catalog.BoxSize, self.Nmesh, paintbrush=self.paintbrush, dtype='f4', comm=self.comm)
        
            # do the FKP painting
            meta = self.catalog.paint(pm)
        
        # compute the monopole, A0(k), and save
        pm.r2c()
        pm.transfer(transfer)
        
        # reuse the memory in c1.real for the 3d power spectrum
        p3d = pm.complex
    
        # calculate the 3d power spectrum, islab by islab to save memory
        for islab in range(len(pm.complex)):
            p3d[islab, ...] = pm.complex[islab]*pm.complex[islab].conj()

        # the complex field is dimensionless; power is L^3
        # ref to http://icc.dur.ac.uk/~tt/Lectures/UA/L4/cosmology.pdf
        p3d[...] *= pm.BoxSize.prod()

        # the 3D k
        k3d = pm.k

        # binning in k out to the minimum nyquist frequency
        # (accounting for possibly anisotropic box)
        dk = 2*numpy.pi/pm.BoxSize.min() if self.dk is None else self.dk
        kedges = numpy.arange(self.kmin, numpy.pi*pm.Nmesh/pm.BoxSize.max() + dk/2, dk)

        # project on to 2d basis
        muedges = numpy.linspace(0, 1, self.Nmu+1, endpoint=True)
        edges = [kedges, muedges]
        
        # result is (k, mu, power, modes)
        result, _ = measurestats.project_to_basis(pm.comm, k3d, p3d, edges, symmetric=True)

        # compute the metadata to return
        Lx, Ly, Lz = pm.BoxSize
        meta.update({'Lx':Lx, 'Ly':Ly, 'Lz':Lz, 'volume':Lx*Ly*Lz})

        # return all the necessary results
        return edges, result, meta

    def save(self, output, result):
        """
        Save the power spectrum results to the specified output file

        Parameters
        ----------
        output : str
            the string specifying the file to save
        result : tuple
            the tuple returned by `run()` -- first argument specifies the bin
            edges and the second is a dictionary holding the data results
        """
        from nbodykit.storage import MeasurementStorage
        
        # only the master rank writes        
        if self.comm.rank == 0:
            
            edges, result, meta = result
            cols = ['k', 'mu', 'power', 'modes']
                
            # write binned statistic
            self.logger.info('measurement done; saving result to %s' %output)
            storage = MeasurementStorage.create('2d', output)
            storage.write(edges, cols, result, **meta)
            


