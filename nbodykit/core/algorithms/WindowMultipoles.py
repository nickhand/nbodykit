from nbodykit.core import Algorithm, DataSource
import numpy
import os
import logging

logger = logging.getLogger('windowpaircount')

def paircount(datasource, poles, redges, comm=None, subsample=1):
    """
    Do the pair counting
    """
    from pmesh.domain import GridND
    import Corrfunc
    from mpi4py import MPI
    
    # some setup
    poles = numpy.array(poles)
    Rmax  = redges[-1]
    if comm is None: comm = MPI.COMM_WORLD
    cosmo = datasource.cosmo
        
    # log some info
    if comm.rank == 0: logger.info('Rmax = %g' %Rmax)
    
    # need to compute cartesian min/max
    pos_min = numpy.array([numpy.inf]*3)
    pos_max = numpy.array([-numpy.inf]*3)

    # read position 
    with datasource.open() as stream:
        
        [[pos1, ra1, dec1, z1]] = stream.read(['Position', 'Ra', 'Dec', 'Redshift'], full=True)
        if len(pos1):
            # global min/max of cartesian coordinates
            pos_min = numpy.minimum(pos_min, pos1.min(axis=0))
            pos_max = numpy.maximum(pos_max, pos1.max(axis=0))
    
    # gather everything to root
    pos_min = comm.gather(pos_min)
    pos_max = comm.gather(pos_max)
    
    if comm.rank == 0:
        
        # find the global coordinate minimum and maximum
        pos_min   = numpy.amin(pos_min, axis=0)
        pos_max   = numpy.amax(pos_max, axis=0)
        
        # used to center the data in the first cartesian quadrant
        BoxSize = abs(pos_max - pos_min)
    else:
        BoxSize = None
     
    ra1  *= 180./numpy.pi 
    dec1 *= 180./numpy.pi 
    ang_coords1 = numpy.vstack([ra1, dec1, cosmo.comoving_distance(z1)]).T
       
    BoxSize = comm.bcast(BoxSize)
    if comm.rank == 0: logger.info("BoxSize = %s" %str(BoxSize))
    
    # determine processor division for domain decomposition
    for Nx in range(int(comm.size**0.3333) + 1, 0, -1):
        if comm.size % Nx == 0: break
    else:
        Nx = 1
    for Ny in range(int(comm.size**0.5) + 1, 0, -1):
        if (comm.size // Nx) % Ny == 0: break
    else:
        Ny = 1
    Nz = comm.size // Nx // Ny
    Nproc = [Nx, Ny, Nz]
    
    if comm.rank == 0: logger.info('Nproc = %s' %str(Nproc))
    
    pos1 = pos1[comm.rank * subsample // comm.size ::subsample]
    ang_coords1 = ang_coords1[comm.rank * subsample // comm.size ::subsample]
    N1   = comm.allreduce(len(ang_coords1))
    
    # read position for field #2
    pos2 = pos1
    ang_coords2 = ang_coords1
    N2 = N1
        
    # domain decomposition
    grid = [numpy.linspace(0, BoxSize[i], Nproc[i]+1, endpoint=True) for i in range(3)]
    domain = GridND(grid, comm=comm)
    
    # exchange field #1 positions
    layout = domain.decompose(pos1, smoothing=0)
    pos1 = layout.exchange(pos1)
    if comm.rank == 0: logger.info('exchange pos1')

    # exchange field #2 positions
    if Rmax > BoxSize.min() * 0.25:
        ang_coords2 = numpy.concatenate(comm.allgather(ang_coords2), axis=0)
    else:
        layout = domain.decompose(pos2, smoothing=Rmax)
        ang_coords2 = layout.exchange(ang_coords2)
    N2   = comm.allreduce(len(ang_coords2))
    if comm.rank == 0: logger.info('exchange pos2')

    # log the sizes of the trees
    logger.info('rank %d correlating %d x %d' %(comm.rank, len(ang_coords1), len(ang_coords2)))
    if comm.rank == 0: logger.info('all correlating %d x %d' %(N1, N2))

    # do the pair counting
    kws                     = {}
    kws['is_comoving_dist'] = True
    kws['output_rpavg']     = True
    kws['RA2']              = ang_coords2[:,0]
    kws['DEC2']             = ang_coords2[:,1]
    kws['CZ2']              = ang_coords2[:,2]
    kws['verbose']          = True
    results = Corrfunc.mocks.DDrppi_mocks(0, 1, 1, Rmax, redges, ang_coords1[:,0], ang_coords1[:,1], ang_coords1[:,2], **kws)
    logger.info('...rank %d done correlating' %(comm.rank))
    
    # do the sum across ranks
    results['rpavg'][:] = comm.allreduce(results['npairs']*results['rpavg'])
    results['npairs'][:] = comm.allreduce(results['npairs'])
    
    idx = results['npairs'] > 0.
    results['rpavg'][idx] /= results['npairs'][idx]

    return results

def binning_type(s):
    """
    Type conversion for use on the command-line that converts 
    a string to an array of bin edges
    """
    if os.path.isfile(s):
        return numpy.loadtxt(s)
    else:
        supported = ["`linspace: min max Nbins`", "`logspace: logmin logmax Nbins`"]
        try:
            f, params = s.split(':')
            params = list(map(float, params.split()))
            params[-1] = int(params[-1]) + 1

            if not hasattr(numpy, f): raise Exception
            if len(params) != 3: raise Exception

            return getattr(numpy, f)(*params)
        except:
            raise TypeError("supported binning format: [ %s ]" %", ".join(supported))


class WindowMultipolesAlgorithm(Algorithm):
    """
    Algorithm to compute window function multipoles via pair counting
    """
    plugin_name = "WindowMultipoles"

    def __init__(self, rbins, field, poles, subsample=1):
                    
        self.rbins     = rbins
        self.field     = field
        self.poles     = poles
        self.subsample = subsample
        
    @classmethod
    def fill_schema(cls):  
        s = cls.schema
        s.description = "compute the window function multipoles via pair counting"
    
        # the positional arguments
        s.add_argument("rbins", type=binning_type, 
            help='the string specifying the binning to use') 
        s.add_argument("field", type=DataSource.from_config, 
            help='the first `DataSource` of objects to correlate; '
                 'run `nbkit.py --list-datasources` for all options')
        s.add_argument("subsample", type=int, help='use 1 out of every N points')
        s.add_argument('poles', nargs='*', type=int,
            help='compute the multipoles for these `ell` values from xi(r,mu)')
   
    def run(self):
        """
        Run the pair-count algorithm
        
        Returns
        -------
        edges : list or array_like
            the array of 1d bin edges or a list of the bin edges in each dimension
        result : dict
            dictionary holding the data results (with associated names as keys) --
            this results `corr`, `RR`, `N` + the mean bin values
        """
        # do the work
        kw = {'comm':self.comm, 'subsample':self.subsample}
        return paircount(self.field, self.poles, self.rbins, **kw)

        
    def save(self, output, result):
        """
        Save the result returned by `run()` to the filename specified by `output`
        
        Parameters
        ----------
        output : str
            the string specifying the file to save
        result : tuple
            the structured array results returned by :func:`Corrfunc.mocks.DDrppi_mocks`
        """
        if self.comm.rank == 0:    
            numpy.savez(output, result)


            



