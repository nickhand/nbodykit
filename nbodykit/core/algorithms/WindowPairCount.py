from nbodykit.core import Algorithm, DataSource
import numpy
import os
import logging

logger = logging.getLogger('windowpaircount')

def paircount(datasources, redges, Nmu=0, comm=None, subsample=1, los='z', poles=[], flatsky=False):
    """
    Do the pair counting
    """
    from pmesh.domain import GridND
    from kdcount import correlate
    from mpi4py import MPI
    
    # some setup
    if los not in "xyz": raise ValueError("`los` must be `x`, `y`, or `z`")
    los   = "xyz".index(los)
    poles = numpy.array(poles)
    Rmax  = redges[-1]
    if comm is None: comm = MPI.COMM_WORLD
        
    # log some info
    if comm.rank == 0: logger.info('Rmax = %g' %Rmax)
    
    # need to compute cartesian min/max
    pos_min = numpy.array([numpy.inf]*3)
    pos_max = numpy.array([-numpy.inf]*3)

    # read position for field #1 
    with datasources[0].open() as stream:
        
        [[pos1]] = stream.read(['Position'], full=True)
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
        
    BoxSize = comm.bcast(BoxSize)
    if comm.rank == 0: logger.info("BoxSize = %s" %str(BoxSize))
    
    # determine processor division for domain decomposition
    #if Rmax > BoxSize.min() * 0.25:
    Nproc = [1, 1, 1]
    idx = numpy.argmax(BoxSize)
    Nproc[idx] = comm.size
    # else:
    #     for Nx in range(int(comm.size**0.3333) + 1, 0, -1):
    #         if comm.size % Nx == 0: break
    #     else:
    #         Nx = 1
    #     for Ny in range(int(comm.size**0.5) + 1, 0, -1):
    #         if (comm.size // Nx) % Ny == 0: break
    #     else:
    #         Ny = 1
    #     Nz = comm.size // Nx // Ny
    #     Nproc = [Nx, Ny, Nz]
    
    if comm.rank == 0: logger.info('Nproc = %s' %str(Nproc))
    
    pos1 = pos1[comm.rank * subsample // comm.size ::subsample]
    N1 = comm.allreduce(len(pos1))
    
    # read position for field #2
    if len(datasources) > 1:
        with datasources[1].open() as stream:
            [[pos2]] = stream.read(['Position'], full=True)
        pos2 = pos2[comm.rank * subsample // comm.size ::subsample]
        N2 = comm.allreduce(len(pos2))
    else:
        pos2 = pos1
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
        pos2 = numpy.concatenate(comm.allgather(pos2), axis=0)
    else:
        layout = domain.decompose(pos2, smoothing=Rmax)
        pos2 = layout.exchange(pos2)
    if comm.rank == 0: logger.info('exchange pos2')

    # initialize the trees to hold the field points
    tree1 = correlate.points(pos1)
    tree2 = correlate.points(pos2)

    # log the sizes of the trees
    logger.info('rank %d correlating %d x %d' %(comm.rank, len(tree1), len(tree2)))
    if comm.rank == 0: logger.info('all correlating %d x %d' %(N1, N2))

    # use multipole binning
    if len(poles):
        if flatsky:
            bins = correlate.FlatSkyMultipoleBinning(redges, poles, los)
        else:
            bins = correlate.MultipoleBinning(redges, poles)
    # use (R, mu) binning
    elif Nmu > 0:
        if flatsky:
            bins = correlate.FlatSkyBinning(redges, Nmu, los)
        else:
            bins = correlate.RmuBinning(redges, Nmu, 0.)
    # use R binning
    else:
        bins = correlate.RBinning(redges)

    # do the pair counting
    # have to set usefast = False to get mean centers, or exception thrown
    pc = correlate.paircount(tree2, tree1, bins, np=0, usefast=False, compute_mean_coords=True)
    logger.info('...rank %d done correlating' %(comm.rank))
    
    pc.sum1[:] = comm.allreduce(pc.sum1)
    
    # get the mean bin values, reducing from all ranks
    pc.pair_counts[:] = comm.allreduce(pc.pair_counts)
    with numpy.errstate(invalid='ignore'):
        if bins.Ndim > 1:
            for i in range(bins.Ndim):
                pc.mean_centers[i][:] = comm.allreduce(pc.mean_centers_sum[i]) / pc.pair_counts
        else:
            pc.mean_centers[:] = comm.allreduce(pc.mean_centers_sum[0]) / pc.pair_counts
    
    # return the correlation and the pair count object
    toret = pc.sum1
    if len(poles):
        toret = toret.T # makes ell the second axis 

    return pc, toret

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


class WindowPairCountAlgorithm(Algorithm):
    """
    Algorithm to compute pair counting of objects using :mod:`kdcount`
    """
    plugin_name = "WindowPairCount"

    def __init__(self, mode, rbins, field, other=None, subsample=1, 
                    los='z', flatsky=False, Nmu=10, poles=[]):
                    
        # positional arguments
        self.mode      = mode
        self.rbins     = rbins
        self.field     = field
        
        # keyword arguments
        self.other     = other
        self.subsample = subsample
        self.los       = los
        self.Nmu       = Nmu
        self.poles     = poles
        self.flatsky   = flatsky
        
        # construct the input fields list
        self.inputs = [self.field]
        if self.other is not None:
            self.inputs.append(self.other)
        
    @classmethod
    def fill_schema(cls):  
        s = cls.schema
        s.description = "pair counting calculator"
    
        # the positional arguments
        s.add_argument("mode", type=str, choices=["1d", "2d"],
            help='measure the correlation function in `1d` or `2d`') 
        s.add_argument("rbins", type=binning_type, 
            help='the string specifying the binning to use') 
        s.add_argument("field", type=DataSource.from_config, 
            help='the first `DataSource` of objects to correlate; '
                 'run `nbkit.py --list-datasources` for all options')
        s.add_argument("other", type=DataSource.from_config, 
            help='the other `DataSource` of objects to cross-correlate with; '
                 'run `nbkit.py --list-datasources` for all options')
        s.add_argument("subsample", type=int, help='use 1 out of every N points')
        s.add_argument("los", choices="xyz",
            help="the line-of-sight: the angle `mu` is defined with respect to when `flatsky` is True")
        s.add_argument("Nmu", type=int,
            help='if `mode == 2d`, the number of mu bins covering mu=[-1,1]')
        s.add_argument('poles', nargs='*', type=int,
            help='compute the multipoles for these `ell` values from xi(r,mu)')
        s.add_argument('flatsky', type=bool,
            help='if True, use `los` to define the line-of-sight')
   
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
        # check multipoles parameters
        if len(self.poles) and self.mode == '2d':
            raise ValueError("you specified multipole numbers but `mode` is `2d` -- perhaps you meant `1d`")

        # set Nmu to 1 if doing 1d
        if self.mode == "1d": self.Nmu = 0

        # do the work
        kw = {'comm':self.comm, 'subsample':self.subsample, 'Nmu':self.Nmu, 'los':self.los, 'poles':self.poles, 'flatsky':self.flatsky}
        pc, RR = paircount(self.inputs, self.rbins, **kw)

        
        # format the results
        if self.mode == '1d':
            if len(self.poles):
                cols = ['r'] + ['npairs_%d' %l for l in self.poles] + ['N']
                result = [pc.mean_centers] + [RR[:,i] for i in range(len(self.poles))] + [pc.pair_counts]
            else:
                cols = ['r', 'npairs', 'N']
                result = [pc.mean_centers, RR, pc.pair_counts]
        else:
            cols = ['r', 'mu', 'npairs', 'N']
            r, mu = pc.mean_centers
            result = [r, mu, RR, pc.pair_counts]

        return pc.edges, dict(zip(cols, result))
        
    def save(self, output, result):
        """
        Save the result returned by `run()` to the filename specified by `output`
        
        Parameters
        ----------
        output : str
            the string specifying the file to save
        result : tuple
            the tuple returned by `run()` -- first argument specifies the bin
            edges and the second is a dictionary holding the data results
        """
        from nbodykit.storage import MeasurementStorage
        
        # only master writes
        if self.comm.rank == 0:
            
            edges, result = result
            storage = MeasurementStorage.create(self.mode, output)
        
            cols = list(result.keys())
            values = list(result.values())
            storage.write(edges, cols, values)

            



