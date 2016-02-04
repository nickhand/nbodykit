from nbodykit.extensionpoints import Algorithm
from nbodykit.plugins import ListPluginsAction, add_plugin_list_argument

import numpy
import os

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


class PairCountCorrelationAlgorithm(Algorithm):
    """
    Algorithm to compute the 1d or 2d correlation function and multipoles
    via direct pair counting
    """
    plugin_name = "PairCountCorrelation"

    @classmethod
    def register(kls):
        """
        Setup the argument parser needed for initialization via the command-line
        """
        from nbodykit.extensionpoints import DataSource
        
        p = kls.parser
        p.description = "correlation function calculator via pair counting"
    
        # the positional arguments
        p.add_argument("mode", choices=["1d", "2d"],
                        help='measure the correlation function in `1d` or `2d`') 
        p.add_argument("rbins", type=binning_type, 
                        help='the string specifying the binning to use') 
        add_plugin_list_argument(p, "inputs", type=lambda l: [DataSource.fromstring(s) for s in l],
                        help='1 or 2 input `DataSource` objects to correlate; run --list-datasource for specifics')

        # add the optional arguments
        p.add_argument("--subsample", type=int, default=1,
                        help='use 1 out of every N points')
        p.add_argument("--los", choices="xyz", default='z', 
                        help="the line-of-sight: the angle `mu` is defined with respect to")
        p.add_argument("--Nmu", type=int, default=10,
                        help='if `mode == 2d`, the number of mu bins covering mu=[-1,1]')
        p.add_argument('--poles', type=lambda s: [int(i) for i in s.split()], metavar="0 2 4", default=[],
                        help='compute the multipoles for these `ell` values from xi(r,mu)')
        p.add_argument("--list-datasource", action=ListPluginsAction(DataSource),
                        help='list the help for each available `DataSource`')

    def finalize_attributes(self):
        """
        Set the communicator object of all `DataSource` plugins in `inputs`
        to the one stored in `self.comm`
        """
        for d in self.inputs:
            d.comm = self.comm
            
    def run(self):
        """
        Run the pair-count correlation function and return the result
        
        Returns
        -------
        edges : list or array_like
            the array of 1d bin edges or a list of the bin edges in each dimension
        result : dict
            dictionary holding the data results (with associated names as keys) --
            this results `corr`, `RR`, `N` + the mean bin values
        """
        from nbodykit import measurestats
    
        # check multipoles parameters
        if len(self.poles) and self.mode == '2d':
            raise ValueError("you specified multipole numbers but `mode` is `2d` -- perhaps you meant `1d`")

        # set Nmu to 1 if doing 1d
        if self.mode == "1d": self.Nmu = 0

        # do the work
        kw = {'comm':self.comm, 'subsample':self.subsample, 'Nmu':self.Nmu, 'los':self.los, 'poles':self.poles}
        pc, xi, RR = measurestats.compute_brutal_corr(self.inputs, self.rbins, **kw)

        # format the results
        if self.mode == '1d':
            if len(self.poles):
                cols = ['r'] + ['corr_%d' %l for l in self.poles] + ['RR', 'N']
                result = [pc.mean_centers] + [xi[:,i] for i in range(len(self.poles))] + [RR, pc.pair_counts]
            else:
                cols = ['r', 'corr', 'RR', 'N']
                result = [pc.mean_centers, xi, RR, pc.pair_counts]
        else:
            cols = ['r', 'mu', 'corr', 'RR', 'N']
            r, mu = pc.mean_centers
            result = [r, mu, xi, RR, pc.pair_counts]

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
        from nbodykit.extensionpoints import MeasurementStorage
        
        # only master writes
        if self.comm.rank == 0:
            
            edges, result = result
            storage = MeasurementStorage.new(self.mode, output)
        
            cols = list(result.keys())
            values = list(result.values())
            storage.write(edges, cols, values)

            



