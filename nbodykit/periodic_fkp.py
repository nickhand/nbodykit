import numpy
import logging
import os
from contextlib import contextmanager
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from nbodykit.extensionpoints import DataSource, Painter, algorithms
from nbodykit.distributedarray import GatherArray

logger = logging.getLogger('PeriodicFKPCatalog')

class FKPCatalog(object):
    """
    A `DataSource` representing a catalog of tracer objects, 
    designed to be used in analysis similar to that first outlined 
    by Feldman, Kaiser, and Peacock (FKP) 1994 (astro-ph/9304022)
    
    In particular, the `FKPCatalog` uses a catalog of random objects
    to define the mean density of the survey, in addition to the catalog
    of data objects
    
    
    Attributes
    ----------
    data: DataSource
        a `DataSource` that returns the position, weight, etc of the 
        true tracer objects, whose intrinsic clustering is non-zero
    randoms: DataSource
        a `DataSource` that returns the position, weight, etc 
        of a catalog of objects generated randomly to match the
        survey geometry and whose instrinsic clustering is zero
    BoxSize:
        the size of the cartesian box -- the Cartesian coordinates
        of the input objects are computed using the input cosmology,
        and then placed into the box
    mean_coordinate_offset: 
        the average coordinate value in each dimension -- this offset
        is used to return cartesian coordinates translated into the
        domain of [-BoxSize/2, BoxSize/2]
    """
    def __init__(self,  data, 
                        randoms, 
                        BoxSize):

        # set the cosmology
        self.cosmo = data.cosmo
        if self.cosmo is None:
            raise ValueError("FKPCatalog requires a cosmology")
        if data.cosmo is not randoms.cosmo:
            raise ValueError("mismatch between cosmology instances of `data` and `randoms` in `FKPCatalog`")
            
        # set the comm
        self.comm = data.comm
        if data.comm is not randoms.comm:
            raise ValueError("mismatch between communicators of `data` and `randoms` in `FKPCatalog`")
        
        # data and randoms datasources
        self.data    = data
        self.randoms = randoms
        self.BoxSize = BoxSize
 
        # default painter
        self.painter = Painter.create('DefaultPainter')
        
    @property
    def data(self):
        """
        Update the ``data`` attribute, keeping track of the total number 
        of objects
        """
        try:
            return self._data
        except:
            cls = self.__class__.__name__
            raise AttributeError("'%s' object has no attribute 'data'" %cls)
        
    @data.setter
    def data(self, val):
        """
        Set the data
        """
        if not hasattr(self, '_data'):
            self._data = val
        else:
            # open the new data automatically
            if not self.closed:
                
                # close the old data
                if hasattr(self, 'data_stream'):
                    self.data_stream.close()
                    del self.data_stream
                
                # set and open the new data
                self._data = val
                defaults = {'Weight':1.}
                self.data_stream = self.data.open()
                self.verify_data_size() # verify the size
            
            else:
                self._data = val
    
    def verify_data_size(self):
        """
        Verify the size of the data, setting it if need be
        """            
        # make sure the size is set properly
        try:
            size = self.data.size
        except:
            if hasattr(self, 'data_stream') and not self.data_stream.closed:            
                # compute the total number
                for [Position] in self.data_stream.read(['Position'], full=False):
                    continue
                self.data.size = self.data_stream.nread
                logger.debug("setting `data` size to %d" %self.data.size)
                        
    @property
    def closed(self):
        """
        Return `True` if the catalog has been setup and the
        data and random streams are open
        """
        if not hasattr(self, 'data_stream'):
            return True
        elif self.data_stream.closed:
            return True
            
        if not hasattr(self, 'randoms_stream'):
            return True
        elif self.randoms_stream.closed:
            return True
        
        return False
    
    def open(self):
        """
        Open the catalog by defining the Cartesian box
        and opening the `data` and `randoms` streams
        """
        # open the streams
        defaults = {'Redshift':-1., 'Nbar':-1., 'Weight':1.}
        self.data_stream = self.data.open(defaults=defaults)
        self.randoms_stream = self.randoms.open(defaults=defaults)
                
        # verify data size
        self.verify_data_size()
    
        # loop over the randoms to get total
        for [pos] in self.randoms_stream.read(['Position'], full=False):
            pass

        # set the size, if not set already
        if not hasattr(self.randoms, 'size'):
            self.randoms.size = self.randoms_stream.nread
                            
    def close(self):
        """
        Close the FKPCatalog by close the `data` and `randoms` streams
        """
        if hasattr(self, 'data_stream'):
            self.data_stream.close()
            del self.data_stream
        if hasattr(self, 'randoms_stream'):
            self.randoms_stream.close()
            del self.randoms_stream
                
    def __enter__ (self):
        if self.closed:
            self.open()
        
    def __exit__ (self, exc_type, exc_value, traceback):
        self.close()
                    
    def read(self, name, columns, full=False):
        """
        Read data from `stream`, which is specified by the `name` argument
        """   
        # check valid columns
        valid = ['Position', 'Weight']
        if any(col not in valid for col in columns):
            raise DataSource.MissingColumn("valid `columns` to read from FKPCatalog: %s" %str(valid))
                             
        if name == 'data':
            stream = self.data_stream
        elif name == 'randoms':
            stream = self.randoms_stream
        else:
            raise ValueError("stream name for FKPCatalog must be 'data' or 'randoms'")
    
        # read position, redshift, and weights from the stream
        columns0 = ['Position', 'Weight']
        for [pos, weight] in stream.read(columns0, full=full):
                    
            P = {}
            P['Position'] = pos
            P['Weight']   = weight
        
            yield [P[key] for key in columns]
            
    def paint(self, pm):
        """
        Paint the FKP weighted density field: ``data - alpha*randoms`` using
        the input `ParticleMesh`
        
        Parameters
        ----------
        pm : ParticleMesh
            the particle mesh instance to paint the density field to
        
        Returns
        -------
        stats : dict
            a dictionary of FKP statistics, including total number, normalization,
            and shot noise parameters (see equations 13-15 of Beutler et al. 2013)
        """  
        if self.closed:
            raise ValueError("'paint' operation on a closed FKPCatalog")
                
        # setup
        columns = ['Position', 'Weight']
        stats = {}
        N_ran = N_data = 0
        S_ran = S_data = 0.
        W_ran = W_data = 0.
        
        # clear the density mesh
        pm.clear()
        
        # determine alpha from the sum of the weights
        # weights below are "completeness weights"
        #----------------------------------------------
        for [weight] in self.read('randoms', ['Weight']):
            W_ran += weight.sum()
        for [weight] in self.read('data', ['Weight']):
            W_data += weight.sum()
            
        W_ran = self.comm.allreduce(W_ran)
        W_data = self.comm.allreduce(W_data)
         
        # alpha is the ratio of the sum of the weights
        # when weights are unity, this is the normal definiton
        alpha = 1.*W_data/W_ran   
   
        # (weighted) nbar of galaxies
        volume = pm.BoxSize.prod()
        nbar   = W_data / volume
        
        # paint -1.0*alpha*N_randoms
        #------------------------------------------------------
        for [position, weight] in self.read('randoms', columns):
            Nlocal = self.painter.basepaint(pm, position, -alpha*weight)
            N_ran += Nlocal
            S_ran += (weight**2).sum()

        A_ran = nbar * W_ran
        N_ran = self.comm.allreduce(N_ran)
        S_ran = self.comm.allreduce(S_ran)
          
        if N_ran != self.randoms.size:
            args = (N_ran, self.randoms.size)
            raise ValueError("`size` mismatch when painting: `N_ran` = %d, `randoms.size` = %d" %args)

        # paint the data
        #--------------------------------------------------------
        for [position, weight] in self.read('data', columns):
            Nlocal = self.painter.basepaint(pm, position, weight)
            N_data += Nlocal 
            S_data += (weight**2).sum()
                        
        A_data = nbar * W_data
        N_data = self.comm.allreduce(N_data)
        S_data = self.comm.allreduce(S_data)
        
        if N_data != self.data.size:
            args = (N_data, self.data.size)
            raise ValueError("`size` mismatch when painting: `N_data` = %d, `data.size` = %d" %args)

        # store the stats (see equations 13-15 of Beutler et al 2013)
        # see equations 13-15 of Beutler et al 2013
        stats['W_data'] = W_data; stats['W_ran'] = W_ran
        stats['N_data'] = N_data; stats['N_ran'] = N_ran
        stats['A_data'] = A_data; stats['A_ran'] = A_ran
        stats['S_data'] = S_data; stats['S_ran'] = S_ran
        stats['alpha'] = alpha
        
        stats['A_ran'] *= alpha
        stats['S_ran'] *= alpha**2
        stats['shot_noise'] = S_ran/A_ran + S_data/A_data # the final shot noise estimate for monopole
        
        return stats
    
        
