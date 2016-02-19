from nbodykit.extensionpoints import DataSource
from nbodykit import files 
import numpy

class TPMLabel(DataSource):
    plugin_name = "TPMLabel"
    
    @classmethod
    def register(kls):
        
        h = kls.parser
        h.add_argument("path", help="path to file")
        h.add_argument("-bunchsize", type=int, 
                default=1024*1024*4, help="number of particles to read per rank in a bunch")
    def finalize_attributes(self):
        if self.comm.rank == 0:
            datastorage = files.DataStorage(self.path, files.HaloLabelFile)
        else:
            datastorage = None
        datastorage = self.comm.bcast(datastorage)
        self.TotalLength = sum(datastorage.npart)

    def read(self, columns, stats, full=False):
        """ read data in parallel. if Full is True, neglect bunchsize. """
        Ntot = 0
        # avoid reading Velocity if RSD is not requested.
        # this is only needed for large data like a TPMSnapshot
        # for small Pandas reader etc it doesn't take time to
        # read velocity

        bunchsize = self.bunchsize
        if full: bunchsize = -1

        if self.comm.rank == 0:
            datastorage = files.DataStorage(self.path, files.HaloLabelFile)
        else:
            datastorage = None
        datastorage = self.comm.bcast(datastorage)

        for round, P in enumerate(
                datastorage.iter(stats=stats, comm=self.comm, 
                    columns=columns, bunchsize=bunchsize)):
            P = dict(zip(columns, P))

            yield [P[key] for key in columns]

#------------------------------------------------------------------------------