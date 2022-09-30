# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for CCL interoperability'''

__version__ = '2022.9.30'


import numpy as np

import pyccl

from glass.generator import receives, yields


@receives('zmin', 'zmax')
@yields('cl')
def ccl_matter_cl(ccl_cosmo, lmax):
    '''generator for the matter angular power spectrum from CCL'''

    l = np.arange(lmax+1)

    tr = cls = None

    while True:
        try:
            zmin, zmax = yield cls
        except GeneratorExit:
            break

        zz = np.linspace(zmin, zmax, 100)
        bz = np.ones_like(zz)
        aa = 1/(1 + zz)
        nz = ccl_cosmo.comoving_angular_distance(aa)**2/ccl_cosmo.h_over_h0(aa)

        tr = pyccl.NumberCountsTracer(ccl_cosmo, False, (zz, nz), (zz, bz), None)

        cl = pyccl.angular_cl(ccl_cosmo, tr, tr, l)

        cls = [cl]
