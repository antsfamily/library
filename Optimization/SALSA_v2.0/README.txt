
SALSA 2.0

First release: December 12, 2009

Revised version: January 13, 2012

---------------------------------------------------------------------
Copyright (2009): Manya Afonso, José M. Bioucas-Dias, and Mário Figueiredo

SALSA is distributed under the terms of the GNU General Public
License 2.0.

Permission to use, copy, modify, and distribute this software for
any purpose without fee is hereby granted, provided that this entire
notice is included in all copies of any software which is or includes
a copy or modification of this software and in all copies of the
supporting documentation for such software.
This software is being provided "as is", without any express or
implied warranty.  In particular, the authors do not make any
representation or warranty of any kind concerning the merchantability
of this software or its fitness for any particular purpose."
---------------------------------------------------------------------

This set of MATLAB files contain an implementation of the algorithms
described in the following papers:

[1] M. Afonso, J. Bioucas-Dias, and M. Figueiredo, "Fast image recovery
 using variable splitting and constrained optimization," IEEE Transactions 
 on Image Processing, vol. 19, no. 9, pp. 2345-2356, September, 2010.

[2] M. Afonso, J. Bioucas-Dias, and M. Figueiredo, "An Augmented
 Lagrangian based Method for the Constrained Formulation of Imaging Inverse  Problems", IEEE Transactions on Image Processing, Vol. 20, no. 3, 
 pp 681 - 695, March, 2011.


SALSA solves the otimization problem

min_x || y - A x ||_2^2 + lambda Phi(x)

and C-SALSA solves

min_x Phi(x) subject to || y - A x ||_2 <= epsilon

with applications in compressed sensing, image restoration
and reconstruction, sparse regression, and several other problems.

The code is available at http://cascais.lx.it.pt/~mafonso/salsa

---------------------------------------------------------------------

The main files implementing the algorithms are SALSA_v2.m and CSALSA_v2.m

For usage details, type  "help SALSA_v2" or "help CSALSA_v2"
at the MATLAB prompt.

INSTALLATION:
To get started, add the folders SALSA_vX\src and SALSA_vX\utils to the path.
The examples included do not need the path to be set.

SYSTEM REQUIREMENTS:
This version of SALSA requires MATLAB 7.4.0 (R2007) or later. All of 
our testing was done on the Windows XP platform.

Some demos may need the Rice wavelet toolbox to be installed.
(available at http://www-dsp.rice.edu/software/rwt.shtml )

The demos are "stand-alone", they do not need any other package (with which we compare
our algorithms) installed, but the experiments require the installation of the following 
packages,  and require them to be added to the MATLAB path:

TwIST : http://www.lx.it.pt/~bioucas/code/TwIST_v1.zip
SpaRSA: http://www.lx.it.pt/~mtf/SpaRSA/
FISTA:  http://ie.technion.ac.il/~becka/papers/rstls_package.zip *
SPGL1:  http://www.cs.ubc.ca/labs/scl/spgl1/ *
NESTA:  http://www.acm.caltech.edu/~nesta/ (the version used was v1.0) *

* Modified versions of the original code (to return vectors of the CPU time 
at each iteration) are included in the folder utils.

Some of the code for generating the demo on MRI reconstruction has 
been taken from l1-magic ( www.acm.caltech.edu/l1magic ).

RELEASE INFORMATION:
	Changes in the way function handles for the least square
	operation are passed to the algorithm.
	No algorithmic changes.

CONTACT INFORMATION:

            manya.afonso@lx.it.pt
            bioucas@lx.it.pt
            mario.figueiredo@lx.it.pt

This code is in development stage; thus any comments or 
bug reports are very welcome.


