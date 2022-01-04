

"""
This is a general writeup of the steps needed to create all of the patient data.

The steps are currently arbitrary and are likely to move later as the flow is understood
"""

import os
from cbctprojections import MakeCBCTProjection
basepath = os.path.join('.', 'Data', 'Patient')
RTIpath = os.path.join(basepath,'RTIMAGE')
CBCTpath = os.path.join(basepath,'CT')
MakeCBCTProjection(RIpath=RTIpath, CBCTpath=CBCTpath)