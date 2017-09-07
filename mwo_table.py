from toolkit import json_to_stars, FitParameter, Measurement, StarProps, stars_to_json
from toolkit.utils import construct_standard_star_table
import numpy as np

f = FitParameter.from_text('calibration_constants/calibrated_f.txt')
c1 = FitParameter.from_text('calibration_constants/calibrated_c1.txt')
c2 = FitParameter.from_text('calibration_constants/calibrated_c2.txt')

calstars = json_to_stars('data/mwo_stars.json')
names = [s.name for s in calstars]

calstars_s_apo = Measurement([s.s_apo.uncalibrated.value for s in calstars],
                             err=[s.s_apo.uncalibrated.err for s in calstars],
                             time=[s.s_apo.time.jd for s in calstars])

##############################################################################
# Solve for HAT-P-11 S-indices:
calstars_s_mwo_err = np.sqrt((f.value * calstars_s_apo.value * c1.err_lower)**2 +
                             (c1.value * f.value * calstars_s_apo.err)**2 +
                             c2.err_lower**2)

calstars_s_mwo = Measurement(c1.value * calstars_s_apo.value + c2.value,
                             err=calstars_s_mwo_err,
                             time=calstars_s_apo.time)

calstars_apo_calibrated = [StarProps(name=names, s_apo=sapo, s_mwo=smwo,
                                     time=sapo.time)
                           for sapo, smwo, names in
                           zip(calstars_s_apo, calstars_s_mwo, names)]

stars_to_json(calstars_apo_calibrated, 'data/mwo_apo_calibrated.json')

#############################################################################

construct_standard_star_table(calstars_apo_calibrated)