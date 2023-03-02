import datetime as dt
from skyfield import api
import pdb

import skyfield


ts = api.load.timescale()
t = ts.utc(2020, 6, 21, 12, 0, 0)

eph = api.load('de421.bsp')
# sun = eph['sun']
earth_centre = eph['earth']
moon = eph['moon']

burbach = earth_centre + api.wgs84.latlon(50.74610,8.08060, elevation_m=360)
# sun_astro = burbach.at(t).observe(sun)
moon_astro = burbach.at(t).observe(moon)

# sun_altitude_azimuth = sun_astro.apparent().altaz()
moon_altitude_azimuth = moon_astro.apparent().altaz()		# moon_astro.apparent().altaz()[0]: altitude, moon_astro.apparent().altaz()[1]: azimuth

# print(sun_altitude_azimuth) # prints altitude and azimuth of sun in a useless format
print(moon_altitude_azimuth) # prints altitude and azimuth of sun in a useless format


# convert to useful units:
# sun_alt_angle_float = skyfield.units.Angle(angle=sun_altitude_azimuth[0]).degrees
# sun_azi_angle_float = skyfield.units.Angle(angle=sun_altitude_azimuth[1]).degrees

moon_alt_angle_float = skyfield.units.Angle(angle=moon_altitude_azimuth[0]).degrees
moon_azi_angle_float = skyfield.units.Angle(angle=moon_altitude_azimuth[1]).degrees

pdb.set_trace()