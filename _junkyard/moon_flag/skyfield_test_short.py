import datetime as dt
from skyfield import api
import pdb

import skyfield

ny_alesund_lat = 78.924444
ny_alesund_lon = 11.928611
ny_alesund_elevation = 15

t = api.load.timescale().utc(2020, 6, 21, 12, 0, 0)
eph = api.load('de421.bsp')
earth_centre = eph['earth']
moon = eph['moon']
station_mwr_nyalesund = earth_centre + api.wgs84.latlon(ny_alesund_lat, ny_alesund_lon, elevation_m=ny_alesund_elevation)
moon_altitude_azimuth = station_mwr_nyalesund.at(t).observe(moon).apparent().altaz()
moon_alt_angle_float = skyfield.units.Angle(angle=moon_altitude_azimuth[0]).degrees
moon_azi_angle_float = skyfield.units.Angle(angle=moon_altitude_azimuth[1]).degrees

pdb.set_trace()