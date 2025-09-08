import pandas as pd
from astroquery.gaia import Gaia as gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
print("Started")

arcsec = u.Unit("arcsec")
hourangle = u.Unit("hourangle")
deg = u.Unit("deg")

SEARCH_RADIUS = 2 * arcsec       # radius for Gaia cone search [2]
DIST_TOL = 0.2                     # ±20% distance window for candidate match [2]
# 1) Load data
# Required columns:
# - 'Name'
# - 'RA (hh:mm:ss)'
# - 'DEC (dd:mm:ss)'
# - 'd (kpc)'
# -----------------------------
df = pd.read_csv("pulsar_candidates.csv")

# Clean and convert distance to parsecs
# If 'd (kpc)' might contain non-numeric characters, strip them first:
# df["d (kpc)"] = df["d (kpc)"].astype(str).str.replace(r"[^\d.\-eE]", "", regex=True)

df["d (kpc)"] = pd.to_numeric(df["d (kpc)"], errors="coerce")                           # numeric kpc [4]
df["DIST_pc"] = df["d (kpc)"] * 1000.0                                                  # pc from kpc [1]

print("Defined")
def sexa_to_deg(ra_hms: str, dec_dms: str):
    """
    Convert sexagesimal strings:
      ra_hms like '00:30:27.6'
      dec_dms like '+06:51:39'
    to ICRS degrees.
    To be compatible with Gaia
    """
    c = SkyCoord(ra_hms, dec_dms, unit=(hourangle, deg), frame="icrs")
    return c.ra.deg, c.dec.deg  # degrees expected by Gaia [5]

def query_gaia(ra_deg: float, dec_deg: float, dist_pc: float, radius: u.Quantity = SEARCH_RADIUS, tol: float = DIST_TOL):
    """
    Query Gaia DR3 around (ra_deg, dec_deg) within 'radius', compute distances from parallax,
    and keep stars within ±tol of dist_pc (pc).
    """
    coord = SkyCoord(ra=ra_deg * deg, dec=dec_deg * deg, frame="icrs")              # ICRS [5]

    adql = f"""
    SELECT gaia.source_id, gaia.ra, gaia.dec,
           gaia.parallax, gaia.phot_g_mean_mag, gaia.bp_rp
    FROM gaiadr3.gaia_source AS gaia
    WHERE 1=CONTAINS(
        POINT('ICRS', gaia.ra, gaia.dec),
        CIRCLE('ICRS', {coord.ra.deg}, {coord.dec.deg}, {radius.to(deg).value})
    )
    """
    job = gaia.launch_job(adql)                            # dr3 source table [2]
    r = job.get_results().to_pandas()

    if r.empty:
        return r

    # Gaia parallaxes are in milliarcseconds; distance(pc) = 1000 / parallax(mas) [6]
    r = r[r["parallax"].notna() & (r["parallax"] != 0)]
    if r.empty:
        return r

    r["dist_pc"] = 1000.0 / r["parallax"]                                               # pc from mas [6]

    # Keep only stars close to the pulsar distance
    lo, hi = (1 - tol) * dist_pc, (1 + tol) * dist_pc
    mask = (r["dist_pc"] >= lo) & (r["dist_pc"] <= hi)
    return r[mask]
print("Functions Defined")
# -----------------------------
# 3) Run for one pulsar - test
# -----------------------------
# row_idx = 0
# name = df.at[df.index[row_idx], "Name"]                                                 # scalar getter [7]
# ra_hms = df.at[df.index[row_idx], "RA (hh:mm:ss)"]
# print(ra_hms)
# dec_dms = df.at[df.index[row_idx], "DEC (dd:mm:ss)"]
# print(dec_dms)
# dist_pc = float(df.at[df.index[row_idx], "DIST_pc"])                                    # numeric scalar [7]
#
# ra_deg, dec_deg = sexa_to_deg(ra_hms, dec_dms)                                          # parse to degrees [5]
# print(f"Checking {name} at distance {dist_pc:.1f} pc")                                  # scalar formatting [8]
#
# results = query_gaia(ra_deg, dec_deg, dist_pc)
# print(results)
#
# 4) Loop all pulsars and collect matches
matches = []
print("Looping")
for i in range(len(df)):
    name_i = df.at[df.index[i], "Name"]
    ra_i, dec_i = sexa_to_deg(
        df.at[df.index[i], "RA (hh:mm:ss)"],
        df.at[df.index[i], "DEC (dd:mm:ss)"]
    )
    dist_i = float(df.at[df.index[i], "DIST_pc"])
    res_i = query_gaia(ra_i, dec_i, dist_i)
    if not res_i.empty:
        res_i = res_i.assign(pulsar=name_i, target_dist_pc=dist_i)
        matches.append(res_i)
all_matches = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame()
all_matches.to_csv("gaia_matches.csv", index=False)
print("Task Finished, matches saved to /gaia_matches.csv")
