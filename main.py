import pandas as pd
import numpy as np
from astroquery.gaia import Gaia as gaia
#import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
print("Started")

arcsec = u.Unit("arcsec")
hourangle = u.Unit("hourangle")
deg = u.Unit("deg")

SEARCH_RADIUS = 3 * arcsec       # radius for Gaia cone search [2]
DIST_TOL = 0.8                     # ±20% distance window for candidate match [2]
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

print(f"Loaded {len(df)} pulsars")
print(f"Column names: {df.columns.tolist()}")
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
    SELECT
        gaia.source_id, gaia.ra, gaia.dec,
        gaia.parallax, gaia.parallax_error,
        gaia.pmra, gaia.pmdec, gaia.pmra_error, gaia.pmdec_error,
        gaia.phot_g_mean_mag, gaia.bp_rp,
        gaia.ruwe, gaia.astrometric_excess_noise,
        gaia.phot_bp_rp_excess_factor
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
    r = r[r["parallax"].notna() & (r["parallax"] > 0) & (r["parallax_error"]>0)]
    print(len(r))
    if r.empty:
        return r

    r["dist_pc"] = 1000.0 / r["parallax"]                                               # pc from mas [6]
    r["parallax_snr"] = r["parallax"]/r["parallax_error"]

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
print(f"Looping {len(df)} times")
for i in range(len(df)):
    name_i = df.at[df.index[i], "Name"]
    ra_i, dec_i = sexa_to_deg(
        df.at[df.index[i], "RA (hh:mm:ss)"],
        df.at[df.index[i], "DEC (dd:mm:ss)"]
    )
    dist_i = float(df.at[df.index[i], "DIST_pc"])
    pulsar_coord = SkyCoord(ra=ra_i*deg,dec=dec_i*deg)
    print("Pulsar number: ",i+1)
    res_i = query_gaia(ra_i, dec_i, dist_i)
    if not res_i.empty:
        #Angular Seperation
        gaia_coords = SkyCoord(ra=res_i["ra"].values*deg, dec=res_i["dec"].values*deg)
        res_i["sep_arcsec"] = pulsar_coord.separation(gaia_coords).arcsec
        
        #Absolute Magnitude, distance from Gaia Parallax
        res_i["M_G"] = res_i["phot_g_mean_mag"] - 5 * np.log10(res_i["dist_pc"]/10)
        
        # Pulsar Metadata
        res_i = res_i.assign(pulsar=name_i, target_dist_pc=dist_i)
        matches.append(res_i)

#Save Resutls
if matches:
    all_matches = pd.concat(matches, ignore_index=True)

    # Save raw matches
    all_matches.to_csv("gaia_matches_raw.csv", index=False)

    # Apply quality cuts
    mask = (all_matches["ruwe"] < 1.4) & (all_matches["astrometric_excess_noise"] < 2)
    filtered = all_matches[mask]
    filtered.to_csv("gaia_matches_filtered.csv", index=False)

    print(f"Matches found: {len(all_matches)}, after cuts: {len(filtered)}")
else:
    print("No matches found.")
    all_matches = pd.DataFrame()

#quick plots
# Doesnt matter for now, only have 1 match for 1 pulsar

#if not all_matches.empty:
#    for pulsar, group in all_matches.groupby("pulsar"):
#        plt.figure(figsize=(6,5))
#        plt.scatter(group["bp_rp"], group["M_G"], c=group["sep_arcsec"], cmap="viridis", s=50)
#        plt.gca().invert_yaxis()
#        plt.colorbar(label="Separation (arcsec)")
#        plt.xlabel("BP - RP")
#        plt.ylabel("Absolute G")
#        plt.title(f"{pulsar} candidates")
#        plt.savefig(f"plots/{pulsar}_cmd.png", dpi=150)
#        plt.close()
#all_matches = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame()
#all_matches.to_csv("gaia_matches.csv", index=False)
print("Task Finished, matches saved to /gaia_matches_raw.csv and /gaia_matches_filtered.csv")
