"""
add_ecoregions.py

Adds the ecoregion, province and realm (Spalding et al.) from the .shp
file that overlaps with each row's co-ordinates

Author: Aidan Murray
Date: 2025-09-26
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path

BASE_PATH = Path("../data")
OUT = BASE_PATH / "ecoregions"
SHAPEFILE = BASE_PATH / "shapefile"

gdf_polygons = gpd.read_file(SHAPEFILE / "marine_ecoregions.shp")
filenames = [
    "combined_filtered.csv", "test.csv", "train_partial.csv", "validation.csv"
    ]

for name in filenames:
    df = pd.read_csv(BASE_PATH / name)

    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df["point.pose.lon"], df["point.pose.lat"]
            ),
        crs="EPSG:4326",
    )

    if gdf_polygons.crs != gdf_points.crs:
        gdf_polygons = gdf_polygons.to_crs(gdf_points.crs)

    gdf_joined = gpd.sjoin(
        gdf_points, gdf_polygons, how="left", predicate="within"
        )

    df = df.join(gdf_joined[['ECOREGION', 'REALM', 'PROVINCE']])

    df.to_csv(OUT / name)