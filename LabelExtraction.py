import numpy as np
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon, MultiPolygon

def plot_polygon(polygon, tif_file):
    """
    Plots a single polygon onto the current matplotlib figure.

    Parameters:
    polygon (Polygon or MultiPolygon): The polygon to plot.
    tif_file (rasterio.io.DatasetReader): The raster file for coordinate transformation.
    """
    exterior_coords = polygon.exterior.coords
    building_pixels = np.array([tif_file.index(*coord) for coord in exterior_coords])
    x_pixels, y_pixels = building_pixels.T
    plt.fill_between(x_pixels, y_pixels, facecolor='red')

def process_geojson(tif_path, geojson_path, output_path, limit=None):
    """
    Processes GeoJSON files and plots building polygons on corresponding TIF images.

    Parameters:
    tif_path (str): Directory containing TIF images.
    geojson_path (str): Directory containing GeoJSON files.
    output_path (str): Directory to save the output images.
    limit (int): Limit the number of files to process. Default is None (process all files).
    """
    tif_filenames = sorted(os.listdir(tif_path))
    geojson_filenames = sorted(os.listdir(geojson_path))
    os.makedirs(output_path, exist_ok=True)

    for file_index, geojson_file in enumerate(geojson_filenames):
        if limit and file_index >= limit:
            break

        geo_df = gpd.read_file(os.path.join(geojson_path, geojson_file))
        tif_filename = tif_filenames[file_index]

        with rasterio.open(os.path.join(tif_path, tif_filename)) as tif_file:
            plt.imshow(np.zeros((tif_file.height, tif_file.width)), cmap='gray')

            for _, building in geo_df.iterrows():
                if building.geometry.geom_type == 'Point':
                    continue
                if isinstance(building.geometry, MultiPolygon):
                    for polygon in building.geometry.geoms:
                        plot_polygon(polygon, tif_file)
                elif isinstance(building.geometry, Polygon):
                    plot_polygon(building.geometry, tif_file)

            plt.savefig(os.path.join(output_path, f"{file_index}.png"), dpi=215.24)
            plt.close()
