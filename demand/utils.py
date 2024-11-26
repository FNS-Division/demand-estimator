import logging
import numpy as np
import itertools
import io
import geopandas as gpd
import folium
import time
import random
import string
from shapely.strtree import STRtree
from shapely import MultiPoint, GeometryCollection
from shapely.ops import voronoi_diagram
from scipy.spatial import cKDTree
import math


def generate_dataset_id(prefix=''):
    """
    Generate a unique dataset ID based on the current timestamp and a random string.

    Args:
        prefix (str): An optional prefix to include in the dataset ID.

    Returns:
        str: The generated dataset ID.
    """
    timestamp = int(time.time())  # Get the current UNIX timestamp
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    dataset_id = f"{prefix}-{timestamp}-{random_str}"
    return dataset_id


def haversine_(lats, lons, R=6371e3, upper_tri=False):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) using the
    Haversine formula.

    Parameters
    ----------
    lats, lons: array-like
        Arrays of latitudes and longitudes of the two points.
        Each array should have shape (2,) where the first element
        is the latitude and the second element is the longitude.
    upper_tri : bool, optional
        If True, returns the distance matrix in upper triangular form.
        Default is False.
    R : float, optional
        Radius of the earth in meters. Default is 6371000.0 m.

    Returns
    -------
    ndarray
        The distance matrix between the points in meters.
        If `upper_tri` is True, returns the upper triangular form of the matrix.

    """

    if not len(lats) == len(lons):
        raise ValueError("The length of 'lats' and 'lons' must be equal.")

    # Convert latitudes and longitudes to radians
    lat_rads = np.radians(lats)
    lon_rads = np.radians(lons)

    # Compute pairwise haversine distances using broadcasting
    dlat = lat_rads[:, np.newaxis] - lat_rads[np.newaxis, :]
    dlon = lon_rads[:, np.newaxis] - lon_rads[np.newaxis, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_rads[:, np.newaxis]) * \
        np.cos(lat_rads[np.newaxis, :]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = R * c

    if len(lats) == 2:
        distances = distances[0, 1]
    elif upper_tri:
        i_upper = np.triu_indices(distances.shape[0], k=1)
        distances = distances[i_upper]

    return distances


def calculate_haversine_for_pair(lat1, lon1, lat2, lon2, R=6371e3):
    """
    Calculate the haversine distance between two pairs of latitude and longitude.

    Parameters:
    - lat1 (float): Latitude of the first point.
    - lon1 (float): Longitude of the first point.
    - lat2 (float): Latitude of the second point.
    - lon2 (float): Longitude of the second point.
    - R (float): Radius of the Earth.

    Returns:
    float: Haversine distance between the two points.
    """

    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c

    return distance


def setup_logger(logger_name, log_filepath, console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Set up a logger with both console and file handlers.

    Parameters:
    - logger_name: str
        Name of the logger.
    - log_filepath: str
        File path for the log file.
    - console_level: int, optional
        Console logging level. Default is logging.INFO.
    - file_level: int, optional
        File logging level. Default is logging.DEBUG.

    Returns:
    - logger: logging.Logger
        Configured logger.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Create file handler and set its level
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(file_level)

    # Create console handler and set its level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def create_default_logger(logger_name=__name__, console_level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(console_handler)

    return logger


def create_voronoi_diagram(points_df: gpd.GeoDataFrame, identifier_column: str = "id"):
    """
    Create a Voronoi diagram based on the provided points.

    This method generates a Voronoi diagram around the points of interest from the given GeoDataFrame.
    It reorders the resulting polygons to follow the same order as the input points and converts
    the diagram into a GeoDataFrame.

    Parameters:
    - points_df (gpd.GeoDataFrame): A GeoDataFrame containing the points for which the Voronoi diagram
    is to be created. It should have columns 'lon' and 'lat' for longitude and latitude. Must have a valid CRS.
    - identifier_column (str): The column name in points_df used as an identifier for the points.
    Defaults to "id".

    Returns:
    - gpd.GeoDataFrame: A GeoDataFrame containing the Voronoi regions as geometries, with the same
    coordinate reference system (crs) as the input points_df. An 'id' column is added from the
    identifier_column of points_df.
    """
    # Check if points_df has a valid CRS
    if points_df.crs is None:
        raise ValueError("Input GeoDataFrame must have a valid CRS set.")

    # Create a Voronoi diagram around the points of interest
    voronoi_points = MultiPoint(points_df[["lon", "lat"]].values)
    voronoi_regions = voronoi_diagram(voronoi_points)

    # Reorder the resulting polygons, such that they follow the same order as the POIs
    voronoi_regions_tree = STRtree(list(voronoi_regions.geoms))
    ordered_voronoi_diagram = GeometryCollection([voronoi_regions.geoms[voronoi_regions_tree.nearest(point)] for point in voronoi_points.geoms])

    # Convert the Voronoi diagram into a GeoDataFrame
    voronoi_regions_unpacked = []
    for geom in ordered_voronoi_diagram.geoms:
        voronoi_regions_unpacked.append({"geometry": geom})
    gdf_poi_voronoireg = gpd.GeoDataFrame(voronoi_regions_unpacked, geometry="geometry", crs=points_df.crs)
    gdf_poi_voronoireg["id"] = points_df[[identifier_column]]

    return gdf_poi_voronoireg


def create_bubble_map(points, column_name, bubble_color='blue', zoom_level=11):
    """
    Create an interactive folium map with bubble markers sized by a specified column.
    
    Parameters:
    -----------
    points : GeoDataFrame
        Input GeoDataFrame containing points and data
    column_name : str
        Name of the column to be used for bubble sizing
    bubble_color : str, optional
        Color of the bubbles (default: 'blue')
    zoom_level : int, optional
        Initial zoom level of the map (default: 11)
    
    Returns:
    --------
    folium.Map
        Interactive folium map
    """
    
    # Convert GeoDataFrame to EPSG:4326 if it's not already
    points_wgs84 = points.to_crs(epsg=4326)

    # Calculate center of the map
    center_lat = points_wgs84.geometry.y.mean()
    center_lon = points_wgs84.geometry.x.mean()

    # Create the map
    m = folium.Map(location=[center_lat, center_lon], 
                   zoom_start=zoom_level, 
                   tiles="cartodb positron")

    # Normalize sizes for visualization (between 5 and 30)
    min_val = points_wgs84[column_name].min()
    max_val = points_wgs84[column_name].max()

    # Add points to the map
    for idx, row in points_wgs84.iterrows():
        # Calculate normalized radius (between 5 and 30)
        radius = 5 + ((row[column_name] - min_val) / (max_val - min_val)) * 25
        
        # Create popup text
        popup_text = f"{column_name}: {row[column_name]}"
        
        # Add circle marker
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=radius,
            color=bubble_color,
            fill=True,
            fill_color=bubble_color,
            fill_opacity=0.6,
            popup=popup_text
        ).add_to(m)

    # Add a legend with circle representation
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; 
                border:2px solid grey; z-index:9999; 
                background-color:white;
                padding: 10px;
                border-radius: 5px;">
         <p><b>{column_name}</b></p>
         <div style="display: inline-block; 
                     width: 20px; 
                     height: 20px; 
                     background-color: {bubble_color};
                     border-radius: 50%;
                     opacity: 0.6;
                     margin-right: 5px;"></div>
         <span>Value per point</span><br>
         <small>Size indicates {column_name.lower()}</small>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m
