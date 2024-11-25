import numpy as np
import pandas as pd
import geopandas as gp
from shapely import wkt
import json
from osgeo import gdal
import pyproj
import uuid
from typing import Optional

import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling


class DataProcessor:

    def generate_uuid(self, size=1):
        """
        Generate UUIDs.

        Parameters:
        - size (int): Number of UUIDs to generate. Default is 1 for a single UUID.

        Returns:
        - str or List[str]: Single UUID or list of UUIDs.
        """

        if size > 1:
            return [str(uuid.uuid4()) for _ in range(size)]
        else:
            return str(uuid.uuid4())

    def get_georeference_columns(self, data, lat_keywords=['latitude', 'lat', 'y', 'lat_', 'lat(s)', '_lat'],
                                 lon_keywords=['longitude', 'lon', 'long', 'x', 'lon_', 'lon(e)', 'long(e)', '_lon']):
        """
        Searches for latitude and longitude columns in the input data using a list of keywords.

        Parameters:
        - data (pandas.DataFrame): A DataFrame containing the data to search for latitude and longitude columns.
        - lat_keywords (list of str): A list of keywords to search for in column names to identify latitude columns.
        - lon_keywords (list of str): A list of keywords to search for in column names to identify longitude columns.

        Returns:
        - Tuple[str, str]: A tuple of two strings representing the names of the latitude and longitude columns, respectively.

        Raises:
        - ValueError: If no unique pair of latitude/longitude columns can be found in the input data.
        """

        # Search for columns that match common names for latitude and longitude
        lat_cols = [col for col in data.columns if any(keyword == col.lower() for keyword in lat_keywords)]
        lon_cols = [col for col in data.columns if any(keyword == col.lower() for keyword in lon_keywords)]

        # Check if exactly one latitude and longitude column is found
        if len(lat_cols) == 1 and len(lon_cols) == 1:
            return lat_cols[0], lon_cols[0]
        elif len(lat_cols) == 0 and len(lon_cols) == 0:
            raise ValueError("No latitude or longitude columns found.")
        elif len(lat_cols) == 0:
            raise ValueError("No latitude columns found.")
        elif len(lon_cols) == 0:
            raise ValueError("No longitude columns found.")
        else:
            raise ValueError("Could not find a unique pair of latitude/longitude columns.")

    def rename_georeference_columns(
        self,
        data,
        lat_col: str = 'lat',
        lon_col: str = 'lon',
        old_lat_col: Optional[str] = None,
        old_lon_col: Optional[str] = None,
        inplace=False
    ) -> pd.DataFrame:
        """
        Renames the georeference columns of a given data frame to the specified column names.

        Parameters:
        - data (pd.DataFrame): The data frame containing the georeference columns to be renamed.
        - lat_col (str): The new name for the latitude column. Default is 'lat'.
        - lon_col (str): The new name for the longitude column. Default is 'lon'.
        - old_lat_col (Optional[str]): The old name for the latitude column. If not specified,
          the method will attempt to find the old name using `get_georeference_columns()`.
        - old_lon_col (Optional[str]): The old name for the longitude column. If not specified,
          the method will attempt to find the old name using `get_georeference_columns()`.
        - inplace (bool): Whether to perform the operation in-place. Default is False.

        Returns:
        - pd.DataFrame: The input data frame with the georeference columns renamed.

        Raises:
        - ValueError: If the specified old column names do not exist in the data frame.
        """

        if not inplace:
            data = data.copy()

        # If the old column names are not specified, attempt to find them
        if old_lat_col is None or old_lon_col is None:
            old_lat_col, old_lon_col = self.get_georeference_columns(data)

        # Check if the old column names exist in the data frame
        if old_lat_col not in data.columns or old_lon_col not in data.columns:
            raise ValueError(
                f"One or both of the specified old column names '{old_lat_col}', '{old_lon_col}' do not exist in the data frame.")

        # If the old column names don't match the new ones, rename them
        if old_lat_col != lat_col:
            data.rename(columns={old_lat_col: lat_col}, inplace=True)
        if old_lon_col != lon_col:
            data.rename(columns={old_lon_col: lon_col}, inplace=True)

        return data if not inplace else None

    def to_geodataframe(self, data, lat_col=None, lon_col=None, rename_georeference_columns=True, crs='EPSG:4326', inplace=False):
        """
        Converts a DataFrame containing latitude and longitude columns into a GeoDataFrame.

        Parameters:
        - data : pandas.DataFrame
            The DataFrame containing the latitude and longitude columns.
        - lat_col : str, optional
            The name of the column containing latitude values. If not provided, the method will try to find it.
        - lon_col : str, optional
            The name of the column containing longitude values. If not provided, the method will try to find it.
        - rename_georeference_columns : bool, optional
            Whether to rename the georeference columns if they have different names. Default is True.
        - crs : str, optional
            The coordinate reference system (CRS) of the GeoDataFrame. Default is 'EPSG:4326'.
        - inplace : bool, optional
            Whether to perform the operation in-place. Default is False.

        Returns:
        - geopandas.GeoDataFrame
            A GeoDataFrame with the same columns as the input DataFrame and a new geometry column containing
            the Point objects created from the latitude and longitude columns.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("The 'data' parameter must be a pandas DataFrame.")

        if not inplace:
            data = data.copy()

        # If the latitude and longitude columns are not already specified, try to find them
        if lat_col is None or lon_col is None:
            if rename_georeference_columns:
                data = self.rename_georeference_columns(data)
                lat_col, lon_col = 'lat', 'lon'
            else:
                lat_col, lon_col = self.get_georeference_columns(data)

        if 'geometry' not in data:
            # Extract the latitude and longitude values from the input data
            latitudes = data[lat_col]
            longitudes = data[lon_col]

            # Create a new GeoDataFrame with the input data and geometry column
            geometry = gp.points_from_xy(longitudes, latitudes)
            gdf = gp.GeoDataFrame(data, geometry=geometry, crs=crs)
        else:
            # Convert 'geometry' column from WKT format to GeoSeries
            data['geometry'] = data['geometry'].apply(wkt.loads)
            gdf = gp.GeoDataFrame(data, crs=crs)

        return gdf if not inplace else None

    def clip_gdf_with_polygon(self, gdf, clip_polygon):
        """
        Clips a GeoDataFrame with a given polygon.

        Parameters:
        - gdf : geopandas.GeoDataFrame
            The GeoDataFrame to be clipped.
        - clip_polygon : shapely.geometry.Polygon
            The polygon used for clipping.

        Returns:
        - geopandas.GeoDataFrame
            The clipped GeoDataFrame.
        """

        clipped_gdf = gdf[gdf.within(clip_polygon)]
        return clipped_gdf

    def latlon_to_utm(self, lat, lon, utm_crs):
        """
        Converts latitude and longitude to UTM coordinates.

        Parameters:
        - lat : float
            Latitude.
        - lon : float
            Longitude.
        - utm_crs : pyproj.CRS
            UTM coordinate reference system.

        Returns:
        - Tuple[float, float]
            UTM x, y coordinates.
        """

        geographic_crs = pyproj.CRS("EPSG:4326")

        # Create a transformer to convert between geographic and UTM CRS
        transformer = pyproj.Transformer.from_crs(geographic_crs, utm_crs, always_xy=True)

        # Transform latitude and longitude to UTM x, y coordinates
        x, y = transformer.transform(lon, lat)

        return x, y

    def buffer_gdf_in_meters(self, gdf, buffer_distance_meters, cap_style=1, inplace=False):
        """
        Buffers a GeoDataFrame with a given buffer distance in meters.

        Parameters:
        - gdf : geopandas.GeoDataFrame
            The GeoDataFrame to be buffered.
        - buffer_distance_meters : float
            The buffer distance in meters.
        - cap_style : int, optional
            The style of caps. 1 (round), 2 (flat), 3 (square). Default is 1.
        - inplace : bool, optional
            Whether to perform the operation in-place. Default is False.

        Returns:
        - geopandas.GeoDataFrame
            The buffered GeoDataFrame.
        """

        if not inplace:
            gdf = gdf.copy()

        input_crs = gdf.crs

        # create a custom UTM CRS based on the calculated UTM zone
        utm_crs = gdf.estimate_utm_crs()

        # transform your GeoDataFrame to the custom UTM CRS:
        gdf_projected = gdf.to_crs(utm_crs)

        # create the buffer in meters:
        gdf["geometry"] = gdf_projected['geometry'].buffer(buffer_distance_meters, cap_style=cap_style)

        # transform the buffer geometry back to input crs
        gdf['geometry'] = gdf.geometry.to_crs(input_crs)

        return gdf if not inplace else None

    def process_tif(self, file_path, band_no: int = 1, drop_nodata: bool = True):
        """
        Processes a .tif file and returns a pandas DataFrame containing the longitude, latitude,
        and pixel values of the file.

        Parameters:
        - file_path (str): The path to the .tif file.
        - band_no (int): The band number of the .tif file to be read. Default is 1.
        - drop_nodata (bool): Whether to drop the pixels with no data value. Default is True.

        Returns:
        - pandas.DataFrame: A DataFrame containing the longitude, latitude, and pixel values of the .tif file.
        """
        # Open the tif file with gdal
        try:
            tif = gdal.Open(file_path)
        except Exception:
            raise ValueError('Unable to open the tif file!')

        # Check if the band exists in the tif file
        if band_no < 1 or band_no > tif.RasterCount:
            raise ValueError(f"Invalid band number {band_no} for file {file_path.split('/')[-1]}.")

        # Get the specified band
        band = tif.GetRasterBand(band_no)

        # Read band values as array
        band_values = band.ReadAsArray()

        # Get the no data value of the band
        nodata_value = band.GetNoDataValue()

        # Get the geotransform parameters of the tif file
        offX, xsize, line1, offY, line2, ysize = tif.GetGeoTransform()

        # Get the number of columns and rows in the tif file
        cols = tif.RasterXSize
        rows = tif.RasterYSize

        # Create one-dimensional arrays for x and y
        x = np.linspace(offX + xsize / 2, offX + xsize / 2 + (cols - 1) * xsize, cols)
        y = np.linspace(offY + ysize / 2, offY + ysize / 2 + (rows - 1) * ysize, rows)

        # Create the mesh based on these arrays
        X, Y = np.meshgrid(x, y)

        # Extract the pixel values, longitude, and latitude arrays from the tif file
        if drop_nodata:
            nodata_mask = band_values != nodata_value
            pixel_values = np.extract(nodata_mask, band_values)
            lons = np.extract(nodata_mask, X)
            lats = np.extract(nodata_mask, Y)
        else:
            pixel_values = band_values.flatten()
            lons = X.flatten()
            lats = Y.flatten()

        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame({
            'lon': lons,
            'lat': lats,
            'pixel_value': pixel_values}
        )

        return df

    def process_rgb_tif(self, file_path):
        """
        Processes a .tif file and returns a pandas DataFrame containing the longitude, latitude,
        and pixel values of the file.

        Parameters:
            file_path (str): The path to the .tif file.
            band_no (int): The band number of the .tif file to be read. Default is 1.
            drop_nodata (bool): Whether to drop the pixels with no data value. Default is True.
            return_res (bool): Whether to return the xsize of the .tif file along with the DataFrame.
                            Default is False.

        Returns:
            pandas.DataFrame: A DataFrame containing the longitude, latitude, and pixel values of the .tif file.
        """

        # Open the tif file with gdal
        try:
            tif = gdal.Open(file_path)
        except Exception:
            raise ValueError('Unable to open the tif file!')

        band_values = {}

        for band_no in [1, 2, 3]:
            # Get the specified band
            band = tif.GetRasterBand(band_no)

            # Read band values as array
            band_values[band_no] = band.ReadAsArray()

        # Get the geotransform parameters of the tif file
        offX, xsize, line1, offY, line2, ysize = tif.GetGeoTransform()

        # Get the number of columns and rows in the tif file
        cols = tif.RasterXSize
        rows = tif.RasterYSize

        # Create one-dimensional arrays for x and y
        x = np.linspace(offX + xsize / 2, offX + xsize / 2 + (cols - 1) * xsize, cols)
        y = np.linspace(offY + ysize / 2, offY + ysize / 2 + (rows - 1) * ysize, rows)

        # Create the mesh based on these arrays
        X, Y = np.meshgrid(x, y)

        # Extract the pixel values, longitude, and latitude arrays from the tif file
        lons = X.flatten()
        lats = Y.flatten()

        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame({
            'lon': lons,
            'lat': lats,
            'r': band_values[1].flatten(),
            'g': band_values[2].flatten(),
            'b': band_values[3].flatten(), }
        )

        return df

    def get_tif_xsixe(self, file_path):

        # Open the tif file with gdal
        try:
            tif = gdal.Open(file_path)
        except Exception:
            raise ValueError('Unable to open the tif file!')

        # Get the geotransform parameters of the tif file
        _, xsize, _, _, _, ysize = tif.GetGeoTransform()

        return xsize

    def process_tif2(self, input_file, drop_nodata=True):

        # Read the input raster data
        with rasterio.open(input_file) as src:
            # Get the pixel values as a 2D array
            band = src.read(1)

            transform = src.transform
            pixel_size_x = transform.a
            pixel_size_y = transform.e

            # Get the coordinates for each pixel
            x_coords, y_coords = np.meshgrid(
                np.linspace(src.bounds.left + pixel_size_x / 2, src.bounds.right - pixel_size_x / 2, src.width),
                np.linspace(src.bounds.top + pixel_size_y / 2, src.bounds.bottom - pixel_size_y / 2, src.height)
            )

            # Extract the pixel values, longitude, and latitude arrays from the tif file
            if drop_nodata:
                nodata_value = src.nodata
                nodata_mask = band != nodata_value
                pixel_values = np.extract(nodata_mask, band)
                lons = np.extract(nodata_mask, x_coords)
                lats = np.extract(nodata_mask, y_coords)
            else:
                pixel_values = band.flatten()
                lons = x_coords.flatten()
                lats = y_coords.flatten()

            # Flatten the arrays and combine them into a DataFrame
            data = pd.DataFrame({
                'lon': lons,
                'lat': lats,
                'pixel_value': pixel_values
            })

        return data

    def resample_data(self, input_file, output_file, scale_factor):

        with rasterio.open(input_file, 'r') as src:
            # Calculate the shape and transform of the resampled raster
            out_shape = (int(src.height * scale_factor), int(src.width * scale_factor))
            out_transform = src.transform * src.transform.scale(
                (src.width / out_shape[1]),
                (src.height / out_shape[0])
            )

            # Create a new raster with the resampled data
            with rasterio.open(output_file, 'w', **src.profile) as dst:
                for i in range(1, src.count + 1):
                    src_data = src.read(i)
                    dst_data = dst.read(i)

                    resampled_data = rasterio.warp.reproject(
                        source=src_data,
                        destination=dst_data,
                        src_transform=src.transform,
                        dst_transform=out_transform,
                        src_crs=src.crs,
                        dst_crs=src.crs,
                        resampling=Resampling.nearest
                    )

                    dst.write(resampled_data, i)

    def clip_data(self, input_file, output_file, clip_polygon_geojson):

        # Read the GeoJSON file with the clip polygon
        with open(clip_polygon_geojson) as f:
            clip_polygon = json.load(f)

        # Read the input raster data
        with rasterio.open(input_file) as src:
            # Clip the raster data using the clip polygon
            out_image, out_transform = mask(src, [clip_polygon], crop=True)

            # Update the metadata of the clipped raster
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})

            # Write the clipped raster data to the output file
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(out_image)
