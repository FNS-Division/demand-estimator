import numpy as np
import time
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import box
import logging

from demand.entities.pointofinterest import PointOfInterestCollection
from demand.handlers.populationdatahandler import PopulationDataHandler
from demand.utils import create_default_logger, create_voronoi_diagram
from demand.dataprocess import DataProcessor


class Demand:
    """
    Initializes a new instance of the Demand class.

    Parameters:
    - points_of_interest (PointOfInterestCollection): The collection of points of interest.
    - population_data_handler (PopulationDataHandler): Handler for population data.
    - radii (list): List of radii for demand analysis.
    - logger (logging.Logger): Logger instance for logging messages.
    - overlap_allowed (bool): Whether to allow the buffers for points of interest (e.g. 1km buffer) to overlap.
        If set to False, the buffer areas for each POI will be clipped to avoid overlapping
        with the buffer areas of all other POIs.

    Attributes:
    - points_of_interest (PointOfInterestCollection): The collection of points of interest.
    - population_data_handler (PopulationDataHandler): Handler for population data.
    - radii (list): List of radii for demand analysis.
    - overlap_allowed (bool): Whether to allow the buffers for points of interest (e.g. 1km buffer) to overlap.
    - are_poi_schools (bool): Whether to the points of interests represent the schools in the country.
    - population_data (pandas.DataFrame): Population data.
    - data_processor (DataProcessor): DataProcessor instance for data processing.
    - logger (logging.Logger): Logger instance for logging messages.
    - analysis_param (dict): Dictionary for storing analysis parameters.
    - analysis_stats (dict): Dictionary for storing analysis statistics.
    - analysis_results (dict): Dictionary for storing analysis results.
    - results_table (pandas.DataFrame): DataFrame containing analysis results.

    Methods:
    - get_neighbor_population_pixels(): Retrieves population data for neighboring pixels.
    - perform_analysis(): Performs demand analysis and prints the analysis summary.
    - get_storage_table(): Returns the results table (DataFrame).
    - get_results_table(): Returns the results table (DataFrame).
    - format_analysis_summary(): Formats the analysis summary for display.

    Example:
    ```python
    poi_collection = PointOfInterestCollection(data)
    pop_data_handler = PopulationDataHandler(data_dir='path/to/data', country_code='USA', dataset_year=2020)
    demand_analysis = Demand(points_of_interest=poi_collection, population_data_handler=pop_data_handler, radii=[1, 3, 5])
    demand_analysis.perform_analysis()
    ```
    """

    def __init__(self,
                 points_of_interest: PointOfInterestCollection,
                 population_data_handler: PopulationDataHandler,
                 radii=[1, 3, 5],
                 radius_for_demand=1,
                 mbps_demand_per_user=5,
                 are_poi_schools=False,
                 overlap_allowed=False,
                 logger=None
                 ):
        self.points_of_interest = points_of_interest
        self.population_data_handler = population_data_handler
        self.radii = radii
        if radius_for_demand not in radii:
            self.radii.append(radius_for_demand)

        self.radius_for_demand = radius_for_demand
        self.mbps_demand_per_user = mbps_demand_per_user
        self.are_poi_schools = are_poi_schools
        self.overlap_allowed = overlap_allowed
        self.population_data = population_data_handler.population_data
        self.data_processor = DataProcessor()

        if logger is None:
            self.logger = create_default_logger()
        elif not isinstance(logger, logging.Logger):
            raise TypeError(f'logger must be an instance of {logging.Logger}')
        else:
            self.logger = logger

        self.analysis_param = dict(
            population_dataset_year=population_data_handler.dataset_year,
            population_un_adjusted=population_data_handler.un_adjusted,
            population_one_km_res=population_data_handler.one_km_res,

        )

        self.analysis_stats = dict(
            num_points_of_interest=len(points_of_interest),
            country_population_count=round(self.population_data.population.sum(), 2)
        )

        self.analysis_results = dict(
            poi_id=points_of_interest._ids,
            population=np.empty(len(points_of_interest), dtype=object),
            poi_count=np.empty(len(points_of_interest), dtype=object)
        )

        self.traffic_demand = None

    def get_neighbor_population_pixels(self):
        """
        Retrieves population data for neighboring pixels.

        Returns:
        pandas.DataFrame: Population data for neighboring pixels.
        """
        population_tree = cKDTree(self.population_data[['lat', 'lon']].to_numpy())

        neighbors = population_tree.query_ball_point(self.points_of_interest.get_lat_lon_pairs(), r=(
            max(self.radii) + 1) * self.population_data_handler.dataset_xsize)

        return self.population_data.iloc[list(set(np.concatenate(neighbors)))]

    def perform_school_analysis(self, minimum_school_coverage=0.5):
        """
        Perform analysis to estimate the number of pupils per school within catchment areas.
        This uses open data from UNESCO on the estimated population of compulsory school age in each country.
        This function will allocate the schhol age population to the schools included in the points of interest data, which represent all the schools in the country.

        Parameters:
        - education_data_blob (str): Blob path to the CSV file containing education data.
          Default is 'education_data/NATMON_DS_03072024050744605.csv'.

        Returns:
        - pd.DataFrame: DataFrame with columns 'id' (school identifier) and 'number_of_pupils'
          (estimated number of pupils per school based on population distribution within catchment areas).
        """
        def _bounding_box(gdf):
            # Get the total bounds of the geodataframe
            minx, miny, maxx, maxy = gdf.total_bounds
            return box(minx, miny, maxx, maxy)

        def _iou(box1, box2):
            # Compute the intersection and union of the two boxes
            intersection = box1.intersection(box2).area
            union = box1.union(box2).area
            return intersection / union

        # Read school population data
        school_population_data = pd.read_csv(
            "https://zstagigaprodeuw1.blob.core.windows.net/gigainframapkit-public-container/education_data/NATMON_DS_03072024050744605.csv"
        )

        school_data_query = (
            f"LOCATION == '{self.population_data_handler.country_code}' and "
            f"Indicator == 'Population of compulsory school age, both sexes (number)' and "
            f"TIME == {self.population_data_handler.dataset_year}"
        )

        # Query the data
        queried_data = school_population_data.query(school_data_query)

        # Check if data exists for the specified country and year
        if queried_data.empty:
            raise ValueError(
                f"Education data not found for country {self.population_data_handler.country_code} "
                f"and year {self.population_data_handler.dataset_year}."
            )
        else:
            number_of_pupils_in_country = queried_data["Value"].squeeze()
            self.logger.info(
                f"The population of compulsory school age in {self.population_data_handler.dataset_year} "
                f"in {self.population_data_handler.country_code} was {number_of_pupils_in_country}"
            )

        # Convert points of interest to GeoDataFrame and create Voronoi diagram
        gdf_poi = self.data_processor.to_geodataframe(self.points_of_interest.data)
        voronoi_diagram = create_voronoi_diagram(gdf_poi)

        # Import country border
        country_boundaries = gpd.read_file('https://zstagigaprodeuw1.blob.core.windows.net/gigainframapkit-public-container/country_boundary_data/boundaries.geojson')
        borders = country_boundaries[country_boundaries["iso3cd"] == self.population_data_handler.country_code]
        del country_boundaries

        # Clip Voronoi diagram to country border
        clipped_voronoi_diagram = gpd.clip(voronoi_diagram, borders.unary_union)

        # Convert population data to GeoDataFrame, WorldPop data is in WGS84 (EPSG:4326)
        population_gpd = gpd.GeoDataFrame(self.population_data, geometry=gpd.points_from_xy(self.population_data['lon'], self.population_data['lat']), crs='epsg:4326')

        # Get bounding boxes
        schools_bbox = _bounding_box(gdf_poi)
        population_bbox = _bounding_box(population_gpd)

        # Compute IoU
        iou = _iou(schools_bbox, population_bbox)

        # Raise error if schools data does not cover the entire country
        if iou < minimum_school_coverage:
            raise ValueError(
                f"Schools data covers {round(iou*100,0)}% of the population data. Ensure that the coverage is {round(minimum_school_coverage*100,0)}%.")  # noqa: F821

        # Perform spatial join to find population within catchment areas
        joined_gdf = gpd.sjoin(population_gpd, clipped_voronoi_diagram, predicate='within')

        # Calculate pupils by school based on population distribution
        pupils_by_school = joined_gdf.groupby('id').agg({'population': 'sum'})
        pupils_by_school["total_population"] = joined_gdf["population"].sum()
        pupils_by_school["percentage"] = pupils_by_school.population / pupils_by_school.total_population
        pupils_by_school["country_total"] = number_of_pupils_in_country
        pupils_by_school["population_schools"] = pupils_by_school["percentage"] * pupils_by_school["country_total"]
        pupils_by_school.reset_index(inplace=True)
        pupils_by_school = pupils_by_school[['id', 'population_schools']]

        return pupils_by_school

    def perform_analysis(self, minimum_school_coverage=0.5):
        """
        Performs demand analysis and prints the analysis summary.
        """
        # Start the timer
        start_time = time.time()

        # Restrict the worldpop data to the area around the points of interest, and convert to a GeoDataFrame
        gdf_pop = self.data_processor.to_geodataframe(self.get_neighbor_population_pixels())

        # Create buffers around the worldpop data, to fill in the voids between points
        self.data_processor.buffer_gdf_in_meters(
            gdf_pop,
            buffer_distance_meters=(1 / 2 if self.population_data_handler.one_km_res else 0.1 / 2) * 1000,
            cap_style=3,
            inplace=True)

        # Total area covered by the population data
        pop_pixel_area = gdf_pop.to_crs('epsg:3857').area.iloc[0]

        # Convert the points of interest to a GeoDataFrame
        gdf_poi = self.data_processor.to_geodataframe(self.points_of_interest.data)

        # Create a dictionary to store the population data for each point of interest
        pop_dict = dict(poi_id=gdf_poi.id.values)

        # If overlap is not allowed, create Voronoi diagram around POIs
        if not self.overlap_allowed:
            self.logger.info('Creating Voronoi diagram around POIs...')
            gdf_poi_voronoireg = create_voronoi_diagram(gdf_poi)

        # Loop through each value in the radii list
        for radius in self.radii:

            self.logger.info(f'Overlaying population data for {radius}km radius around a point of interest...')

            # Create a buffer around the points of interest
            gdf_poi_area = self.data_processor.buffer_gdf_in_meters(
                gdf_poi, buffer_distance_meters=radius * 1000, cap_style=1)

            # If overlap is not allowed
            if not self.overlap_allowed:
                # Clip the buffer around the points of interest with the Voronoi diagram
                gdf_clipped_buffers = gdf_poi_area.reset_index().overlay(
                    gdf_poi_voronoireg,
                    how='intersection')

                # Filter out the areas that do not intersect
                gdf_clipped_buffers = gdf_clipped_buffers.loc[gdf_clipped_buffers.id_1 == gdf_clipped_buffers.id_2, :].copy()
                gdf_clipped_buffers = gdf_clipped_buffers.loc[:, ["id_1"] + gdf_poi_area.columns[1:].tolist()]
                gdf_clipped_buffers.rename(columns={'id_1': 'id'}, inplace=True)

                # Substitute the GeoDataFrame of buffers with a GeoDataFrame of clipped buffers
                gdf_poi_area = gdf_clipped_buffers.copy()

            # Overlay the population data with the buffer around the points of interest, which creates a new row
            # for each intersection between a POI buffer and population square km grid
            gdf_overlayed = gdf_pop.reset_index().overlay(
                gdf_poi_area,
                how='intersection')

            # Convert to EPSG 3857 which is projected, and worldwide, and compute area
            gdf_overlayed['area_joined'] = gdf_overlayed.to_crs('epsg:3857').area

            # Approximate the population for each intersection
            gdf_overlayed[f'population_{radius}km'] = (
                gdf_overlayed['area_joined'] / pop_pixel_area) * gdf_overlayed['population']

            # Checks if any points of interest have no population data
            missing_ids = [id for id in gdf_poi.id if id not in list(gdf_overlayed['id'])]

            # Add rows with population=0 for points of interest with no population data
            gdf_overlayed = pd.concat([gdf_overlayed, pd.DataFrame(
                {'id': missing_ids, f'population_{radius}km': np.repeat(0, len(missing_ids))})])

            # Add the population data to the dictionary for the given radius
            pop_dict[f'population_{radius}km'] = np.round(
                gdf_overlayed[['id', f'population_{radius}km']].groupby(
                    'id').sum().loc[gdf_poi.id, f'population_{radius}km'].values
            ).astype(int)

            # Populate the analysis stats dictionary with the mean, median, and sum of the population data
            self.analysis_stats[f'mean_population_{radius}km'] = round(np.mean(pop_dict[f'population_{radius}km']), 2)
            self.analysis_stats[f'median_population_{radius}km'] = round(
                np.median(pop_dict[f'population_{radius}km']), 2)
            self.analysis_stats[f'sum_population_{radius}km'] = round(np.sum(pop_dict[f'population_{radius}km']), 2)

            self.logger.info(f'Gathering point of interest count for {radius}km radius around point...')

            # Count the number of other POIs within the radius of each POI
            pop_dict[f'poi_count_{radius}km'] = gdf_poi_area.sjoin(gdf_poi).groupby(
                'id_left').count().loc[gdf_poi.id, 'id_right'].values

            self.analysis_stats[f'mean_poi_count_{radius}km'] = round(np.mean(pop_dict[f'poi_count_{radius}km']), 2)
            self.analysis_stats[f'median_poi_count_{radius}km'] = np.median(pop_dict[f'poi_count_{radius}km'])
            self.analysis_stats[f'sum_poi_count_{radius}km'] = np.sum(pop_dict[f'poi_count_{radius}km'])

        # If the POIs are schools, then also perform the school analysis
        if self.are_poi_schools:
            self.logger.info('Performing schools population analysis...')
            schools_analysis_result = self.perform_school_analysis(minimum_school_coverage)
            schools_analysis_result = schools_analysis_result.set_index('id', inplace=False)
            schools_analysis_result = schools_analysis_result.reindex(pop_dict["poi_id"])
            schools_analysis_result["population_schools"] = schools_analysis_result["population_schools"].fillna(0)
            pop_dict["population_schools"] = schools_analysis_result["population_schools"].values

        # Store results in dictionary format in analysis_results
        for i in range(len(self.points_of_interest)):
            population_merged_dict = dict()
            poi_count_merged_dict = dict()
            for key, values in pop_dict.items():
                if key.startswith('pop'):
                    population_merged_dict[key] = values[i]
                elif key.startswith('poi_count'):
                    poi_count_merged_dict[key] = values[i]
            self.analysis_results['population'][i] = population_merged_dict
            self.analysis_results['poi_count'][i] = poi_count_merged_dict

        # Compute the demand in mbps per POI
        traffic_demand = pd.DataFrame({'poi_id': [id for id in self.points_of_interest.data.id], 'mbps_demand_per_user': [
                                      x['mbps_demand_per_user'] if 'mbps_demand_per_user' in x else self.mbps_demand_per_user for x in self.points_of_interest.data.additional_fields]})  # noqa: E501
        traffic_demand = traffic_demand.set_index('poi_id', inplace=False)

        # Get the number of users per POI: this is either the population in a given radius around POI or the number of students
        if self.are_poi_schools:
            number_of_users = schools_analysis_result.copy()
            number_of_users = number_of_users.rename(columns={'population_schools': 'number_of_users'})
        else:
            number_of_users = pd.DataFrame({
                'poi_id': pop_dict["poi_id"],
                'number_of_users': pop_dict[f"population_{self.radius_for_demand}km"]})
            number_of_users = number_of_users.set_index('poi_id', inplace=False)

        # Multiply the demand per user by the number of users to get the total demand in mbps per POI
        traffic_demand = traffic_demand.merge(number_of_users, left_index=True, right_index=True)
        traffic_demand["total_mbps"] = traffic_demand["mbps_demand_per_user"] * traffic_demand["number_of_users"]
        self.traffic_demand = traffic_demand.reset_index().rename(columns={'index': 'poi_id'})

        # Store the time taken for the analysis
        self.analysis_stats['analysis_time'] = round(time.time() - start_time, 2)

        # Store demand analysis reslts as class attribute and print analysis summary
        self.results_table = pop_dict
        return print(self.format_analysis_summary())

    def get_results_table(self):
        """
        Returns the storage table (DataFrame).
        """
        storage_table = pd.DataFrame(self.analysis_results).merge(self.traffic_demand[["poi_id", "number_of_users", "total_mbps"]], on='poi_id', how='left')
        return storage_table

    def get_storage_table(self):
        """
        Returns the results table (DataFrame).
        """
        results_table = pd.DataFrame(self.results_table).merge(self.traffic_demand[["poi_id", "number_of_users", "total_mbps"]], on='poi_id', how='left')
        return results_table

    def format_analysis_summary(self):
        """
        Formats the analysis summary for display.
        """
        summary = ""

        # Format the analysis_stats dictionary into a human-readable summary
        summary += "Demand Analysis Summary:\n"
        summary += f"Number of points of interest: {self.analysis_stats['num_points_of_interest']}\n"
        summary += f"Country population count: {self.analysis_stats['country_population_count']}\n"

        for radius in self.radii:
            summary += f"Mean population count of {radius}km area around a point of interest: {self.analysis_stats[f'mean_population_{radius}km']}\n"
            summary += f"Median population count of {radius}km area around a point of interest: {self.analysis_stats[f'median_population_{radius}km']}\n"
            summary += f"Sum population count of {radius}km area around a point of interest: {self.analysis_stats[f'sum_population_{radius}km']}\n"
            summary += f"Mean point of interest count of {radius}km area around a point of interest: {self.analysis_stats[f'mean_poi_count_{radius}km']}\n"
            summary += f"Median point of interest count of {radius}km area around a point of interest: {self.analysis_stats[f'median_poi_count_{radius}km']}\n"
            summary += f"Sum point of interest count of {radius}km area around a point of interest: {self.analysis_stats[f'sum_poi_count_{radius}km']}\n"

        summary += f"Time taken for analysis: {self.analysis_stats['analysis_time']} seconds\n"

        return summary
