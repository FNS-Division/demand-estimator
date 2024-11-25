from demand.entities.entity import Entity, EntityCollection


class PointOfInterest(Entity):
    """
    Represents a Point of Interest (POI) in geographical space.

    Parameters:
    - poi_id (str): Unique identifier for the POI.
    - lat (float): Latitude of the POI in decimal degrees.
    - lon (float): Longitude of the POI in decimal degrees.
    - poi_type: Type or category of the POI.
    - is_connected (bool): Indicates whether the POI is connected.
    - connectivity_type: Type of connectivity for the POI.
    - entity_type (str, optional): Type of the entity, default is 'Point of Interest'.
    - **kwargs: Additional fields to store in the POI.

    Attributes:
    - poi_type: Type or category of the POI.
    - is_connected (bool): Indicates whether the POI is connected.
    - connectivity_type: Type of connectivity for the POI.

    Example:
    ```python
    # Create a PointOfInterest instance
    poi = PointOfInterest(poi_id='poi1', lat=34.0522, lon=-118.2437, poi_type='Restaurant', is_connected=True, connectivity_type='Wi-Fi')

    # Access attributes of the POI
    poi_id = poi.poi_id
    poi_type = poi.poi_type
    is_connected = poi.is_connected
    ```
    """

    def __init__(self,
                 poi_id: str,
                 lat: float,
                 lon: float,
                 poi_type,
                 is_connected,
                 connectivity_type,
                 entity_type='Point of Interest',
                 **kwargs):

        super().__init__(id=poi_id, lat=lat, lon=lon, entity_type=entity_type, **kwargs)

        self.poi_type = poi_type
        self.is_connected = is_connected
        self.connectivity_type = connectivity_type

    @property
    def poi_id(self):
        # Get the unique identifier of the Point of Interest.
        return self.id

    @poi_id.setter
    def poi_id(self, value):
        # Set the unique identifier of the Point of Interest.
        self.id = value

    def __repr__(self):
        return f"PointOfInterest(poi_id={self.id}, lat={self.lat}, lon={self.lon},\
            is_connected= {self.is_connected}, connectivity_type= {self.connectivity_type}, attributes={self.additional_fields})"


class PointOfInterestCollection(EntityCollection):
    """
    Represents a collection of Point of Interest entities.

    Parameters:
    - points_of_interest (list, optional): List of PointOfInterest instances.
    - poi_records (list, optional): List of dictionaries representing POI records.
    - poi_file_path (str, optional): Path to a CSV file containing POI records.

    Example:
    ```python
    # Create a PointOfInterestCollection instance
    poi_collection = PointOfInterestCollection(points_of_interest=[poi1, poi2, ...])

    # Load POIs from a CSV file
    poi_collection.load_from_csv('poi_data.csv')
    ```
    """

    def __init__(self, points_of_interest=None, poi_records=None, poi_file_path=None):

        super().__init__(entities=points_of_interest, entity_records=poi_records, entity_file_path=poi_file_path)

    @property
    def poi_ids(self):
        # Get the unique identifiers of all Points of Interest in the collection.
        return [s.id for s in self.entities]

    @property
    def points_of_interest(self):
        # Get the list of PointOfInterest instances in the collection.
        return self.entities

    def add_entity(self, poi):
        # Add a PointOfInterest instance to the collection.
        if not isinstance(poi, PointOfInterest):
            raise TypeError(f'entity must be an instance of {PointOfInterest}')

        super().add_entity(entity=poi)

    def load_from_records(self, poi_records):
        """
        Load PointOfInterest instances from a list of dictionaries representing POI records.

        Parameters:
        - poi_records (list): List of dictionaries representing POI records.
        """
        for row in poi_records:
            poi = PointOfInterest.from_dict(row)
            self.add_entity(poi)

    def get_not_connected(self, only_fiber=False):
        """
        Get a collection of PointOfInterest instances that are not connected.

        Parameters:
        - only_fiber (bool): If True, only return PointOfInterest instances that are not connected via fiber.

        Returns:
        - PointOfInterestCollection: A collection of PointOfInterest instances that are not connected.
        """
        if only_fiber is True:
            not_connected_pois = [poi for poi in self.entities if (poi.is_connected is False) or (poi.connectivity_type != 'fiber')]
        else:
            not_connected_pois = [poi for poi in self.entities if poi.is_connected is False]

        return PointOfInterestCollection(points_of_interest=not_connected_pois)

    def get_connected(self, only_fiber=False):
        """
        Get a collection of PointOfInterest instances that are connected.

        Parameters:
        - only_fiber (bool): If True, only return PointOfInterest instances that are connected via fiber.

        Returns:
        - PointOfInterestCollection: A collection of PointOfInterest instances that are connected.
        """
        if only_fiber is True:
            connected_pois = [poi for poi in self.entities if (poi.is_connected is True) and (poi.connectivity_type == 'fiber')]
        else:
            connected_pois = [poi for poi in self.entities if poi.is_connected is True]

        return PointOfInterestCollection(points_of_interest=connected_pois)
