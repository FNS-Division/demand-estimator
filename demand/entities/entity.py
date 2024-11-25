import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from demand.utils import haversine_, calculate_haversine_for_pair
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances
from typing import Optional


class Entity:
    """
    Represents an entity with geographical coordinates.

    Attributes:
    - id (str): Identifier for the entity.
    - lat (float): Latitude of the entity.
    - lon (float): Longitude of the entity.
    - entity_type (str): Type of the entity (default is 'Unidentified').
    - additional_fields (dict): Additional fields associated with the entity.

    Methods:
    - from_dict(cls, data_dict): Class method to create an Entity instance from a dictionary.
    - get_point_geometry(): Returns the shapely Point geometry object for the entity.
    - get_utm_coordinates(): [Not implemented]
    - get_distance(other): Calculates the great circle distance to another Entity.
    - __eq__(other): Checks if two entities are equal based on their ids.
    - __hash__(): Computes the hash value based on the entity's id.
    - __repr__(): Returns a string representation of the Entity.
    """
    valid_lat_range = (-90, 90)
    valid_lon_range = (-180, 180)

    def __init__(self, id, lat, lon, entity_type='Unidentified', **kwargs):

        self.id = id
        self._lat = None
        self._lon = None
        self.entity_type = entity_type
        self.additional_fields = kwargs

        self.lat = lat
        self.lon = lon

    @property
    def lat(self):
        return self._lat

    @lat.setter
    def lat(self, value):
        if self._is_valid_lat(value):
            self._lat = value
        else:
            raise ValueError(f"Invalid latitude value: {value}")

    @property
    def lon(self):
        return self._lon

    @lon.setter
    def lon(self, value):
        if self._is_valid_lon(value):
            self._lon = value
        else:
            raise ValueError(f"Invalid longitude value: {value}")

    def get_nearest_from_lon_lat_pairs(self, lon_lat_pairs):
        closest_index = min(enumerate(lon_lat_pairs), key=lambda item: calculate_haversine_for_pair(self.lat, self.lon, item[1][1], item[1][0]))[0]
        return closest_index

    @classmethod
    def from_dict(cls, data_dict):
        return cls(**data_dict)

    def _is_valid_lat(self, lat):
        return self.valid_lat_range[0] <= lat <= self.valid_lat_range[1]

    def _is_valid_lon(self, lon):
        return self.valid_lon_range[0] <= lon <= self.valid_lon_range[1]

    def get_point_geometry(self):
        return Point(self._lon, self._lat)

    def get_utm_coordinates(self):
        pass

    def get_distance(self, other):
        if not isinstance(other, Entity):
            raise TypeError(f'other must be an instance of {Entity}')

        return haversine_(lats=[self._lat, other._lat], lons=[self._lon, other._lon])

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"Entity(id= {self.id}, lat= {self.lat}, lon= {self.lon}, attributes = {self.additional_fields})"


class EntityCollection:
    """
    Represents a collection of Entity instances.

    Attributes:
    - entities (list): List of Entity instances.

    Methods:
    - add_entity(entity): Adds an Entity to the collection.
    - load_entities(entities): Loads a list of entities into the collection.
    - load_from_records(entity_records): Loads entities from a list of records (dictionaries).
    - load_from_csv(file_path, **kwargs): Loads entities from a CSV file.
    - get_subset(ids): Returns a subset of entities based on provided ids.
    - get_entity_types(): Returns a list of unique entity types in the collection.
    - delete_entities(ids): Deletes entities with specified ids.
    - check_duplicates(): Checks if there are duplicate entities in the collection.
    - get_duplicates(): Returns a list of duplicate entity pairs.
    - get_nth_entity(index): Returns the nth entity in the collection.
    - get_entities_by_entity_type(entity_type): Returns entities of a specific entity type.
    - get_lat_lon_pairs(): Returns an array of latitude-longitude pairs for all entities.
    - get_lat_array(): Returns an array of latitude values for all entities.
    - get_lon_array(): Returns an array of longitude values for all entities.
    - get_geoseries(): Returns a GeoSeries containing Point geometries for all entities.
    - get_entity_distance_matrix(): Returns a matrix of distances between entities.
    - cluster_entities(n_clusters, random_state): Clusters entities using KMeans algorithm.
    - __contains__(entity_id): Checks if an entity with a given id is in the collection.
    - __len__(): Returns the number of entities in the collection.
    - __repr__(): Returns a string representation of the EntityCollection.
    """

    def __init__(self, entities: list = None, entity_records=None, entity_file_path=None):

        self.entities = []

        if entities is not None:
            self.load_entities(entities)

        if entity_records is not None:
            self.load_from_records(entity_records)

        if entity_file_path is not None:
            self.load_from_csv(entity_file_path)

    @property
    def _ids(self):
        return [s.id for s in self.entities]

    @property
    def data(self):
        return pd.DataFrame([entity.__dict__ for entity in self.entities])

    def add_entity(self, entity):

        if not isinstance(entity, Entity):
            raise TypeError(f'entity must be an instance of {Entity}')

        self.entities.append(entity)

    def load_entities(self, entities):
        for entity in entities:
            self.add_entity(entity)

    def load_from_records(self, entity_records):

        for row in entity_records:
            entity = Entity.from_dict(row)
            self.add_entity(entity)

    def load_from_csv(self, file_path, **kwargs):
        df = pd.read_csv(file_path, **kwargs)
        self.load_from_records(df.to_dict('records'))

    def get_subset(self, ids: list):
        subset_entities = [entity for entity in self.entities if entity.id in ids]
        return self.__class__(entities=subset_entities)

    def delete_entities(self, ids: list):
        self.entities = [entity for entity in self.entities if entity.id not in ids]

    def check_duplicates(self):
        return len(self.entities) != len(set(self.entities))

    def get_duplicates(self):
        duplicates = []
        for i, entity1 in enumerate(self.entities):
            for entity2 in self.entities[i + 1:]:
                if entity1 == entity2:
                    duplicates.append((entity1, entity2))
        return duplicates

    def get_nth_entity(self, index):
        if 0 <= index < len(self.entities):
            return self.entities[index]
        else:
            raise IndexError("Index out of range")

    def get_entities_by_entity_type(self, entity_type):
        matching_entities = [entity for entity in self.entities if entity.entity_type == entity_type]

        return EntityCollection(entities=matching_entities)

    def get_lat_lon_pairs(self):
        n = len(self.entities)
        lat_lon_pairs = np.empty((n, 2))

        for i, entity in enumerate(self.entities):
            lat_lon_pairs[i] = [entity.lat, entity.lon]

        return lat_lon_pairs

    def get_lat_array(self):
        return np.array([s.lat for s in self.entities])

    def get_lon_array(self):
        return np.array([s.lon for s in self.entities])

    def get_geoseries(self):
        lat_lon_pairs = self.get_lat_lon_pairs()
        lats, lons = lat_lon_pairs[:, 0], lat_lon_pairs[:, 1]

        return gp.GeoSeries(gp.points_from_xy(lons, lats), crs='EPSG:4326')

    def get_entity_distance_matrix(self):

        lat_lon_pairs = self.get_lat_lon_pairs()
        lats, lons = lat_lon_pairs[:, 0], lat_lon_pairs[:, 1]

        return haversine_(lats=lats, lons=lons)

    def get_entity_types(self):
        # Extract unique entity types from the entities in the collection
        unique_entity_types = set(entity.entity_type for entity in self.entities)

        return list(unique_entity_types)

    def cluster_entities(self,
                         n_clusters: Optional[int] = None,
                         max_distance: Optional[int] = None,
                         transmission_node_indices: Optional[list] = None,
                         constrained: bool = False,
                         random_state=0):
        """
        Perform clustering on entities. If constrained is True, it ensures that each cluster has at least one transmission node by merging clusters.
        As a result, the final number of clusters might be lower than n_clusters.
        Parameters:
        - n_clusters (int): The desired number of clusters.
        - max_distance (int): Determines the maximum distance of an entity before a new cluster is created
        - poi_indices (list): Indices of the entities that are POIs.
        - transmission_node_indices (list): Indices of the entities that are transmission nodes.
        - constrained (bool): Whether to enforce the transmission node constraint.
        - random_state (int): Random state for KMeans clustering.
        Returns:
        dict: A dictionary where keys are cluster labels, and values are lists of entities in each cluster.
        """

        if (n_clusters is None and max_distance is None) or (n_clusters is not None and max_distance is not None):
            raise ValueError("You must specify either n_clusters (KMeans) or max_distance (Hierarchical), but not both.")

        def _enforce_transmission_node_constraint(labels, all_points, transmission_node_indices):
            """
            Ensure each cluster has at least one transmission node.
            If a cluster doesn't have a transmission node, merge it with the nearest cluster that does.

            Parameters:
            - labels: array of cluster labels for all points
            - transmission_indices: indices of transmission nodes

            Returns:
            - labels: updated array of cluster labels
            """
            # Select all transmission node points
            transmission_points = all_points[transmission_node_indices]

            # Extract the cluster labels of transmission nodes
            transmission_node_labels = labels[transmission_node_indices]

            # Iterate through all cluster labels
            for i in range(max(labels) + 1):
                # Check if the current cluster doesn't contain a transmission node
                if i not in transmission_node_labels:
                    # Select all points in the current cluster
                    cluster_points = all_points[labels == i]

                    # Calculate pairwise distances between cluster points and transmission nodes
                    # This results in a matrix where each row represents a cluster point
                    # and each column represents a transmission node
                    distances = pairwise_distances(cluster_points, transmission_points)

                    # Find the index of the nearest transmission node to the cluster
                    nearest_transmission_node = np.argmin(distances.min(axis=0))

                    # Get the cluster label of the nearest transmission node to the cluster
                    nearest_cluster = labels[transmission_node_indices[nearest_transmission_node]]

                    # Merge the current cluster (without a transmission node)
                    # into the cluster containing the nearest transmission node
                    # This updates all points in the current cluster to have the new label
                    labels[labels == i] = nearest_cluster

            # Return the updated labels array
            return labels

        if max_distance:
            entities = self.entities
            n = len(entities)
            distance_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    lat1, lon1 = entities[i].lat, entities[i].lon
                    lat2, lon2 = entities[j].lat, entities[j].lon
                    distance = calculate_haversine_for_pair(lat1, lon1, lat2, lon2)
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance

            # Perform hierarchical clustering
            linked = linkage(squareform(distance_matrix), method='complete')

            # Form flat clusters based on the maximum distance threshold
            labels = fcluster(linked, max_distance, criterion='distance')

            # Labels are numbered from 1 to n, we want to start at 0 to aligne with the KMeans method
            labels -= 1

            if constrained:
                all_points = self.get_lat_lon_pairs()
                scaler = StandardScaler()
                all_points_scaled = scaler.fit_transform(all_points)

                if transmission_node_indices is None:
                    raise ValueError("You must specify transmission_node_indices to use the constrained parameter")

                labels = _enforce_transmission_node_constraint(labels, all_points_scaled, transmission_node_indices)

        else:
            # Prepare data
            all_points = self.get_lat_lon_pairs()

            # Step 1: Standardize the data
            scaler = StandardScaler()
            all_points_scaled = scaler.fit_transform(all_points)

            # Step 2: Initial clustering using k-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            labels = kmeans.fit_predict(all_points_scaled)

            if constrained:
                # Step 3: Ensure each cluster has at least one transmission node
                if transmission_node_indices is None:
                    raise ValueError("You must specify transmission_node_indices to use the constrained parameter")

                labels = _enforce_transmission_node_constraint(labels, all_points_scaled, transmission_node_indices)

        # Create clusters dictionary
        unique_labels = np.unique(labels)
        clusters = {label: [] for label in unique_labels}

        # Assign entities to clusters
        for idx, label in enumerate(labels):
            clusters[label].append(self.entities[idx])

        # Reformat cluster to ensure labels are properly numbered
        clusters = {i + 1: value for i, (key, value) in enumerate(clusters.items())}

        return clusters

    def __contains__(self, entity_id):
        return entity_id in self._ids

    def __len__(self):
        return len(self.entities)

    def __repr__(self):
        return f"{self.__class__.__name__}: {len(self.entities)} entities"

    def __iter__(self):
        return iter(self.entities)

    def __getitem__(self, index):
        return self.entities[index]
