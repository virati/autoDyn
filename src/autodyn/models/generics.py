import typing

class network:
    def __init__(self):
        pass

    def set_connectivity(self, input_connectivity):
        pass

    def set_driftf(self, input_drift_function):
        pass

    def set_region_euclidean(self, input_region_coords):
        pass

    def set_tract_euclidean(self, input_tract_coords):
        pass

    def set_euclideans(self, input_anatomy):
        self.set_region_euclidean(input_anatomy['regions'])
        self.set_tract_euclidean(input_anatomy['tracts'])
