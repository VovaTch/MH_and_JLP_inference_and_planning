from __future__ import print_function
import gtsam
# Fake classifier model import
import libNewFactor

class ClsModel:

    def __init__(self):
        # Initializing the classifier models, one per class
        self.ClsModelCore = [libNewFactor.ClsModelReal1(), libNewFactor.ClsModelReal1()]

        # Symbol initialization
        self.X = lambda i: int(gtsam.Symbol('x', i))
        self.XO = lambda j: int(gtsam.Symbol('o', j))

    def add_measurement(self, measurement, new_da_realization, graph, initial, da_realization, class_realization):
        """

        :type graph: gtsam.NonlinearFactorGraph
        :type initial: gtsam.Values
        """

        # Adding a factor per each measurement with corresponding data association realization
        measurement_number = 0
        path_length = len(da_realization) - 1
        for realization in new_da_realization:
            realization = int(realization)
            graph.add(libNewFactor.ClsModelFactor3(self.X(path_length), self.XO(realization),
                                                       self.ClsModelCore[class_realization[new_da_realization[measurement_number]-1]-1],
                                                       measurement[measurement_number]))
            measurement_number += 1

            # Initialization of object of the realization
            # TODO: CHANGE TO SOMETHING MORE ACCURATE / LESS BASIC
            if initial.exists(self.XO(realization)) is False:
                initial.insert(self.XO(realization), gtsam.Pose3(gtsam.Pose2(0.0, 0.0, 0.0)))

    def log_pdf_value(self, measurement, cls, point_x, point_xo):
        """

        :type point_x: gtsam.Pose3
        :type point_xo: gtsam.Pose3
        :type measurement: gtsam.Pose3
        """

        # Create an auxilary factor and compute his log-pdf value (without the normalizing constant)
        aux_factor = libNewFactor.ClsModelFactor3(1, 2, self.ClsModelCore[cls - 1], measurement)
        poses = gtsam.Values()
        poses.insert(1, point_x)
        poses.insert(2, point_xo)

        return aux_factor.error(poses)


