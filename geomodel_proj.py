from __future__ import print_function
import gtsam
import libBearingRangePoseFactor



# Basic relative point measurement model
class GeoModel:

    def __init__(self, model_noise, adding_constant=0):

        # The instances differ from each other by model noise
        self.model_noise = model_noise
        self.adding_constant = adding_constant

        # Symbol initialization
        self.X = lambda i: int(gtsam.Symbol('x', i))
        self.XO = lambda j: int(gtsam.Symbol('o', j))

    def add_measurement(self, measurement, new_da_realization, graph, initial, da_realization):
        """

        :type graph: gtsam.NonlinearFactorGraph
        :type initial: gtsam.Values
        """
        # If measurement is Pose3, extract the Point3 out of it

        # Adding a factor per each measurement with corresponding data association realization
        measurement_number = 0
        path_length = len(da_realization)
        for realization in new_da_realization:
            realization = int(realization)
            graph.add(libBearingRangePoseFactor.BearingRangePoseFactor3(self.X(path_length), self.XO(realization),
                                                measurement[measurement_number][0], measurement[measurement_number][1],
                                                measurement[measurement_number][2], self.model_noise))
            measurement_number += 1

            # Initialization of object of the realization
            # TODO: CHANGE TO SOMETHING MORE ACCURATE / LESS BASIC
            if initial.exists(self.XO(realization)) is False:
                print('1121')
                initial.insert(self.XO(realization), gtsam.Pose3(gtsam.Pose2(0.0, 0.0, 0.0)))

        # Saving the data association realization in a list
        da_realization[path_length-1] = new_da_realization

    # Compute un-normlized log-likelihood of measurement
    def log_pdf_value(self, measurement, point_x, point_xo):
        """

        :type point_x: gtsam.Pose3
        :type point_xo: gtsam.Pose3
        :type measurement: gtsam.Pose3
        """

        # Create an auxilary factor and compute his log-pdf value (without the normalizing constant)
        aux_factor = libBearingRangePoseFactor.BearingRangePoseFactor3(1, 2, measurement[0], measurement[1], measurement[2], self.model_noise)
        poses = gtsam.Values()
        poses.insert(1, point_x)
        poses.insert(2, point_xo)

        return aux_factor.error(poses) - self.adding_constant