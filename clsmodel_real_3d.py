from __future__ import print_function
import gtsam
# Fake classifier model import
import libNewFactor_dim3, libNewFactor_dim3_rand_2, libNewFactor_dim3_rand_3

class ClsModel:

    def __init__(self, addresses):
        # Initializing the classifier models, one per class
        libNewFactor_dim3.initialize_python_objects(addresses[0], addresses[1])
        libNewFactor_dim3_rand_2.initialize_python_objects(addresses[2], addresses[3])
        libNewFactor_dim3_rand_3.initialize_python_objects(addresses[4], addresses[5])
        self.ClsModelCore = [libNewFactor_dim3, libNewFactor_dim3_rand_2, libNewFactor_dim3_rand_3]

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

        for obj in new_da_realization:

            obj = int(obj)

            for obj_index in class_realization:
                if obj == obj_index[0]:
                    obj_class = obj_index[1]
                    obj_class = int(obj_class)

            measurement[measurement_number] = list(measurement[measurement_number])
            graph.add(self.ClsModelCore[obj_class - 1].ClsModelFactor3(self.X(path_length), self.XO(obj),
                                                                               measurement[measurement_number]))
        #
        # for realization in new_da_realization:
        #     realization = int(realization)
        #     graph.add(self.ClsModelCore[class_realization[new_da_realization[measurement_number]-1]-1].
        #               ClsModelFactor3(self.X(path_length), self.XO(realization), measurement[measurement_number]))
            measurement_number += 1

            # Initialization of object of the realization
            # TODO: CHANGE TO SOMETHING MORE ACCURATE / LESS BASIC
            if initial.exists(self.XO(obj)) is False:
                initial.insert(self.XO(obj), gtsam.Pose3(gtsam.Pose2(0.0, 0.0, 0.0)))

    def log_pdf_value(self, measurement, cls, point_x, point_xo):
        """

        :type point_x: gtsam.Pose3
        :type point_xo: gtsam.Pose3
        :type measurement: gtsam.Pose3
        """

        # Create an auxilary factor and compute his log-pdf value (without the normalizing constant)
        aux_factor = self.ClsModelCore[cls - 1].ClsModelFactor3(1, 2, measurement)
        poses = gtsam.Values()
        poses.insert(1, point_x)
        poses.insert(2, point_xo)
        # return self.ClsModelCore[cls-1].error_out(point_xo.between(point_x), measurement)

        return aux_factor.error(poses)


