/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * 
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file     gtsam_example.h
 * @brief    Example wrapper interface file for Python
 * @author   Varun Agrawal
 */

// This is an interface file for automatic Python wrapper generation.
// See gtsam.h for full documentation and more examples.

virtual class gtsam::NonlinearFactor;

// The namespace should be the same as in the c++ source code.
namespace gtsam {
#include <src/JointLambdaPoseFactorFake1D.h>

// Python Interpreter Activation, only that is needed
void initialize_python_objects();


// Factor class; the main show
virtual class JLPFactor: gtsam::NonlinearFactor {
	JLPFactor();
	JLPFactor(const gtsam::Key& pose_x, const gtsam::Key& pose_o, const gtsam::Key& prev_lambda, const gtsam::Key& current_lambda, const gtsam::Vector& lgamma_exp, const gtsam::Vector& lgamma_cov);
	NonlinearFactor::shared_ptr clone() const;
};

// Auxilary outputs class; needed for the Python model implementation
class AuxOutputsJLP {
	static Vector net_output_1(gtsam::Vector& relative_Position_Py);
        static Vector net_output_2(gtsam::Vector& relative_Position_Py);
        static void predictions(gtsam::Pose3& relativePose);
        static void error_prediction(gtsam::Pose3& relativePose, gtsam::Vector& prev_lambda, gtsam::Vector& cur_lambda, gtsam::Vector& measurement_exp, gtsam::Vector& measurement_cov);
        static double error_out(gtsam::Pose3& relativePose, gtsam::Vector& prev_lambda, gtsam::Vector& cur_lambda, gtsam::Vector& measurement_exp, gtsam::Vector& measurement_cov);
};

}  // namespace gtsam
