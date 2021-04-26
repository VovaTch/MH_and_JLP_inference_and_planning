// TODO: CONVERT STD::VECTOR TO EIGEN::VECTOR, CHECK IF MATRIX HAS TO BE THE SAME

// Including basic libraries
#include <iostream>
#include <ostream>
#include <functional>
#include <math.h>
#include <vector>
#include <memory>
#include <Python.h>
#include <stdexcept>
#include <string>
#include <Python.h>
#include <numeric>
#include <mutex>
//#include <Eigen/Dense>

// PyTorch C++ interface
// #include <torch/script.h>

// Including gtsam stuff
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/GaussianFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/slam/InitializePose3.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/geometry/Pose2.h>

// Including boost python to interface with python
#define BOOST_PYTHON_STATIC_LIB
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/python/stl_iterator.hpp>

// Including PyBind11 stuff
//#include <pybind11/pybind11.h>

// Include header
#include "JointLambdaPoseFactorFake1DConf.h"

using std::cout;
using std::endl;

//namespace py = boost::python;
//namespace py = pybind11;

// print vector
void print_vector(std::vector<double> vec)
{
    int vec_size = vec.size();
    for (int i=0; i<vec_size; i++)
        cout << vec[i] << " ";
    cout << endl << endl;
}

// print matrix
void print_matrix(std::vector<std::vector<double> > mat)
{
    int x_mat_size = mat.size();
    int y_mat_size = mat[0].size();
    for (int x=0; x<x_mat_size; x++)
    {
        for (int y=0; y<y_mat_size; y++)
            cout << mat[x][y] << " ";
        cout << endl;
    }
    cout << endl;

}

// Fix angles to be between -pi and pi
double angle_fix(double angle)
{
    while (angle > M_PI)
    {
        angle -= 2 * M_PI;
    }
    while (angle < -M_PI)
    {
        angle += 2 * M_PI;
    }
    return angle;
}

// Convert py objects to vectors
std::vector<double> listTupleToVector_Double(PyObject* incoming) {
	std::vector<double> data;
	if (PyTuple_Check(incoming)) {
		for(Py_ssize_t i = 0; i < PyTuple_Size(incoming); i++) {
			PyObject *value = PyTuple_GetItem(incoming, i);
			data.push_back( PyFloat_AsDouble(value) );
		}
	} else {
		if (PyList_Check(incoming)) {
			for(Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
				PyObject *value = PyList_GetItem(incoming, i);
				data.push_back( PyFloat_AsDouble(value) );
			}
		} else {
			throw std::logic_error("Passed PyObject pointer was not a list or tuple!");
		}
	}
	return data;
}

// Convert C++ vectors to lists
PyObject* vectorToList_Double(const std::vector<double> &data) {
  PyObject* listObj = PyList_New( data.size() );
	if (!listObj) throw std::logic_error("Unable to allocate memory for Python list");
	for (unsigned int i = 0; i < data.size(); i++) {
		PyObject *num = PyFloat_FromDouble(data[i]);
		if (!num) {
			Py_DECREF(listObj);
			throw std::logic_error("Unable to allocate memory for Python list");
		}
		PyList_SET_ITEM(listObj, i, num);
	}
	return listObj;
}

// Dot multiplication function
double dot_multiplication(std::vector<double> v_1, std::vector<double> v_2)
{
    double product = 0.0;
    int size_v_1 = v_1.size();
    if (v_1.size() != v_2.size())
        throw std::logic_error("Vectors are not at the same size");
    for (int i = 0; i != size_v_1; i++)
    {
        product = product + v_1[i] * v_2[i];
    }
    return product;
}

// Matrix/vector multiplication
std::vector<double> mat_vec_multiplication(std::vector<std::vector<double> > mat, std::vector<double> vec)
{
    int vec_length = vec.size();
    int mat_x_size = mat.size();
    int mat_y_size = mat[0].size();
    if (vec_length != mat_y_size)
        throw std::logic_error("Vector and matrix dimensions do not agree");

    //
    std::vector<double> mult_vector;
    double temp = 0.0;
    for (int i = 0; i != mat_x_size; i++)
    {
        temp = 0.0;
        for (int j = 0; j != mat_y_size; j++)
        {
            temp += mat[i][j] * vec[j];
        }
        mult_vector.push_back(temp);
    }

    return mult_vector;
}

// Matmul like python; matrix multiplication
std::vector<std::vector<double> > mat_mat_multiplication(std::vector<std::vector<double> > mat_1, std::vector<std::vector<double> > mat_2)
{
    int mat_1_num_rows = mat_1.size();
    int mat_1_num_cols = mat_1[0].size();
    int mat_2_num_rows = mat_2.size();
    int mat_2_num_cols = mat_2[0].size();

    if (mat_1_num_cols != mat_2_num_rows)
        throw std::logic_error("Matrix dimensions do not agree");

    std::vector<std::vector<double> > mult_matrix(mat_1_num_rows, std::vector<double>(mat_2_num_cols, 0.0));
    double temp = 0.0;
    for (int i = 0; i != mat_1_num_rows; i++)
    {
        for (int j = 0; j != mat_2_num_cols; j++)
        {
        temp = 0.0;
            for (int k = 0; k != mat_1_num_cols; k++)
            {
                temp += mat_1[i][k] * mat_2[k][j];
            }
        mult_matrix[i][j] = temp;
        }
    }

    return mult_matrix;
}

// Vector subtractions v2-v1
std::vector<double> vec_vec_subtraction(std::vector<double> vector_1, std::vector<double> vector_2)
{
    int vec_size_1 = vector_1.size();
    int vec_size_2 = vector_2.size();

    if (vec_size_1 != vec_size_2)
        throw std::logic_error("Vector dimensions for subtraction do not agree");

    std::vector<double> sub_vector(vec_size_1);
    for (int i = 0; i != vec_size_1; i++)
    {
        sub_vector[i] = vector_2[i] - vector_1[i];
    }

    return sub_vector;
}

// FACTOR CLASS ---------------------------------------------------------------------

// Functions to convert Pose to std::vector. Necessary to talk with Libtorch because it refuses directly to talk with gtsam
std::vector<double> Pose_to_vector(gtsam::Pose2 pose)
{
    return {pose.x(), pose.y(), pose.theta()};
}

std::vector<double> Pose_to_vector(gtsam::Pose3 pose)
{
    return {pose.x(), pose.y(), pose.z(), pose.rotation().yaw(), pose.rotation().pitch(), pose.rotation().roll()};
}

// Point 2 from a vector; necessary for the entire thing to work
gtsam::Point2 gtsam_point2_from_vector(std::vector<double> vec_vec)
{
    gtsam::Point2 point_lambda = gtsam::Point2(vec_vec[0], vec_vec[1]);
    return point_lambda;
}

// Function to convert a std::vector vector to gtsam::Vector
gtsam::Vector gtsam_vector_from_vector(std::vector<double> vec_vec)
{
    gtsam::Vector vec = gtsam::Vector::Map(vec_vec.data(), vec_vec.size());
    return vec;
}

// Function to convert a gtsam::Vector to std::vector
std::vector<double> vectorDouble_from_gtsam_vector(gtsam::Vector vec_vec)
{
    std::vector<double> vec(vec_vec.data(), vec_vec.data() + vec_vec.rows() * vec_vec.cols());
    return vec;
}

// Function to convert a std::vector<vector> matrix to gtsam::Matrix
gtsam::Matrix gtsam_matrix_from_vector(std::vector<std::vector<double> > vec_mat)
{
    int rows = vec_mat.size();
    int cols = vec_mat[0].size();
    gtsam::Matrix mat; mat.setZero(rows, cols);

    for (int i = 0; i != rows; i++)
    {
        for (int j = 0; j != cols; j++)
        {
            mat(i,j) = vec_mat[i][j];
        }
    }
    //std::cout << "Converted Gtsam Matrix:\n" << mat << "\n";
    return mat;
}

std::vector<double> Reverse_pose2(std::vector<double> pose2)
{
    return {-pose2[0], -pose2[1], pose2[2] + 3.1415};
}

std::vector<double> Reverse_pose3(std::vector<double> pose3)
{
    return {-pose3[0], -pose3[1], -pose3[2], pose3[3] + 3.1415, pose3[4] + 3.1415, pose3[5] + 3.1415};
}

// FACTOR CLASS ---------------------------------------------------------------------

using boost::assign::cref_list_of;

namespace gtsam
{

JLPFactor::JLPFactor(const Key& pose_x, const Key& pose_o, const Key& prev_lambda, const Key& current_lambda, const Vector& lgamma_exp, const Vector& lgamma_cov):
    Base(cref_list_of<4>(pose_x)(pose_o)(prev_lambda)(current_lambda)),
    model_input_exp(lgamma_exp),
    model_input_cov(lgamma_cov) {}

//Unwhitened error function;
Vector JLPFactor::unwhitenedError(const Values& x, boost::optional<std::vector<Matrix>&> H) const
{
    //std::cout << "unwhitenedError function\n";
    Pose3 pose1 = x.at<Pose3>(keys_[0]);
    Pose3 pose2 = x.at<Pose3>(keys_[1]);
    Vector prev_lam_vec = x.at<Vector>(keys_[2]);
    Vector cur_lam_vec = x.at<Vector>(keys_[3]);
    std::vector<double> model_input_exp_std = vectorDouble_from_gtsam_vector(model_input_exp);
    std::vector<double> model_input_cov_std = vectorDouble_from_gtsam_vector(model_input_cov);
    std::vector<double> prev_lam = {prev_lam_vec[0]};
    std::vector<double> cur_lam = {cur_lam_vec[0]};

    double psi2earth, theta2earth;

    //std::cout << "After pose extraction\n";
    Pose3 pose1to2(pose2.between(pose1));
    std::vector<double> pose1to2_vector = Pose_to_vector(pose1to2);
    if (typeid(pose2) == typeid(Pose2(0.0, 0.0, 0.0)))
    {
        std::vector<double> pose2_vector = Pose_to_vector(pose2);
        psi2earth = pose2_vector[2];
        theta2earth = 0;
    }
    else
    {
        std::vector<double> pose2_vector = Pose_to_vector(pose2);
        psi2earth = pose2_vector[2];
        theta2earth = pose2_vector[1];
    }

//        std::cout << "Unwhitened Error \n";
//        std::cout << "Relative Psi: " << atan2(pose1to2_vector[1], pose1to2_vector[0]) << "\n";
//        std::cout << "Psi to earth variable: " << psi2earth << "\n";

    std::vector<double> prediction = PythonInterpreterWrapper::interpreter->classif_regressor->expectation(pose1to2_vector, prev_lam, model_input_exp_std);
    std::vector<double> err;
    for (size_t i=0; i < prediction.size(); i++)
        err.push_back( - cur_lam[i] + prediction[i]);
    Vector err_gtsam = gtsam_vector_from_vector(err);
	//std::cout << "Unwhitened Error: " << err_gtsam << "\n";
    double R_dim = err.size();

	

    std::vector<std::vector<double> > H_1;
    std::vector<std::vector<double> > H_2;
    std::vector<std::vector<double> > H_3;
    std::vector<std::vector<double> > H_4;
    PythonInterpreterWrapper::interpreter->jacobian_matrix(H_1, H_2, H_3, H_4, model_input_exp_std, model_input_cov_std, prev_lam, pose1to2_vector, psi2earth, theta2earth);

    Matrix H_1_t = gtsam_matrix_from_vector(H_1);
    Matrix H_2_t = gtsam_matrix_from_vector(H_2);
    Matrix H_3_t = gtsam_matrix_from_vector(H_3);
    Matrix H_4_t = gtsam_matrix_from_vector(H_4);

        //std::cout << "H_1 matrix: " << H_1_t << std::endl;
        //std::cout << "H_2 matrix: " << H_2_t << std::endl;
        //std::cout << "H_3 matrix: " << H_3_t << std::endl;
        //std::cout << "H_4 matrix: " << H_4_t << std::endl;

    if (typeid(pose1) == typeid(gtsam::Pose2(0.0, 0.0, 0.0)))
    {
        if (H)
        {
            (*H)[0].resize(R_dim,3);
            (*H)[0] << H_1_t;

            (*H)[1].resize(R_dim,3);
            (*H)[1] << H_2_t;

            (*H)[2].resize(R_dim, R_dim);
            (*H)[2] << H_3_t;

            (*H)[3].resize(R_dim, R_dim);
            (*H)[3] << H_4_t;
        }
    }
    else
    {
        if (H)
        {
            (*H)[0].resize(R_dim,6);
            (*H)[0] << H_1_t;

            (*H)[1].resize(R_dim,6);
            (*H)[1] << H_2_t;

            (*H)[2].resize(R_dim, R_dim);
            (*H)[2] << H_3_t;

            (*H)[3].resize(R_dim, R_dim);
            (*H)[3] << H_4_t;
        }
    }
    //std::cout << "Breakpoint 6\n"; //...............
    return err_gtsam;
}

//Whitened error function;
Vector JLPFactor::whitenedError(const Values& x, boost::optional<std::vector<Matrix>&> H) const
{

    Pose3 pose1 = x.at<Pose3>(keys_[0]);
    Pose3 pose2 = x.at<Pose3>(keys_[1]);
    Vector prev_lam_vec = x.at<Vector>(keys_[2]);
    Vector cur_lam_vec = x.at<Vector>(keys_[3]);
    std::vector<double> model_input_exp_std = vectorDouble_from_gtsam_vector(model_input_exp);
    std::vector<double> model_input_cov_std = vectorDouble_from_gtsam_vector(model_input_cov);
    std::vector<double> prev_lam = {prev_lam_vec[0]};
    std::vector<double> cur_lam = {cur_lam_vec[0]};


    Pose3 pose1to2(pose2.between(pose1)); //TODO: Fix this pose mess
    std::vector<double> pose1to2_vector = Pose_to_vector(pose1to2);

    Vector err = unwhitenedError(x, H);

    //Some matrix sqrt of the covariance CHANGED WHEN NN IS READY
    std::vector<double> prediction = PythonInterpreterWrapper::interpreter->classif_regressor->expectation(pose1to2_vector, prev_lam, model_input_exp_std);
    auto cov_matrix = PythonInterpreterWrapper::interpreter->r_regressor->covariance_matrix(pose1to2_vector, model_input_cov_std);
    Matrix cov_matrix_t = gtsam_matrix_from_vector(cov_matrix);
    noiseModel::Gaussian::shared_ptr noise_matrix = noiseModel::Gaussian::Covariance(cov_matrix_t);
    noiseModel_ = noise_matrix;	

    if(H) {
        //std::cout << "H is existent\n";
        noiseModel_->WhitenSystem(*H, err);
    }
    else {
        //std::cout << "H is non-existent\n";
        noiseModel_->whitenInPlace(err);
    }
	//std::cout << "Whitened Error: " << err << std::endl;
    return err;
}

//Total error
double JLPFactor::error(const Values& x) const {
    //std::cout << "error function\n";
    //std::cout << "Error: " << whitenedError(x).squaredNorm() << std::endl;
    //std::cout << "total error: " << whitenedError(x).squaredNorm() << std::endl;
    return whitenedError(x).squaredNorm();
}

// Linearization function; (do we need it?)
boost::shared_ptr<GaussianFactor> JLPFactor::linearize(const Values& x) const
{
    //std::cout << "linearize function\n";
// Only linearize if the factor is active
    if (!this->active(x))
        return boost::shared_ptr<JacobianFactor>();

    Matrix A1, A2, A3, A4;
    std::vector<Matrix> A(this->size());
    //std::cout << "Breakpoint 11\n"; //...............
    Vector b = -whitenedError(x, A);
    A1 = A[0];
    A2 = A[1];
    A3 = A[2];
    A4 = A[3];
    //std::cout << "Breakpoint 12\n"; //...............

	//std::cout << "b in Ax=b: " << b << std::endl;
	//std::cout << "A in Ax=b: " << A1 << A2 << A3 << A4 << std::endl;

    std::vector<std::pair<Key, Matrix> > pair_vector;
    std::pair<Key, Matrix> temp;

    for (int i=0; i != 4; i++)
    {
        temp.first = keys_[i];
        temp.second = A[i];
        pair_vector.push_back(temp);
    }


    bool constrained = false;
            if(constrained)
                return GaussianFactor::shared_ptr(
                        new JacobianFactor(pair_vector, b, noiseModel::Constrained::All(1)));
            else
                return gtsam::GaussianFactor::shared_ptr(
                        new JacobianFactor(pair_vector, b, noiseModel::Unit::Create(b.size())));
    //std::cout << "Breakpoint 13\n"; //...............
}

//Print function TODO:::::: Do an appropriate one
void JLPFactor::print(const std::string& s, const KeyFormatter& keyFormatter) const
{
    Base::print(s + "JLP Factor", keyFormatter);
    std::cout << "Measurements: ( Expectation: ";
    std::cout << model_input_exp << std::endl;
    std::cout << "Covariance: ";
    std::cout << model_input_cov << std::endl;
    std::cout << ")\n";
}

// @return a deep copy of this factor
NonlinearFactor::shared_ptr JLPFactor::clone() const
{
    //cout << "clone function" << endl;
    return boost::static_pointer_cast<NonlinearFactor>(
        NonlinearFactor::shared_ptr(new This(*this)));
}




Vector AuxOutputsJLP::net_output_1(const Vector& relative_Position_Py)
{
    std::vector<double> relative_position = vectorDouble_from_gtsam_vector(relative_Position_Py);
    std::vector<double> output_exp = PythonInterpreterWrapper::interpreter->classif_regressor->net_output_1(relative_position);
    std::vector<std::vector<double> > output_rinf = PythonInterpreterWrapper::interpreter->r_regressor->net_output_R1(relative_position);

    std::vector<double> output_total = {output_exp[0], output_rinf[0][0]};
    Vector output_total_py = gtsam_vector_from_vector(output_total);
    return output_total_py;
}

Vector AuxOutputsJLP::net_output_2(const Vector& relative_Position_Py)
{
    std::vector<double> relative_position = vectorDouble_from_gtsam_vector(relative_Position_Py);
    std::vector<double> output_exp = PythonInterpreterWrapper::interpreter->classif_regressor->net_output_2(relative_position);
    std::vector<std::vector<double> > output_rinf = PythonInterpreterWrapper::interpreter->r_regressor->net_output_R2(relative_position);

    std::vector<double> output_total = {output_exp[0], output_rinf[0][0]};
    Vector output_total_py = gtsam_vector_from_vector(output_total);
    return output_total_py;
}

//Print predictions
void AuxOutputsJLP::predictions(Pose3& relativePose)
{
    std::vector<double> exp_prediction_1 = PythonInterpreterWrapper::interpreter->classif_regressor->net_output_1(Pose_to_vector(relativePose));
    std::vector<double> exp_prediction_2 = PythonInterpreterWrapper::interpreter->classif_regressor->net_output_2(Pose_to_vector(relativePose));
    std::vector<std::vector<double> > rinf_prediction_1 = PythonInterpreterWrapper::interpreter->r_regressor->net_output_R1(Pose_to_vector(relativePose));
    std::vector<std::vector<double> > rinf_prediction_2 = PythonInterpreterWrapper::interpreter->r_regressor->net_output_R2(Pose_to_vector(relativePose));


    std::cout << "Expectation 1 network output:\n";
    print_vector(exp_prediction_1);
    std::cout << "Expectation 2 network output:\n";
    print_vector(exp_prediction_2);
    std::cout << "\nR information matrix 1 network output:\n";
    print_matrix(rinf_prediction_1);
    std::cout << "\nR information matrix 2 network output:\n";
    print_matrix(rinf_prediction_2);
    std::cout << "\n";
}

//Print error prediction TODO: A good error predictor
void AuxOutputsJLP::error_prediction(Pose3& relativePose, const Vector& prev_lambda, const Vector& cur_lambda, const Vector& measurement_exp, const Vector& measurement_cov)
{
    std::vector<double> measurement_exp_vec = vectorDouble_from_gtsam_vector(measurement_exp);
    std::vector<double> measurement_cov_vec = vectorDouble_from_gtsam_vector(measurement_cov);
    std::vector<double> prev_lambda_vec = vectorDouble_from_gtsam_vector(prev_lambda);
    std::vector<double> cur_lambda_vec = vectorDouble_from_gtsam_vector(cur_lambda);
    std::vector<double> exp_prediction = PythonInterpreterWrapper::interpreter->classif_regressor->expectation(Pose_to_vector(relativePose), prev_lambda_vec, measurement_exp_vec);
    std::vector<std::vector<double> > cov_prediction = PythonInterpreterWrapper::interpreter->r_regressor->covariance_matrix(Pose_to_vector(relativePose), measurement_cov_vec);

    std::vector<double> err_vec;
    for (size_t i=0; i < exp_prediction.size(); i++)
        err_vec.push_back( + cur_lambda_vec[i] - exp_prediction[i]);

    std::vector<double> err_temp = mat_vec_multiplication(cov_prediction, err_vec);
    double err = dot_multiplication(err_temp, err_vec);

    std::cout << "Predicted error: " << err << "\n";
}

//Error out TODO: fix according to new model
double AuxOutputsJLP::error_out(Pose3& relativePose, const Vector& prev_lambda, const Vector& cur_lambda, const Vector& measurement_exp, const Vector& measurement_cov)
{
    std::vector<double> measurement_exp_vec = vectorDouble_from_gtsam_vector(measurement_exp);
    std::vector<double> measurement_cov_vec = vectorDouble_from_gtsam_vector(measurement_cov);
    std::vector<double> prev_lambda_vec = vectorDouble_from_gtsam_vector(prev_lambda);
    std::vector<double> cur_lambda_vec = vectorDouble_from_gtsam_vector(cur_lambda);
    std::vector<double> exp_prediction = PythonInterpreterWrapper::interpreter->classif_regressor->expectation(Pose_to_vector(relativePose), prev_lambda_vec, measurement_exp_vec);
    std::vector<std::vector<double> > cov_prediction = PythonInterpreterWrapper::interpreter->r_regressor->covariance_matrix(Pose_to_vector(relativePose), measurement_cov_vec);

    std::vector<double> err_vec;
    for (size_t i=0; i < exp_prediction.size(); i++)
        err_vec.push_back( + cur_lambda_vec[i] - exp_prediction[i]);

    std::vector<double> err_temp = mat_vec_multiplication(cov_prediction, err_vec);
    double err = dot_multiplication(err_temp, err_vec);


    return err;
}

} // Namespace gtsam

// CLASSIFIER MODEL CLASS, [TODO: CHANGE TO NEURAL NETWORK] -------------------------------------------


namespace gtsam
{

// Initialize the network class
void initialize_python_objects()
{
    PythonInterpreterWrapper::initialize_python_interpreter();
}


// Initialize expectation network
PythonInterpreterWrapper::ClassifRegressor::ClassifRegressor()
{
    std::unique_lock<std::mutex> lock(interpreter_mtx);
    std::cout << "Fake classification regressor is loaded " << std::endl;
}

// Classifier output NETWORK 1
std::vector<double> PythonInterpreterWrapper::ClassifRegressor::net_output_1(std::vector<double> relative_position)
{
    std::unique_lock<std::mutex> lock(interpreter_mtx);
    //std::cout << "net output function\n";
    // Convert vector to python list

    double psi = std::atan2(relative_position[1], relative_position[0]); // Might be needed later
    double theta;
    if (relative_position.size() == 3) // Might be needed later
    {
        theta = 0.0;
    }
    else
    {
        theta = std::atan2(relative_position[2], sqrt( pow(relative_position[1], 2) + pow(relative_position[0], 2)) );
    }
    psi = angle_fix(psi);
    theta = angle_fix(theta);

    //std::vector<double> cls_output_vector = {0.5 * cos(psi + theta) + 0.5};
    std::vector<double> cls_output_vector = {0.3 * cos(2 * psi + 2 * theta) + 0.3};

    return cls_output_vector;
}

// Classifier output NETWORK 2
std::vector<double> PythonInterpreterWrapper::ClassifRegressor::net_output_2(std::vector<double> relative_position)
{
    std::unique_lock<std::mutex> lock(interpreter_mtx);
    //std::cout << "net output function\n";
    // Convert vector to python list

    double psi = std::atan2(relative_position[1], relative_position[0]); // Might be needed later
    double theta;
    if (relative_position.size() == 3) // Might be needed later
    {
        theta = 0.0;
    }
    else
    {
        theta = std::atan2(relative_position[2], sqrt( pow(relative_position[1], 2) + pow(relative_position[0], 2)) );
    }
    psi = angle_fix(psi);
    theta = angle_fix(theta);

    //std::vector<double> cls_output_vector = {- 0.5 * cos(psi + theta) - 0.5};
    std::vector<double> cls_output_vector = {- 0.3 * cos(2 * psi + 2 * theta) - 0.3};

    return cls_output_vector;
}

// Initialize root information network
PythonInterpreterWrapper::RRegressor::RRegressor()
{
    std::unique_lock<std::mutex> lock(interpreter_mtx);
    std::cout << "Root information regressor is deployed " << std::endl;

}

// Rinf output NETWORK 1
std::vector<std::vector<double> >
PythonInterpreterWrapper::RRegressor::net_output_R1(std::vector<double> relative_position)
{
    std::unique_lock<std::mutex> lock(interpreter_mtx);
    //std::cout << "net output function\n";
    // Convert vector to python list

    double K = 2.0;

    double psi = std::atan2(relative_position[1], relative_position[0]); // Might be needed later
    double theta;
    if (relative_position.size() == 3) // Might be needed later
    {
        theta = 0.0;
    }
    else
    {
        theta = std::atan2(relative_position[2], sqrt( pow(relative_position[1], 2) + pow(relative_position[0], 2)) );
    }
    psi = angle_fix(psi);
    theta = angle_fix(theta);
    //std::vector<std::vector<double> > cls_output_matrix = {{K * (0.8 + 0.2 * cos(2 * (psi + theta) + 3.14))}};
    std::vector<std::vector<double> > cls_output_matrix = {{K * (0.7 + 0.3 * cos(psi + theta))}};
	//cout << "cls_output_1: " << cls_output_matrix[0][0] << std::endl;

    return cls_output_matrix;
}

// Rinf output NETWORK 2
std::vector<std::vector<double> >
PythonInterpreterWrapper::RRegressor::net_output_R2(std::vector<double> relative_position)
{
    std::unique_lock<std::mutex> lock(interpreter_mtx);
    //std::cout << "net output function\n";
    // Convert vector to python list

    double K = 2.0;

    double psi = std::atan2(relative_position[1], relative_position[0]); // Might be needed later
    double theta;
    if (relative_position.size() == 3) // Might be needed later
    {
        theta = 0.0;
    }
    else
    {
        theta = std::atan2(relative_position[2], sqrt( pow(relative_position[1], 2) + pow(relative_position[0], 2)) );
    }
    psi = angle_fix(psi);
    theta = angle_fix(theta);
    //std::vector<std::vector<double> > cls_output_matrix = {{K * (0.8 + 0.2 * cos(2 * (psi + theta) + 3.14))}};
    std::vector<std::vector<double> > cls_output_matrix = {{K * (0.7 - 0.3 * cos(psi + theta))}};
	//cout << "cls_output_2: " << cls_output_matrix[0][0] << std::endl;

    return cls_output_matrix;
}

// Computes the expectation of the model
std::vector<double>
PythonInterpreterWrapper::ClassifRegressor::expectation(std::vector<double> relative_position, std::vector<double> prev_lambda, std::vector<double> model_input_exp)
{
    //std::cout << "Expectation Function\n";
    //double default_value = 0.0;
	//print_vector(relative_position);
    std::vector<std::vector<double> > R1_matrix = PythonInterpreterWrapper::interpreter->r_regressor->net_output_R1(relative_position);
    std::vector<std::vector<double> > R2_matrix = PythonInterpreterWrapper::interpreter->r_regressor->net_output_R2(relative_position);
    std::vector<double> Exp1_vector = net_output_1(relative_position);
    std::vector<double> Exp2_vector = net_output_2(relative_position);
    int R_dim = 1;

    Matrix E_R1_matrix(R_dim, R_dim), E_R2_matrix(R_dim, R_dim),
        E_cov_matrix(R_dim, R_dim), Phi_1(R_dim, R_dim),
        E_inf1_matrix(R_dim, R_dim), E_inf2_matrix(R_dim, R_dim), E_inf_matrix(R_dim, R_dim);

    Vector E_Exp1_vector(R_dim), E_Exp2_vector(R_dim), E_model_exp_input(R_dim), Phi_2(R_dim), E_prev_lambda(R_dim);

    for (auto i = 0; i != R_dim; i++)
    {
        for (auto j = 0; j != R_dim; j++)
        {
            E_R1_matrix(i,j) = R1_matrix[i][j];
            E_R2_matrix(i,j) = R2_matrix[i][j];
        }
        E_Exp1_vector(i) = Exp1_vector[i];
        E_Exp2_vector(i) = Exp2_vector[i];
        E_model_exp_input(i) = model_input_exp[i];
        E_prev_lambda(i) = prev_lambda[i];
    }
	

    E_inf1_matrix = E_R1_matrix.transpose() * E_R1_matrix;
    E_inf2_matrix = E_R2_matrix.transpose() * E_R2_matrix;
    Vector temp_vec1 = E_Exp1_vector.transpose() * E_inf1_matrix;
    Vector temp_vec2 = E_Exp2_vector.transpose() * E_inf2_matrix;

    for (int i = 0; i != R_dim; i++)
    {
        Phi_1(0, i) = temp_vec1(i) - temp_vec2[i];
    }

    Phi_2(0) = (temp_vec1.transpose() * E_Exp1_vector - temp_vec2.transpose() * E_Exp2_vector)(0);

    Vector E_exp_out = E_prev_lambda + Phi_1 * E_model_exp_input - 0.5 * Phi_2;
	
	//cout << "Phi_1: " << Phi_1 << endl;
	//cout << "Phi_2: " << Phi_1 << endl;
	//cout << "E(lG): " << E_model_exp_input << endl;
    //cout << "E(lL): " << E_exp_out << endl;
    std::vector<double> expectation_output = {E_exp_out[0]};

    return expectation_output;
}


// Covariance matrix; convert to eigen, then back
std::vector<std::vector<double> >
PythonInterpreterWrapper::RRegressor::covariance_matrix(std::vector<double> relative_position, std::vector<double> model_cov_input)
{
    //std::cout << "covariance function\n";
    double default_value = 0.0;
    double epsilon = 0.001;
    std::vector<std::vector<double> > R1_matrix = net_output_R1(relative_position);
    std::vector<std::vector<double> > R2_matrix = net_output_R2(relative_position);
    std::vector<double> Exp1_vector = PythonInterpreterWrapper::interpreter->classif_regressor->net_output_1(relative_position);
    std::vector<double> Exp2_vector = PythonInterpreterWrapper::interpreter->classif_regressor->net_output_2(relative_position);
    int R_dim = Exp1_vector.size();

    Matrix E_R1_matrix(R_dim, R_dim), E_R2_matrix(R_dim, R_dim),
        E_cov_matrix(R_dim, R_dim), Phi_1(R_dim, R_dim),
        E_inf1_matrix(R_dim, R_dim), E_inf2_matrix(R_dim, R_dim), E_inf_matrix(R_dim, R_dim), E_model_cov_input(R_dim, R_dim);

    Vector E_Exp1_vector(R_dim), E_Exp2_vector(R_dim);

    for (auto i = 0; i != R_dim; i++)
    {
        for (auto j = 0; j != R_dim; j++)
        {
            E_R1_matrix(i,j) = R1_matrix[i][j];
            E_R2_matrix(i,j) = R2_matrix[i][j];
        }
        E_Exp1_vector[i] = Exp1_vector[i];
        E_Exp2_vector[i] = Exp2_vector[i];
    }

    E_model_cov_input(0, 0) = model_cov_input[0];

    E_inf1_matrix = E_R1_matrix.transpose() * E_R1_matrix;
    E_inf2_matrix = E_R2_matrix.transpose() * E_R2_matrix;
    Vector temp_vec1 = E_Exp1_vector.transpose() * E_inf1_matrix;
    Vector temp_vec2 = E_Exp2_vector.transpose() * E_inf2_matrix;

    for (int i = 0; i != R_dim; i++)
    {
        Phi_1(0, i) = temp_vec1(i) - temp_vec2[i];
    }

    gtsam::Matrix E_cov_safeguard(R_dim, R_dim);
    E_cov_safeguard(0, 0) = epsilon;

    E_cov_matrix = Phi_1 * E_model_cov_input * Phi_1.transpose() + E_cov_safeguard;
    //cout << "Cov(lL) matrix: " << E_cov_matrix << endl;
    //cout << "Phi1 matrix: " << Phi_1 << endl;
	
    std::vector<std::vector<double> > cov_matrix(R_dim, std::vector<double> (R_dim, default_value));

    for (auto i = 0; i != R_dim; i++)
    {
        for (auto j = 0; j != R_dim; j++)
        {
            cov_matrix[i][j] = E_cov_matrix(i,j);
        }
    }

	//cout << "E_cov_matrix: " << cov_matrix[0][0] << endl;

    return cov_matrix;
}

// Perform Cholesky decomposition of 2x2 matrix
std::vector<std::vector<double> > cholesky_2x2(std::vector<std::vector<double> > matrix)
{
    std::vector<std::vector<double> > matrix_output = {{0.0, 0.0}, {0.0, 0.0}};
    matrix_output[0][0] = std::sqrt(matrix[0][0]);
    matrix_output[0][1] = matrix[0][1] / matrix_output[0][0];
    matrix_output[1][1] = std::sqrt(matrix[1][1] - matrix_output[0][1] * matrix_output[0][1]);

    return matrix_output;
}

// Compute inverse of 2x2 matrix
std::vector<std::vector<double> > inverse_2x2(std::vector<std::vector<double> > matrix)
{
    std::vector<std::vector<double > > inv_matrix = {{0.0, 0.0}, {0.0, 0.0}};
    double det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    inv_matrix[0][0] = matrix[1][1] / det;
    inv_matrix[0][1] = - matrix[0][1] / det;
    inv_matrix[1][0] = - matrix[1][0] / det;
    inv_matrix[1][1] = matrix[0][0] / det;

    return inv_matrix;
}



void PythonInterpreterWrapper::jacobian_matrix(std::vector<std::vector<double> >& H_1, std::vector<std::vector<double> >& H_2,
                                                std::vector<std::vector<double> >& H_3, std::vector<std::vector<double> >& H_4,
                                                std::vector<double> model_input_exp, std::vector<double> model_input_cov, std::vector<double> prev_lambda,
                                                std::vector<double> relative_position, double psi2earth, double theta2earth) const
{

    // TODO: IMPLEMENT CHOLESKY DECOMPOSITION FROM SAMPLE MATRIX
    double default_value = 0.0;
    double inf_der_slow = 0.002;
    // Variable initialization
    double psi, theta, rad, d_psi = 0.01, d_theta = 0.01;
    // double d_theta_d_x, d_theta_d_y, d_theta_d_z;

    std::vector<std::vector<double> > H_1_num {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    std::vector<std::vector<double> > H_2_num {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    H_3 = {{1.0}};
    H_4 = {{-1.0}};

    psi = std::atan2(relative_position[1], relative_position[0]);

    if (relative_position.size() == 3)
    {
        theta = 0.0;
    }
    else
    {
        theta = std::atan2(relative_position[2], sqrt( pow(relative_position[1], 2) + pow(relative_position[0], 2)) );
    }

    psi = angle_fix(psi);
    theta = angle_fix(theta);

    // Euler angle rotations
    std::vector<std::vector<double> > rot2earth
        {{std::cos(psi2earth), std::cos(theta2earth) * std::sin(psi2earth), std::sin(theta2earth) * std::sin(psi2earth)},
        {- std::sin(psi2earth), std::cos(theta2earth) * std::cos(psi2earth), std::sin(theta2earth) * std::cos(psi2earth)},
        {0.0, - std::sin(theta2earth), std::cos(theta2earth)}};

    std::vector<std::vector<double> > rot2obj
        {{std::cos(psi2earth),- std::sin(psi2earth),0.0},
        {std::cos(theta2earth) * std::sin(psi2earth),std::cos(theta2earth) * std::cos(psi2earth),- std::sin(theta2earth)},
        {std::sin(theta2earth) * std::sin(psi2earth),std::sin(theta2earth) * std::cos(psi2earth),std::cos(theta2earth)}};

    double d_psi_d_x = - relative_position[1] / ( pow(relative_position[1], 2) + pow(relative_position[0], 2) );
    double d_psi_d_y = + relative_position[0] / ( pow(relative_position[1], 2) + pow(relative_position[0], 2) );
    double d_psi_d_z = 0.0;

    double d_psi_d_xr = d_psi_d_x * rot2earth[0][0] + d_psi_d_y * rot2earth[1][0] + d_psi_d_z * rot2earth[2][0];
    double d_psi_d_yr = d_psi_d_x * rot2earth[0][1] + d_psi_d_y * rot2earth[1][1] + d_psi_d_z * rot2earth[2][1];
    // double d_psi_d_zr = d_psi_d_x * rot2earth[0][2] + d_psi_d_y * rot2earth[1][2] + d_psi_d_z * rot2earth[2][2]; MAY BE NEEDED LATER

    // double d_theta_d_xr, d_theta_d_yr, d_theta_d_zr;

    rad = std::pow(relative_position[0]*relative_position[0] + relative_position[1]*relative_position[1] + relative_position[2]*relative_position[2], 0.5);
    // Position with small pertubations
    std::vector<double> relative_position_d_psi = { rad * cos(psi + d_psi) * cos(theta) , rad * sin(psi + d_psi) * cos(theta) , rad * sin(theta) };
    std::vector<double> relative_position_d_theta = { rad * cos(psi) * cos(theta + d_theta) , rad * sin(psi) * cos(theta + d_theta) , rad * sin(theta + d_theta) };

    std::vector<double> output = classif_regressor->expectation(relative_position, prev_lambda, model_input_exp);
    std::vector<std::vector<double> > output_cov = r_regressor->covariance_matrix(relative_position, model_input_cov);
    std::vector<std::vector<double> > output_inf = {{1.0 / output_cov[0][0]}};
    std::vector<std::vector<double> > output_r = {{std::sqrt(output_inf[0][0])}};
    std::vector<std::vector<double> > output_r_inv = {{1.0 / output_r[0][0]}};
                       
    std::vector<double> output_d_psi = classif_regressor->expectation(relative_position_d_psi, prev_lambda, model_input_exp);
    std::vector<std::vector<double> > output_ppsi_cov = r_regressor->covariance_matrix(relative_position_d_psi, model_input_cov);
    std::vector<std::vector<double> > output_ppsi_inf = {{1.0 / output_ppsi_cov[0][0]}};
    std::vector<std::vector<double> > output_ppsi_r = {{std::sqrt(output_ppsi_inf[0][0])}};
//    std::vector<std::vector<double> > output_ppsi_r_inv = inverse_2x2(output_ppsi_r);

    std::vector<double> output_d_theta = classif_regressor->expectation(relative_position_d_theta, prev_lambda, model_input_exp);
    std::vector<std::vector<double> > output_ptheta_cov = r_regressor->covariance_matrix(relative_position_d_theta, model_input_cov);
    std::vector<std::vector<double> > output_ptheta_inf = {{1.0 / output_ptheta_cov[0][0]}};
    std::vector<std::vector<double> > output_ptheta_r = {{std::sqrt(output_ptheta_inf[0][0])}};
//    std::vector<std::vector<double> > output_ptheta_r_inv = inverse_2x2(output_ptheta_r);

    int R_dim = output.size();

    // Compute derivatives or input and R w.r.t to psi and theta
    std::vector<double> d_f_d_psi;
    for (int i = 0; i != R_dim; i++)
        d_f_d_psi.push_back(( output_d_psi[i] - output[i] ) / d_psi);
    std::vector<double> d_f_d_theta;
    for (int i = 0; i != R_dim; i++)
        d_f_d_theta.push_back(( output_d_theta[i] - output[i] ) / d_theta);

    // Setting up numerical steps
    std::vector<double> small_xr_change {0.001, 0.0, 0.0};
    std::vector<double> small_x_change = mat_vec_multiplication(rot2obj, small_xr_change);
    std::vector<double> relative_position_dx {relative_position[0] + small_x_change[0], relative_position[1] + small_x_change[1], relative_position[2] + small_x_change[2]};
    std::vector<double> output_px = classif_regressor->expectation(relative_position_dx, prev_lambda, model_input_exp);
    std::vector<std::vector<double> > output_px_cov = r_regressor->covariance_matrix(relative_position_dx, model_input_cov);
    std::vector<std::vector<double> > output_px_inf = {{1.0 / output_px_cov[0][0]}};
    std::vector<std::vector<double> > output_px_r = {{std::sqrt(output_px_inf[0][0])}};
//    std::vector<std::vector<double> > output_px_r_inv = inverse_2x2(output_px_r);

    std::vector<double> small_yr_change {0.0, 0.001, 0.0};
    std::vector<double> small_y_change = mat_vec_multiplication(rot2obj, small_yr_change);
    std::vector<double> relative_position_dy {relative_position[0] + small_y_change[0], relative_position[1] + small_y_change[1], relative_position[2] + small_y_change[2]};
    std::vector<double> output_py = classif_regressor->expectation(relative_position_dy, prev_lambda, model_input_exp);
    std::vector<std::vector<double> > output_py_cov = r_regressor->covariance_matrix(relative_position_dy, model_input_cov);
    std::vector<std::vector<double> > output_py_inf = {{1.0 / output_py_cov[0][0]}};
    std::vector<std::vector<double> > output_py_r = {{std::sqrt(output_py_inf[0][0])}};
//    std::vector<std::vector<double> > output_py_r_inv = inverse_2x2(output_py_r);


    std::vector<double> small_zr_change {0.0, 0.0, 0.001};
    std::vector<double> small_z_change = mat_vec_multiplication(rot2obj, small_zr_change);
    std::vector<double> relative_position_dz {relative_position[0] + small_z_change[0], relative_position[1] + small_z_change[1], relative_position[2] + small_z_change[2]};
    std::vector<double> output_pz = classif_regressor->expectation(relative_position_dz, prev_lambda, model_input_exp);
    std::vector<std::vector<double> > output_pz_cov = r_regressor->covariance_matrix(relative_position_dz, model_input_cov);
    std::vector<std::vector<double> > output_pz_inf = {{1.0 / output_pz_cov[0][0]}};
    std::vector<std::vector<double> > output_pz_r = {{std::sqrt(output_pz_inf[0][0])}};
//    std::vector<std::vector<double> > output_pz_r_inv = inverse_2x2(output_pz_r);


    std::vector<std::vector<double> > d_r_d_psi(R_dim, std::vector<double> (R_dim, default_value)),
                                        d_r_d_theta(R_dim, std::vector<double> (R_dim, default_value)),
                                        d_r_d_x(R_dim, std::vector<double> (R_dim, default_value)),
                                        d_r_d_y(R_dim, std::vector<double> (R_dim, default_value)),
                                        d_r_d_z(R_dim, std::vector<double> (R_dim, default_value));
    for (int i = 0; i != R_dim; i++)
    {
        for (int j = 0; j != R_dim; j++)
        {
            d_r_d_psi[i][j] = ( output_ppsi_r[i][j] - output_r[i][j] ) / d_psi;
            d_r_d_theta[i][j] = ( output_ptheta_r[i][j] - output_r[i][j] ) / d_theta;
            d_r_d_x[i][j] = ( output_px_r[i][j] - output_r[i][j] ) / small_xr_change[0];
            d_r_d_y[i][j] = ( output_py_r[i][j] - output_r[i][j] ) / small_yr_change[1];
            d_r_d_z[i][j] = ( output_pz_r[i][j] - output_r[i][j] ) / small_zr_change[2];
        }
    }

    std::vector<std::vector<double> > R_change_psi = mat_mat_multiplication(output_r_inv, d_r_d_psi);
    std::vector<std::vector<double> > R_change_theta = mat_mat_multiplication(output_r_inv, d_r_d_theta);
    std::vector<std::vector<double> > R_change_x = mat_mat_multiplication(output_r_inv, d_r_d_x);
    std::vector<std::vector<double> > R_change_y = mat_mat_multiplication(output_r_inv, d_r_d_y);
    std::vector<std::vector<double> > R_change_z = mat_mat_multiplication(output_r_inv, d_r_d_z);

    // Classifier score prediction error

    std::vector<double> prediction_error(R_dim);
    for (int i = 0; i != R_dim; i++)
        prediction_error[i] = - model_input_exp[i] + output[i];

    if (relative_position.size() == 3)
    {
//        d_theta_d_x = 0;
//        d_theta_d_y = 0;
//        d_theta_d_z = 0;
//        double d_theta_d_xr = 0;
//        double d_theta_d_yr = 0;
//        double d_theta_d_zr = 0;

        H_1 = {{0.0, 0.0, 0.0}};
        H_2 = {{0.0, 0.0, 0.0}};

        for (int i = 0; i != R_dim; i++)
        {
            H_1[i][0] = - d_f_d_psi[i] * d_psi_d_xr;
            H_2[i][0] = - H_1[i][0];

            H_1[i][1] = - d_f_d_psi[i] * d_psi_d_yr;
            H_2[i][1] = - H_1[i][1];

            H_2[i][2] = d_f_d_psi[i];
        }
    }

    else
    {
//        d_theta_d_x = ( relative_position[2] * relative_position[0] ) / ( ( pow(relative_position[0], 2) + pow(relative_position[1], 2) + pow(relative_position[2], 2) ) *
//                            sqrt(pow(relative_position[0], 2) + pow(relative_position[1], 2)));
//        d_theta_d_y = ( relative_position[2] * relative_position[1] ) / ( ( pow(relative_position[0], 2) + pow(relative_position[1], 2) + pow(relative_position[2], 2) ) *
//                            sqrt(pow(relative_position[0], 2) + pow(relative_position[1], 2)));
//        d_theta_d_z = ( sqrt(pow(relative_position[0], 2) + pow(relative_position[1], 2)) ) /
//                            ( pow(relative_position[0], 2) + pow(relative_position[1], 2) + pow(relative_position[2], 2) );

//        double d_theta_d_xr = d_theta_d_x * rot2earth[0][0] + d_theta_d_y * rot2earth[1][0] + d_theta_d_z * rot2earth[2][0];
//        double d_theta_d_yr = d_theta_d_x * rot2earth[0][1] + d_theta_d_y * rot2earth[1][1] + d_theta_d_z * rot2earth[2][1];
//        double d_theta_d_zr = d_theta_d_x * rot2earth[0][2] + d_theta_d_y * rot2earth[1][2] + d_theta_d_z * rot2earth[2][2];

        H_1 = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
        H_2 = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

        for (int i = 0; i < R_dim; i++)
        {
            H_1_num[i][1] = dot_multiplication(R_change_theta[i], prediction_error) * inf_der_slow;
            H_2_num[i][1] = ( d_f_d_theta[i] - dot_multiplication(R_change_theta[i], prediction_error) ) * inf_der_slow;
            H_1[i][1] = H_1_num[i][1];
            H_2[i][1] = H_2_num[i][1];

            H_1_num[i][2] = dot_multiplication(R_change_psi[i], prediction_error) * inf_der_slow;
            H_2_num[i][2] = ( d_f_d_psi[i] + dot_multiplication(R_change_psi[i], prediction_error) ) * inf_der_slow;
            H_1[i][2] = H_1_num[i][2];
            H_2[i][2] = H_2_num[i][2];

            H_1_num[i][3] = ( ( output_px[i] - output[i] ) / 0.001 + dot_multiplication(R_change_x[i], prediction_error) ) * inf_der_slow;
            H_2_num[i][3] = -H_1_num[i][3];
            H_1[i][3] = H_1_num[i][3];
            H_2[i][3] = H_2_num[i][3];

            H_1_num[i][4] = ( ( output_py[i] - output[i] ) / 0.001 + dot_multiplication(R_change_y[i], prediction_error) ) * inf_der_slow;
            H_2_num[i][4] = -H_1_num[i][4];
            H_1[i][4] = H_1_num[i][4];
            H_2[i][4] = H_2_num[i][4];

            H_1_num[i][5] = ( ( output_pz[i] - output[i] ) / 0.001 + dot_multiplication(R_change_z[i], prediction_error) ) * inf_der_slow;
            H_2_num[i][5] = -H_1_num[i][5];

            H_1[i][5] = H_1_num[i][5];
            H_2[i][5] = H_2_num[i][5];
        }
    }
	//std::cout << "H_1 matrix: ";
	//print_matrix(H_1);
	//std::cout << "H_2 matrix: ";
	//print_matrix(H_2);
}

}
