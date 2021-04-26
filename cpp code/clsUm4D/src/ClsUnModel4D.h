#ifndef JOINTPOSELAMBDAFACTORFAKE2D_H_INCLUDED
#define JOINTPOSELAMBDAFACTORFAKE2D_H_INCLUDED

#include <torch/script.h>

using std::cout;
using std::endl;

// Factor code
namespace gtsam
{

void initialize_python_objects(std::string exp_path_1, std::string exp_path_2, std::string exp_path_3, std::string exp_path_4, std::string exp_path_5,
			       std::string r_path_1, std::string r_path_2, std::string r_path_3, std::string r_path_4, std::string r_path_5);

// PYTHON INTERPRETER CLASS DECLARATION
class PythonInterpreterWrapper
{
    class ClassifRegressor
    {

	torch::jit::script::Module module_exp_1;
        torch::jit::script::Module module_exp_2;
        torch::jit::script::Module module_exp_3;
	torch::jit::script::Module module_exp_4;
        torch::jit::script::Module module_exp_5;

    public:
        ClassifRegressor(std::string exp_model_str_1, std::string exp_model_str_2, std::string exp_model_str_3, std::string exp_model_str_4, std::string exp_model_str_5);
        ClassifRegressor(const ClassifRegressor&)=delete;
        ClassifRegressor& operator=(const ClassifRegressor&)=delete;
        ~ClassifRegressor(){}

        std::vector<double> net_output_1(std::vector<double> relative_position);
        std::vector<double> net_output_2(std::vector<double> relative_position);
        std::vector<double> net_output_3(std::vector<double> relative_position);
        std::vector<double> net_output_4(std::vector<double> relative_position);
        std::vector<double> net_output_5(std::vector<double> relative_position);
        std::vector<double> expectation(std::vector<double> relative_position, int candidate_class);
    };

    class RRegressor
    {

        torch::jit::script::Module module_rinf_1;
        torch::jit::script::Module module_rinf_2;
        torch::jit::script::Module module_rinf_3;
	torch::jit::script::Module module_rinf_4;
        torch::jit::script::Module module_rinf_5;

    public:
        RRegressor(std::string rinf_model_str_1, std::string rinf_model_str_2, std::string rinf_model_str_3, std::string rinf_model_str_4, std::string rinf_model_str_5);
        RRegressor(const RRegressor&)=delete;
        RRegressor& operator=(const RRegressor&)=delete;
        ~RRegressor(){}


        std::vector<std::vector<double> > net_output_R1(std::vector<double> relative_position);
        std::vector<std::vector<double> > net_output_R2(std::vector<double> relative_position);
        std::vector<std::vector<double> > net_output_R3(std::vector<double> relative_position);
	std::vector<std::vector<double> > net_output_R4(std::vector<double> relative_position);
        std::vector<std::vector<double> > net_output_R5(std::vector<double> relative_position);
        std::vector<std::vector<double> > covariance_matrix(std::vector<double> relative_position, int candidate_class);
    };



    PythonInterpreterWrapper(std::string exp_model_str_1, std::string exp_model_str_2, std::string exp_model_str_3, std::string exp_model_str_4, std::string exp_model_str_5,
                             std::string rinf_model_str_1, std::string rinf_model_str_2, std::string rinf_model_str_3, std::string rinf_model_str_4, std::string rinf_model_str_5)
    {
        Py_Initialize();

        classif_regressor.reset(new ClassifRegressor(exp_model_str_1, exp_model_str_2, exp_model_str_3, exp_model_str_4, exp_model_str_5));
        r_regressor.reset(new RRegressor(rinf_model_str_1, rinf_model_str_2, rinf_model_str_3, rinf_model_str_4, rinf_model_str_5));
    }

public:
    static void initialize_python_interpreter(std::string exp_model_str_1, std::string exp_model_str_2, std::string exp_model_str_3, std::string exp_model_str_4, std::string exp_model_str_5,
                                              std::string rinf_model_str_1, std::string rinf_model_str_2, std::string rinf_model_str_3, std::string rinf_model_str_4, std::string rinf_model_str_5)
    {
        interpreter.reset(new PythonInterpreterWrapper(exp_model_str_1, exp_model_str_2, exp_model_str_3, exp_model_str_4, exp_model_str_5, 
						       rinf_model_str_1, rinf_model_str_2, rinf_model_str_3, rinf_model_str_4, rinf_model_str_5));
    }

    ~PythonInterpreterWrapper()
    {
        //Py_DECREF(object_ClsModel);
        cout << "Finalizing python" << endl;
        Py_Finalize();
        cout << "Distruct distruct" << endl;
    }

    void jacobian_matrix(std::vector<std::vector<double> >& H_1, std::vector<std::vector<double> >& H_2, std::vector<double> model_input,
                                                std::vector<double> relative_position, double psi2earth, double theta2earth, int candidate_class) const;

    static std::unique_ptr<PythonInterpreterWrapper> interpreter;

    std::unique_ptr<ClassifRegressor> classif_regressor;
    std::unique_ptr<RRegressor> r_regressor;

    static std::mutex interpreter_mtx;
};

std::unique_ptr<PythonInterpreterWrapper> PythonInterpreterWrapper::interpreter{nullptr};
std::mutex PythonInterpreterWrapper::interpreter_mtx;

// FACTOR CLASS DECLARATION
// FACTOR CLASS DECLARATION
class ClsUModelFactor: public gtsam::NonlinearFactor
{
    private:

        typedef ClsUModelFactor This;
        typedef NonlinearFactor Base;
        Vector model_input;
        int candidate_class;
        mutable SharedNoiseModel noiseModel_;
        ClsUModelFactor& operator=(const ClsUModelFactor&) =delete;

    public:

        ClsUModelFactor(){}
        ClsUModelFactor(const Key& pose_x, const Key& pose_o, const Vector& model_input, int candidate_class);
        virtual size_t dim() const { return 1; }
        Vector unwhitenedError(const Values& x, boost::optional<std::vector<Matrix>&> H = boost::none) const;
        Vector whitenedError(const Values& x, boost::optional<std::vector<Matrix>&> H = boost::none) const;
        virtual double error(const Values& x) const;
        virtual boost::shared_ptr<GaussianFactor> linearize(const Values& x) const;
        virtual void print(const std::string& s, const KeyFormatter& keyFormatter = DefaultKeyFormatter) const;
        virtual NonlinearFactor::shared_ptr clone() const;
};

// NET OUTPUTS CLASS DECLARATION
class AuxOutputs
{
    public:
        static Vector net_output_1(const Vector& relative_Position_Py);
        static Vector net_output_2(const Vector& relative_Position_Py);
        static Vector net_output_3(const Vector& relative_Position_Py);
        static Vector net_output_4(const Vector& relative_Position_Py);
        static Vector net_output_5(const Vector& relative_Position_Py);
        static void predictions(Pose3& relativePose);
        static void error_prediction(Pose3& relativePose, const Vector& cpv, int candidate_class);
        static double error_out(Pose3& relativePose, const Vector& cpv, int candidate_class);
};

} // Namespace gtsam

#endif // JOINTPOSELAMBDAFACTORFAKE2D_H_INCLUDED
