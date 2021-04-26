#ifndef JOINTPOSELAMBDAFACTORFAKE2D_H_INCLUDED
#define JOINTPOSELAMBDAFACTORFAKE2D_H_INCLUDED

using std::cout;
using std::endl;

// Factor code
namespace gtsam
{

void initialize_python_objects();

// PYTHON INTERPRETER CLASS DECLARATION
class PythonInterpreterWrapper
{
    class ClassifRegressor
    {

    public:
        ClassifRegressor();
        ClassifRegressor(const ClassifRegressor&)=delete;
        ClassifRegressor& operator=(const ClassifRegressor&)=delete;
        ~ClassifRegressor(){}

        std::vector<double> net_output_1(std::vector<double> relative_position);
        std::vector<double> net_output_2(std::vector<double> relative_position);
        std::vector<double> expectation(std::vector<double> relative_position, std::vector<double> prev_lambda, std::vector<double> model_input_exp);
    };

    class RRegressor
    {

    public:
        RRegressor();
        RRegressor(const RRegressor&)=delete;
        RRegressor& operator=(const RRegressor&)=delete;
        ~RRegressor(){}


        std::vector<std::vector<double> > net_output_R1(std::vector<double> relative_position);
        std::vector<std::vector<double> > net_output_R2(std::vector<double> relative_position);
        std::vector<std::vector<double> > covariance_matrix(std::vector<double> relative_position, std::vector<double> model_input_cov);
    };



    PythonInterpreterWrapper()
    {
        Py_Initialize();

        classif_regressor.reset();
        r_regressor.reset();
    }

public:
    static void initialize_python_interpreter()
    {
        interpreter.reset(new PythonInterpreterWrapper());
    }

    ~PythonInterpreterWrapper()
    {
        //Py_DECREF(object_ClsModel);
        cout << "Finalizing python" << endl;
        Py_Finalize();
        cout << "Distruct distruct" << endl;
    }

    void jacobian_matrix(std::vector<std::vector<double> >& H_1, std::vector<std::vector<double> >& H_2, std::vector<std::vector<double> >& H_3, std::vector<std::vector<double> >& H_4,
                        std::vector<double> model_input_exp, std::vector<double> model_input_cov, std::vector<double> prev_lambda,
                        std::vector<double> relative_position, double psi2earth, double theta2earth) const;

    static std::unique_ptr<PythonInterpreterWrapper> interpreter;

    std::unique_ptr<ClassifRegressor> classif_regressor;
    std::unique_ptr<RRegressor> r_regressor;

    static std::mutex interpreter_mtx;
};

std::unique_ptr<PythonInterpreterWrapper> PythonInterpreterWrapper::interpreter{nullptr};
std::mutex PythonInterpreterWrapper::interpreter_mtx;

// FACTOR CLASS DECLARATION
class JLPFactor: public gtsam::NonlinearFactor
{
    private:

        typedef JLPFactor This;
        typedef NonlinearFactor Base;
        Vector model_input_exp;
        Vector model_input_cov;
        mutable SharedNoiseModel noiseModel_;
        JLPFactor& operator=(const JLPFactor&) =delete;

    public:

        JLPFactor(){}
        JLPFactor(const Key& pose_x, const Key& pose_o, const Key& prev_lambda, const Key& current_lambda, const Vector& lgamma_exp, const Vector& lgamma_cov);
        virtual size_t dim() const { return 1; }
        Vector unwhitenedError(const Values& x, boost::optional<std::vector<Matrix>&> H = boost::none) const;
        Vector whitenedError(const Values& x, boost::optional<std::vector<Matrix>&> H = boost::none) const;
        virtual double error(const Values& x) const;
        virtual boost::shared_ptr<GaussianFactor> linearize(const Values& x) const;
        virtual void print(const std::string& s, const KeyFormatter& keyFormatter = DefaultKeyFormatter) const;
        virtual NonlinearFactor::shared_ptr clone() const;
};

// NET OUTPUTS CLASS DECLARATION
class AuxOutputsJLP
{
    public:
        static Vector net_output_1(const Vector& relative_Position_Py);
        static Vector net_output_2(const Vector& relative_Position_Py);
        static void predictions(Pose3& relativePose);
        static void error_prediction(Pose3& relativePose, const Vector& prev_lambda, const Vector& cur_lambda, const Vector& measurement_exp, const Vector& measurement_cov);
        static double error_out(Pose3& relativePose, const Vector& prev_lambda, const Vector& cur_lambda, const Vector& measurement_exp, const Vector& measurement_cov);
};

} // Namespace gtsam

#endif // JOINTPOSELAMBDAFACTORFAKE2D_H_INCLUDED
