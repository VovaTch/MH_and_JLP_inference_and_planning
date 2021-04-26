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

        // Expectation with a switch
        std::vector<double> expectation(std::vector<double> relative_position, int candidate_class);
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

        // Covariance with a switch
        std::vector<std::vector<double> > covariance_matrix(std::vector<double> relative_position, int candidate_class);
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

	AuxOutputs(){}
        static Vector net_output_1(const Vector& relative_Position_Py);
        static Vector net_output_2(const Vector& relative_Position_Py);
        static void predictions(Pose3& relativePose);
        static void error_prediction(Pose3& relativePose, const Vector& cpv, int candidate_class);
        static double error_out(Pose3& relativePose, const Vector& cpv, int candidate_class);
};

} // Namespace gtsam

#endif // JOINTPOSELAMBDAFACTORFAKE2D_H_INCLUDED

