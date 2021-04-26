cdef extern from "main.cpp":
	cdef cppclass LambdaPriorFactor
		LambdaPriorFactor(const gtsam::Key& plambda_key, PyObject* plambda_exp, gtsam::noiseModel::Diagonal::shared_ptr plambda_cov) except +
		gtsam::Vector unwhitenedError(const gtsam::Values& x, boost::optional<std::vector<gtsam::Matrix>&> H = boost::none) const
		gtsam::Vector whitenedError(const gtsam::Values& x, boost::optional<std::vector<gtsam::Matrix>&> H = boost::none) const
		virtual double error(const gtsam::Values& x) const
		virtual boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& x) const
		virtual void print(const std::string& s, const gtsam::KeyFormatter& keyFormatter = gtsam::DefaultKeyFormatter) const
		virtual gtsam::NonlinearFactor::shared_ptr clone() const
		
