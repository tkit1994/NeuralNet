#include <iostream>
#include "NeuralNet.hpp"
#include <cmath>
#define PI 3.141592653589

#define EIGEN_SUPERLU_SUPPORT
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE


int main(int argc, char** argv)
{
	auto steps = 100000l;
	if (argc > 1)
	{
		steps = atol(argv[1]);
	}
	std::cout << "loop for " << steps << " steps." << std::endl;
	neural_net<double> net{0.001};
	for (auto i = 0; i < steps; i++)
	{
		auto x = static_cast<double>(std::rand()) / RAND_MAX * PI ;
		net.set_x(x);
		net.forward_cpu();
		net.loss();
		net.backward_cpu();
		net.optimize();
		if (i % 1000 == 0)
		{
			std::cout <<"sin(" << x << ")=" << std::sin(x) << "\t" << net.get_y() << "\t" << net.get_loss() << std::endl;
		}
	}
	// Matrix<float, Dynamic, Dynamic> mat = Matrix<float, Dynamic, Dynamic>::Random(10000, 1000);
	// Matrix<float, Dynamic, Dynamic> result = mat * mat.transpose();
	return 0;
}
