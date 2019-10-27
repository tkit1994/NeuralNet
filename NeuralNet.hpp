#ifndef NEURALNET_HPP
#define NEURALNET_HPP
#include <Eigen/Core>
using namespace Eigen;

const int n1 = 100, n2 = 100, n3 = 1;

template <typename T>
class neural_net
{
	// Weights
	Matrix<T, n1, 1> w1_, dw1_;
	Matrix<T, n2, n1> w2_, dw2_;
	Matrix<T, n3, n2> w3_, dw3_;
	// bias
	Matrix<T, n1, 1> b1_, db1_;
	Matrix<T, n2, 1> b2_, db2_;
	Matrix<T, n3, 1> b3_, db3_;

	Matrix<T, Dynamic, Dynamic> z1_, dz1_, a1_, da1_, z2_, dz2_, a2_, da2_, z3_, dz3_, a3_, da3_;
	T lr_ = 0, x_ = 0, loss_ = 0, y_ = 0;
public:

	T get_loss() const;
	T get_y() const;
	T get_lr() const;
	void set_lr(const T& lr);
	T get_x() const;
	void set_x(const T& x);
	neural_net() = default;
	explicit neural_net(const T& lr)
		: lr_(lr)
	{
		param_init();
	};

	void param_init();
	void forward_cpu();
	void loss();
	void backward_cpu();
	void optimize();
	template <typename Derived>
	static Array<T, Dynamic, Dynamic> sigmoid(const ArrayBase<Derived>& a)
	{
		 auto result = 1 / (1 + exp(-a));
		 return result;
	}
};


template <typename T>
T neural_net<T>::get_loss() const
{
	return loss_;
}

template <typename T>
T neural_net<T>::get_y() const
{
	return y_;
}

template <typename T>
T neural_net<T>::get_lr() const
{
	return lr_;
}

template <typename T>
void neural_net<T>::set_lr(const T& lr)
{
	lr_ = lr;
}

template <typename T>
T neural_net<T>::get_x() const
{
	return x_;
}

template <typename T>
void neural_net<T>::set_x(const T& x)
{
	x_ = x;
}

template <typename T>
void neural_net<T>::param_init()
{
	w1_ = Matrix<T, n1, 1>::Random();
	w2_ = Matrix<T, n2, n1>::Random();
	w3_ = Matrix<T, n3, n2>::Random();
	b1_ = Matrix<T, n1, 1>::Random();
	b2_ = Matrix<T, n2, 1>::Random();
	b3_ = Matrix<T, n3, 1>::Random();
}

template <typename T>
void neural_net<T>::forward_cpu()
{
	z1_ = w1_ * x_ + b1_;
	a1_ = sigmoid(z1_.array());
	
	z2_ = w2_ * a1_ + b2_;
	a2_ = sigmoid(z2_.array());

	z3_ = w3_ * a2_ + b3_;
	a3_ = sigmoid(z3_.array());

	y_ = a3_(0, 0);
}

template <typename T>
void neural_net<T>::loss()
{
	loss_ = 1. / n3 * pow(y_ - sin(x_), 2);
}

template <typename T>
void neural_net<T>::backward_cpu()
{
	da3_ = 1 / n3 * 2 * (a3_.array() - sin(x_));
	dz3_ = a3_.array() * (1 - a3_.array()) * da3_.array();
	db3_ = dz3_;
	dw3_ = dz3_ * a2_.transpose();
	//
	da2_ = w3_.transpose() * dz3_;
	dz2_ = a2_.array() * (1 - a2_.array()) * da2_.array();
	db2_ = dz2_;
	dw2_ = dz2_ * a1_.transpose();

	da1_ = w2_.transpose() * dz2_;
	dz1_ = a1_.array() * (1 - a1_.array()) * da1_.array();
	db1_ = dz1_;
	dw1_ = dz1_ * x_;
}

template <typename T>
void neural_net<T>::optimize()
{
	w1_ -= lr_ * dw1_;
	b1_ -= lr_ * db1_;
	w2_ -= lr_ * dw2_;
	b2_ -= lr_ * db2_;
	w3_ -= lr_ * dw3_;
	b3_ -= lr_ * db3_;
}



#endif // NEURALNET_HPP
