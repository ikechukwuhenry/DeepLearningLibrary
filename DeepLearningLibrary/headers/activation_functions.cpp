//
//  activation_functions.cpp
//  DeepLearningLibrary
//
//  Created by IK on 09/04/2024.
//  This file contains implementations of various activation functions used in deep learning.
//  These functions are essential for introducing non-linearity into neural networks, allowing them to learn
//  complex patterns in data.
// they are culled from wikipedia equations https://en.wikipedia.org/wiki/Activation_function
// The functions include:
//  - Identity
//  - Binary Step
//  - ReLU (Rectified Linear Unit)
//  - Leaky ReLU        
//  - PReLU (Parametric ReLU)
//  - Sigmoid
//  - Tanh (Hyperbolic Tangent)
//  - ELU (Exponential Linear Unit)
//  - Softplus
//  - Softsign
//  - Swish
//  - Mish
//  - GELU (Gaussian Error Linear Unit)
//  - Gaussian
//  - Sinusoidal
//  These functions are implemented as standalone functions that take a single float input and return a float output.
//  The functions are designed to be efficient and easy to use in deep learning applications.
//  The code is written in C++ and uses the standard library for mathematical operations.
//  The functions can be used in various deep learning frameworks and libraries, such as TensorFlow 
#include <iostream>

#ifndef M_PI
#define M_PI  3.14159265358979323846 // Define M_PI if not already defined
#endif

int identity(int x)
{
    return x;
}

int binary_step(int x)
{
    if (x >= 0) {
        return 1;
    } else {
        return 0;
    }
}

// Equation :- A(x) = max(0,x). It gives an output x if x is positive and 0 otherwise.
// Output range:- [0, inf)
float relu(float x)
{
    return std::max(x, float(0.0));
}

// lerelu(x) =  x if x>0
// lerelu(x) = 0.01 * x if x<=0
float leakyrelu(float x)
{
//    if (x > 0) {
//        return x;
//    } else {
//        return 0.01 * x;
//    }
    return std::max(float(0.01 * x), x);
}

// PReLU (Parametric ReLU) is a variant of Leaky ReLU where alpha is a learnable parameter.
// Equation: A(x) = max(alpha * x, x)
// Output range: (-inf, inf)
// Note: alpha is a hyperparameter, usually set to 0.01
// Default value of alpha is 0.01, but it can be adjusted based on the model's performance.
// PReLU is similar to Leaky ReLU, but it allows the slope for negative inputs to be learned during training, rather than being fixed.
// This can lead to better performance in some cases, as the model can adapt the negative slope based on the data.
// PReLU is often used in deep learning models to improve convergence and performance, especially in
float prelu(float x, float alpha = 0.01f)
{
    return std::max(alpha * x, x);
}


// Equation : A = 1/(1 + e-x)
// Ouput range: 0 - 1
float sigmoid(float x)
{
    float value = float(1.0) + std::exp(-x);
    return float(1.0)/value;
}

// Tanhh it’s actually mathematically shifted version of the sigmoid function
// f(x) = tanh(x)
// Output range:- (-1, 1)
float dllib_tanh(float x)
{
    return std::tanh(x);
}

// elu(x) =  x if x>0
// elu(x) = alpha * (exp(x)-1) if x<0
// alpha is a hyperparameter, usually set to 1.0
// Output range: (-alpha, inf)
float elu(float x, float alpha)
{
    if (x > 0) {
        return x;
    } else {
        return alpha * (std::exp(x) - 1.0f);
    }
}

// Softplus function: A(x) = ln(1 + exp(x))
// Output range: (0, inf)
float softplus(float x, float alpha = 1.0f) {
    return std::log(1.0f + std::exp(x));
}

// Softsign function: A(x) = x / (1 + |x|)
// Output range: (-1, 1)
float softsign(float x) {
    return x / (1.0f + std::abs(x));
}

// Swish function: A(x) = x * sigmoid(x)
// Output range: (-inf, inf)
// aka Sigmoid-weighted Linear Unit (SiLU)
float swish(float x) {
    return x * sigmoid(x);
}

// Mish function: A(x) = x * tanh(ln(1 + exp(x)))
// Output range: (-inf, inf)
float mish(float x) {
    return x * std::tanh(std::log(1.0f + std::exp(x)));
}

// GELU (Gaussian Error Linear Unit) function
// GELU function: A(x) = 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))))
// Output range: (-inf, inf)
float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * std::pow(x, 3))));
}

// Gaussian function: A(x) = exp(-x^2)
// Output range: (0, 1)
float gaussian(float x) {
    return std::exp(-std::pow(x, 2));
}


// Sinusoidal function: A(x) = sin(x)
// Output range: (-1, 1)
float sinusoid(float x) {
    return std::sin(x);
}