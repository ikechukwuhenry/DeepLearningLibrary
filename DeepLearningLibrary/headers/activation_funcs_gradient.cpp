// the gradient of the activation function
//  activation_funcs_gradient.cpp
//  DeepLearningLibrary
//  Created by IK on 09/04/2024.
// #include "activation_funcs_gradient.h"
#include <cmath>
#include <vector>
#include <stdexcept>
#include <iostream>
// This file contains implementations of the gradients of various activation functions used in deep learning.
// These gradients are essential for backpropagation in neural networks, allowing the model to learn from
// the errors during training.
// The functions include:
//  - Identity Gradient
//  - Binary Step Gradient
//  - ReLU Gradient
//  - Leaky ReLU Gradient
//  - PReLU Gradient
//  - Sigmoid Gradient
//  - Tanh Gradient 
//  - ELU Gradient
//  - Softplus Gradient
//  - Softsign Gradient
//  - Swish Gradient
//  - Mish Gradient
//  - GELU Gradient
//  - Gaussian Gradient
//  - Sinusoidal Gradient
//  - Softmax Gradient
//  These functions are implemented as standalone functions that take a single float input and return a float
//  output for scalar inputs, or a vector of floats for vector inputs (like softmax).
//  The functions are designed to be efficient and easy to use in deep learning applications.

#ifndef M_PI
#define M_PI  3.14159265358979323846 // Define M_PI if not already defined
#endif

// gradient of identity function
float identity_gradient(float x) {
    return 1.0f; // The gradient of the identity function is always 1
}   

// Gradient of the Binary Step function
float binary_step_gradient(float x) {
    // The binary step function is not differentiable at 0, but we can return
    // 0 for all other values since it is constant elsewhere.
    return (x == 0) ? 0.0f : 1.0f; // Return 0 at x=0, 1 otherwise
}

// Gradient of the ReLU function
// The gradient of ReLU is 1 for positive inputs and 0 for negative inputs.
float relu_gradient(float x) {
    return (x > 0) ? 1.0f : 0.0f;
}

//gradient of the Leaky ReLU function
// The gradient of Leaky ReLU is 1 for positive inputs and alpha (default 0.01) for negative inputs.
float leakyrelu_gradient(float x, float alpha = 0.01f) {
    return (x > 0) ? 1.0f : alpha; // Return alpha for negative inputs
}

// Gradient of the PReLU function
// The gradient of PReLU is 1 for positive inputs and alpha (default 0.01) for negative inputs.
// Note: alpha is a learnable parameter in PReLU,
// but we will use a fixed value for simplicity in this example.
float prelu_gradient(float x, float alpha = 0.01f) {
    return (x > 0) ? 1.0f : alpha; // Return alpha for negative inputs
}

// Gradient of the Sigmoid function
float sigmoid_gradient(float x) {
    float sig = 1.0f / (1.0f + std::exp(-x));
    return sig * (1.0f - sig);
}

// Gradient of the Tanh function
float tanh_gradient(float x) {
    float tanh_x = std::tanh(x);
    return 1.0f - tanh_x * tanh_x;
}

// Gradient of the ELU function
float elu_gradient(float x, float alpha = 1.0f) {
    if (x > 0) {
        return 1.0f;
    } else {
        return alpha * std::exp(x);
    }
}

// Gradient of the Softplus function
float softplus_gradient(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Gradient of the Softsign function
float softsign_gradient(float x) {
    float denom = 1.0f + std::abs(x);
    return 1.0f / (denom * denom);
}   

// Gradient of the Swish function
float swish_gradient(float x) {
    float sig = 1.0f / (1.0f + std::exp(-x));
    return sig + x * sig * (1.0f - sig);
}

// Gradient of the Mish function
float mish_gradient(float x) {
    float exp_x = std::exp(x);
    float tanh_part = std::tanh(std::log(1.0f + exp_x));
    float grad_tanh = 1.0f - tanh_part * tanh_part;
    return tanh_part + x * exp_x / (1.0f + exp_x) * grad_tanh;
}

// Gradient of the GELU function
float gelu_gradient(float x) {
    float tanh_part = std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * std::pow(x, 3)));
    float grad_tanh = 1.0f - tanh_part * tanh_part;
    return 0.5f * (1.0f + tanh_part) + 0.5f * x * std::sqrt(2.0f / M_PI) * (1.0f + 0.134145f * std::pow(x, 2)) * grad_tanh;
}

// Gradient of the Gaussian function
float gaussian_gradient(float x) {
    return -2.0f * x * std::exp(-std::pow(x, 2));
}

// Gradient of the Sinusoidal function
float sinusoidal_gradient(float x) {
    return std::cos(x);
}

// Gradient of the Softmax function
// The gradient of the softmax function is more complex and typically requires the Jacobian matrix.
// For simplicity, we will return a placeholder value here.
// In practice, you would compute the Jacobian matrix for the softmax function.
std::vector<std::vector<float>> softmax_gradient(const std::vector<float>& logits) {
    size_t n = logits.size();
    std::vector<std::vector<float>> jacobian(n, std::vector<float>(n, 0.0f));
    std::vector<float> softmax_values(n);       
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    for (float logit : logits) {
        sum_exp += std::exp(logit - max_logit);
    }
    for (size_t i = 0; i < n; ++i) {
        softmax_values[i] = std::exp(logits[i] - max_logit) / sum_exp;
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                jacobian[i][j] = softmax_values[i] * (1.0f - softmax_values[i]);
            } else {
                jacobian[i][j] = -softmax_values[i] * softmax_values[j];
            }
        }
    }
    return jacobian;
}   